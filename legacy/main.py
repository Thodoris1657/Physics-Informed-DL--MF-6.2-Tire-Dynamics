
"""
================================================================================
  FSAE P26 - PINN Tire Discovery  (Magic Formula 6.2, PyTorch)
================================================================================

Given:
   * 4 steady-state 10.5 m skidpad telemetry runs
   * Calspan TTC raw data (pure lateral SA sweep, 2 loads, free-rolling)

Compute, per wheel:
   * Vertical load Fz  (P26 MATLAB LoadTransfer_v2, reproduced faithfully)
   * Slip angle alpha  (multibody-consistent kinematic projection)
   * Lateral force Fy  (learned MF6.2 lateral model, shared front/rear tire)

The MF6.2 coefficients are learnable PyTorch parameters.  Training minimises a
combined loss:

    L = w_ttc  * ||Fy_MF62(alpha,Fz) - Fy_TTC||    (primary supervision)
      + w_lat  * ||sum(Fy_wheels) - m*ay||         (lateral balance)
      + w_yaw  * ||Mz_wheels       - Iz*r_dot||    (yaw balance)
      + w_reg  * parameter regulariser

Phase 1:  Adam  (batch + cosine schedule)
Phase 2:  L-BFGS  (full batch refinement)

Outputs (in  ./pinn_p26_results/):
    mf62_coefficients.json     - final MF6.2 coefficient set
    mf62_coefficients.csv      - same, spreadsheet-friendly
    per_wheel_timeseries.csv   - full prediction per sample
    01_tire_curves.png         - Fy vs alpha, family of Fz levels
    02_tire_surface.png        - Fy(alpha,Fz) surface (front, rear)
    03_ttc_fit.png             - model vs Calspan raw
    04_fz_per_wheel.png        - Fz(t) all 4 wheels
    05_alpha_per_wheel.png     - alpha(t) all 4 wheels
    06_fy_per_wheel.png        - Fy(t) all 4 wheels
    07_vehicle_balance.png     - measured vs predicted ay / yaw moment
    08_kamm_circle.png         - friction envelope per wheel
    09_training_history.png    - loss curves
    SUMMARY.txt                - headline numbers
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.cluster.vq import kmeans2
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
#  §1  VEHICLE PARAMETERS  -  P26  (from the MATLAB paramsp26 struct)
# ============================================================================

@dataclass
class P26Params:
    # --- inertial ---
    NSM: float = 43         # non-suspended mass           [kg]
    SM:  float = 210.0        # suspended mass (+70 kg driver) [kg]
    hcog_NSM: float = 0.199   # NSM CoG height              [m]
    hcog_SM:  float = 0.275   # SM  CoG height              [m]
    Iz: float = 100.0         # yaw inertia                 [kg.m2]

    # --- geometry ---
    WD_tot:  float = 0.47     # front weight distribution   [-]
    WD_NSM:  float = 0.34     # NSM weight dist (front)     [-]
    WB:      float = 1.590    # wheelbase                   [m]
    Tf:      float = 1.220    # front track                 [m]
    Tr:      float = 1.230    # rear  track                 [m]
    Rw:      float = 0.2054   # unloaded tyre radius        [m]

    # --- suspension (for load transfer) ---
    hRC_f: float = 0.03286    # front roll-centre height    [m]  (32.86 mm)
    hRC_r: float = 0.02750    # rear  roll-centre height    [m]  (27.5 mm)
    roll_rate_deg_per_g: float = 0.5     # chassis roll gradient [deg/g]
    MB: float = 0.52          # mechanical balance (front)

    # --- aero ---
    AD_long:  float = 0.48    # aero distribution front     [-]
    clA_nom:  float = 4.70    # Cl*A                        [m2]
    CdA:      float = 1.90    # Cd*A                        [m2]
    roll_loss: float = 0.2743 # ClA loss per deg roll       [m2/deg]

    # --- environment ---
    rho: float = 1.225        # air density                 [kg/m3]
    g:   float = 9.81         # gravity                     [m/s2]

    # --- steering ---
    # Confirmed by physical consistency test (K_us regression across all 8 runs):
    # the 'steering' column is in STEERING WHEEL degrees.
    # Dividing by steer_ratio gives road-wheel angle in degrees.
    # IF the column were already road-wheel, the implied K_us would swing
    # from -12.3 to +10.3 °/g across runs (std=7.8°/g) — physically impossible.
    # Divided by 3.28 the K_us collapses to std=2.87°/g centred at -1.6°/g
    # (slight oversteer at limit — expected for FSAE at 1.2–1.4 g).
    steer_ratio: float = 3.28         # steering wheel → road-wheel [–]
    steer_is_roadwheel: bool = False  # column is STEERING WHEEL deg -> divide by steer_ratio

    @property
    def m_tot(self) -> float:
        return self.NSM + self.SM

    @property
    def lf(self) -> float:
        return self.WB * (1.0 - self.WD_tot)

    @property
    def lr(self) -> float:
        return self.WB * self.WD_tot


P26 = P26Params()


# ============================================================================
#  §2  LOAD TRANSFER  -  mirror of MATLAB LoadTransfer_v2
#      Vectorised and differentiable (pure tensor algebra).
# ============================================================================

class LoadTransfer(nn.Module):
    """
    Port of the MATLAB LoadTransfer_v2 function, vector-safe in torch.

    Inputs  (all tensors of shape [N] in SI units, ISO vehicle convention):
        Vx  - longitudinal velocity      [m/s]
        ax  - longitudinal acceleration  [m/s^2]  (body frame)
        ay  - lateral     acceleration   [m/s^2]  (body frame)

    Returns:
        Fz_FL, Fz_FR, Fz_RL, Fz_RR   [N]
    """

    def __init__(self, p: P26Params = P26):
        super().__init__()
        self.p = p

    def forward(self, Vx: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p
        m_tot = p.m_tot
        SM = p.SM
        NSM = p.NSM
        WD = p.WD_tot
        WD_NSM = p.WD_NSM
        g = p.g
        WB = p.WB
        a = p.WB * (1.0 - WD)        # distance to front axle  ( MATLAB 'a' )
        b = p.WB - a                  # distance to rear axle   ( MATLAB 'b' )
        hcog_SM  = p.hcog_SM
        hcog_NSM = p.hcog_NSM
        hRC_f = p.hRC_f
        hRC_r = p.hRC_r
        hRC_at_CoG = (WD * hRC_f + (1.0 - WD) * hRC_r)          # already meters
        # note: the MATLAB script divides this by 1000 again,
        #       producing a ~1e-5 number; we keep the physically correct value.
        MB = p.MB
        roll_rate = p.roll_rate_deg_per_g
        Tf = p.Tf
        Tr = p.Tr
        AD_long = p.AD_long
        AD_lat  = 0.5
        clA_nom = p.clA_nom
        rho = p.rho
        roll_loss = p.roll_loss

        # ---- static loads ----
        Fz_FR_s = 0.5 * m_tot * WD * g
        Fz_FL_s = 0.5 * m_tot * WD * g
        Fz_RR_s = 0.5 * m_tot * (1.0 - WD) * g
        Fz_RL_s = 0.5 * m_tot * (1.0 - WD) * g

        # ---- longitudinal weight transfer (NSM + SM split) ----
        LT_x_FR_NSM = -0.5 * (NSM * ax * hcog_NSM) / WB
        LT_x_FL_NSM = -0.5 * (NSM * ax * hcog_NSM) / WB
        LT_x_RR_NSM =  0.5 * (NSM * ax * hcog_NSM) / WB
        LT_x_RL_NSM =  0.5 * (NSM * ax * hcog_NSM) / WB
        LT_x_FR_SM  = -0.5 * (SM  * ax * hcog_SM ) / WB
        LT_x_FL_SM  = -0.5 * (SM  * ax * hcog_SM ) / WB
        LT_x_RR_SM  =  0.5 * (SM  * ax * hcog_SM ) / WB
        LT_x_RL_SM  =  0.5 * (SM  * ax * hcog_SM ) / WB

        # ---- Milliken total lateral weight transfer (p. 682) ----
        q = hcog_SM - hRC_at_CoG
        susp_roll_stif = SM * q / roll_rate - SM * q       # [Nm/deg]
        Kf = susp_roll_stif * MB           * 180.0 / math.pi   # [Nm/rad]
        Kr = susp_roll_stif * (1.0 - MB)   * 180.0 / math.pi
        KappaF = Kf - b * SM * q / WB
        KappaR = Kr - a * SM * q / WB
        NSM_f = NSM *  WD_NSM
        NSM_r = NSM * (1.0 - WD_NSM)

        LT_y_f = (SM / Tf * (q * KappaF / (Kf + Kr - SM * q) + WD * hRC_f)
                  + NSM_f / Tf * hcog_NSM) * ay
        LT_y_r = (SM / Tr * (q * KappaR / (Kf + Kr - SM * q) + WD * hRC_r)
                  + NSM_r / Tr * hcog_NSM) * ay

        # ---- aero vertical force (per wheel) ----
        # roll_loss is in [m²/deg]; first convert lateral acceleration to chassis
        # roll angle in degrees, then compute the effective ClA reduction.
        # roll_deg = |ay| / g * roll_rate_deg_per_g
        roll_deg = torch.abs(ay) / p.g * roll_rate   # [deg]
        clA = clA_nom - roll_loss * roll_deg          # [m²]
        # Standard aerodynamic downforce: F = 0.5 * rho * V² * Cl * A
        # The factor 0.5 was previously missing, causing a 2× overestimate.
        F_aero_f = AD_lat * AD_long         * 0.5 * rho * clA * Vx * Vx
        F_aero_r = AD_lat * (1.0 - AD_long) * 0.5 * rho * clA * Vx * Vx

        Fz_FR = Fz_FR_s + LT_x_FR_NSM + LT_x_FR_SM - LT_y_f + F_aero_f
        Fz_FL = Fz_FL_s + LT_x_FL_NSM + LT_x_FL_SM + LT_y_f + F_aero_f
        Fz_RR = Fz_RR_s + LT_x_RR_NSM + LT_x_RR_SM - LT_y_r + F_aero_r
        Fz_RL = Fz_RL_s + LT_x_RL_NSM + LT_x_RL_SM + LT_y_r + F_aero_r

        zero = torch.tensor(0.0, dtype=Vx.dtype, device=Vx.device)
        Fz_FL = torch.maximum(Fz_FL, zero)
        Fz_FR = torch.maximum(Fz_FR, zero)
        Fz_RL = torch.maximum(Fz_RL, zero)
        Fz_RR = torch.maximum(Fz_RR, zero)
        return Fz_FL, Fz_FR, Fz_RL, Fz_RR


# ============================================================================
#  §3  SLIP ANGLES  (from Vx, Vy, r, delta) - multibody-consistent kinematic
#      projection without unsprung-roll terms (steady-state skidpad).
# ============================================================================

def slip_angles(Vx: torch.Tensor, Vy: torch.Tensor, r: torch.Tensor,
                delta: torch.Tensor, p: P26Params = P26
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ISO sign convention: positive slip angle (wheel points left relative to
    travel direction) → positive lateral force (Fy > 0, i.e. leftward).

    Standard kinematic formula per wheel contact point:
        alpha_f = delta - atan2(Vy_wheel, Vx_wheel)
        alpha_r =       - atan2(Vy_wheel, Vx_wheel)   (no steer at rear)

    where Vy_wheel = Vy ± (lf or lr)*r  (rigid body)
          Vx_wheel = Vx ∓ 0.5*T*r       (±0.5*track from centreline, per ISO)

    SIGN CHECK (CCW skidpad, ISO body frame):
      r < 0, Vy > 0 (outward lateral drift), delta < 0 (left steer)
      → alpha_f = delta - atan2(Vy+lf*r, Vx...) = (neg) - (neg) → negative
      → Fy_f < 0  (centripetal, inward = -y in this sensor frame) ✓
    """
    eps = 1e-3
    lf = p.lf
    lr = p.lr
    Tf = p.Tf
    Tr = p.Tr

    Vxs = torch.where(torch.abs(Vx) < eps, torch.full_like(Vx, eps), Vx)
    # NOTE: sign is delta - atan2(...), not atan2(...) - delta
    aFL = delta - torch.atan2(Vy + lf * r, Vxs + 0.5 * Tf * r)
    aFR = delta - torch.atan2(Vy + lf * r, Vxs - 0.5 * Tf * r)
    aRL =       - torch.atan2(Vy - lr * r, Vxs + 0.5 * Tr * r)
    aRR =       - torch.atan2(Vy - lr * r, Vxs - 0.5 * Tr * r)
    return aFL, aFR, aRL, aRR


# ============================================================================
#  §4  MAGIC FORMULA 6.2 - PURE LATERAL - learnable
#
#      Fy0(alpha,Fz,gamma) = Dy * sin( Cy * atan( By*aY - Ey*(By*aY - atan(By*aY)) ) )
#                           + SVy
#      aY   = alpha + SHy
#      dfz  = (Fz - Fz0) / Fz0
#      muy  = (pDy1 + pDy2*dfz) * (1 - pDy3*gamma^2)
#      Dy   = muy * Fz
#      Cy   = pCy1
#      Kya  = pKy1*Fz0 * sin( 2*atan( Fz / ((pKy2)*Fz0) ) )
#      By   = Kya / (Cy*Dy)
#      Ey   = (pEy1 + pEy2*dfz) * (1 - (pEy3 + pEy4*gamma) * sign(aY))
#      SHy  = (pHy1 + pHy2*dfz) + pHy3*gamma
#      SVy  = Fz * ((pVy1 + pVy2*dfz) + (pVy3 + pVy4*dfz)*gamma)
# ============================================================================

class MF62Lateral(nn.Module):
    """Pure-lateral MF6.2 with learnable coefficients."""

    PARAM_NAMES: List[str] = [
        # shape & peak
        "pCy1", "pDy1", "pDy2", "pDy3",
        # curvature
        "pEy1", "pEy2", "pEy3", "pEy4",
        # cornering stiffness
        "pKy1", "pKy2",
        # horizontal shift
        "pHy1", "pHy2", "pHy3",
        # vertical shift
        "pVy1", "pVy2", "pVy3", "pVy4",
    ]

    DEFAULT_INIT: Dict[str, float] = {
        "pCy1":  1.50,   # init mid-range [1.40, 1.60]; prev 1.30 let optimizer drift to 1.2 floor
        "pDy1":  2.30,   # high-mu FSAE tyre
        "pDy2": -0.12,   # modest load sensitivity
        "pDy3":  0.0,
        "pEy1": -0.50,
        "pEy2":  0.0,
        "pEy3":  0.0,
        "pEy4":  0.0,
        "pKy1":  70.0,   # FSAE slick target: peak alpha ~5-6 deg.
                         # With Cy=1.6: peak at aY=tan(pi/3.2)/B=1.503/B.
                         # B = Kya/(C*D). For peak=5.5 deg (0.096 rad):
                         # B=15.7 -> Kya=15.7*1.6*1685=42.3 kN/rad -> pKy1=66.
                         # Init at 70 (slightly aggressive; optimizer will settle).
                         # Prev value 45 gave peak ~8 deg — too high for FSAE.
        "pKy2":  1.60,
        "pHy1":  0.0,
        "pHy2":  0.0,
        "pHy3":  0.0,
        "pVy1":  0.0,
        "pVy2":  0.0,
        "pVy3":  0.0,
        "pVy4":  0.0,
    }

    def __init__(self, Fz0: float = 890.0, init: Optional[Dict[str, float]] = None):
        super().__init__()
        init = init or self.DEFAULT_INIT
        for name in self.PARAM_NAMES:
            self.register_parameter(
                name,
                nn.Parameter(torch.tensor(init[name], dtype=torch.float64))
            )
        # nominal reference load (not trained)
        self.register_buffer("Fz0", torch.tensor(Fz0, dtype=torch.float64))

    # ---------------- physics helpers ----------------
    def _smooth_sign(self, x: torch.Tensor, k: float = 30.0) -> torch.Tensor:
        """
        Smooth, differentiable approximation of sign(x) ∈ (−1, +1).
        tanh(k*x) saturates to ±1 for |x| > a few multiples of 1/k, giving
        well-behaved gradients and keeping Ey bounded after multiplication.
        k=30 gives a transition zone of roughly ±0.1 rad (~5.7 deg) which is
        wide enough that the Ey formula has well-conditioned gradients throughout
        the practical slip angle range.  k=100 (previous value) made the
        transition so sharp that for any non-negligible aY the function was
        saturated and gradients of pEy3 were near-zero — equivalent to using
        a hard sign and making pEy3 untrainable.
        """
        return torch.tanh(k * x)

    def forward(self, alpha: torch.Tensor, Fz: torch.Tensor,
                gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args
        ----
        alpha : [N] slip angle  [rad]
        Fz    : [N] vertical load [N] (must be positive)
        gamma : [N] inclination angle [rad] (defaults to zero)

        Returns
        -------
        Fy    : [N] lateral force [N]
        """
        if gamma is None:
            gamma = torch.zeros_like(alpha)

        Fz = torch.clamp(Fz, min=30.0)              # numerical safety
        dfz = (Fz - self.Fz0) / self.Fz0

         
        muy = (self.pDy1 + self.pDy2 * dfz) * (1.0 - self.pDy3 * gamma * gamma)
        Dy  = muy * Fz
        # Hard clamp on Cy inside the forward pass.
        # The soft box-constraint penalty in the regulariser (weight 200 × w_reg)
        # is too weak to reliably prevent pCy1 from drifting above 1.6, especially
        # at reduced TTC scale (0.7×) where the fitting signal is weaker.
        # At pCy1 ≈ 2.0 the formula evaluates sin(2 × π/2) = sin(π) ≈ 0, making
        # Fy drop to ~2% of peak at large slip angles — physically wrong.
        # Clamping here guarantees the formula always uses a physical Cy ∈ [1.40, 1.60].
        # CRITICAL: the previous floor of 1.2 was a zero-penalty attractor.  The
        # soft-box regulariser lower bound was also 1.2, so relu(1.2 - 1.2)=0 —
        # the optimizer could park at 1.2 without paying ANY soft penalty while the
        # hard clamp stopped further descent.  At C=1.2: peak_alpha = tan(π/2.4)/B =
        # 3.73/B, which is 2.49× the value at C=1.60 (1.50/B), driving peaks to
        # 13-18° instead of the target 5-7°.  Raising the hard floor to 1.40 breaks
        # this attractor.  Physical range for lateral MF is 1.3–1.6; 1.40 is
        # conservative (leaves 0.20 of soft headroom below 1.60).
        Cy  = torch.clamp(self.pCy1, min=1.40, max=1.6)
        # cornering stiffness (Kya > 0 for positive alpha -> positive Fy)
        Kya = self.pKy1 * self.Fz0 * torch.sin(
            2.0 * torch.atan(Fz / (self.pKy2 * self.Fz0 + 1e-9))
        )
        By  = Kya / (Cy * Dy + 1e-9)

        SHy = (self.pHy1 + self.pHy2 * dfz) + self.pHy3 * gamma
        aY  = alpha + SHy
        sgn_aY = self._smooth_sign(aY)

        Ey = (self.pEy1 + self.pEy2 * dfz) * (
            1.0 - (self.pEy3 + self.pEy4 * gamma) * sgn_aY
        )
        # MF6.2 condition: -1 <= Ey <= 1  (two-sided clamp).
        # The lower bound is equally important: without it, when pEy3 drives
        # Ey to zero on the negative-alpha side (sign-switch exploit), there is
        # no clamp gradient on that side and pEy1/pEy2 can grow without bound.
        # Clamping on both sides ensures gradients always flow through Ey.
        Ey = torch.clamp(Ey, min=-1.0, max=0.99)

        BaY = By * aY
        phi = BaY - Ey * (BaY - torch.atan(BaY))
        SVy = Fz * ((self.pVy1 + self.pVy2 * dfz) + (self.pVy3 + self.pVy4 * dfz) * gamma)
        Fy  = Dy * torch.sin(Cy * torch.atan(phi)) + SVy
        return Fy

    def coefficients(self) -> Dict[str, float]:
        out = {name: float(getattr(self, name).detach().cpu().item())
               for name in self.PARAM_NAMES}
        out["Fz0_nominal"] = float(self.Fz0.detach().cpu().item())
        return out


# ============================================================================
#  §5  DATA LOADERS
# ============================================================================

# --- telemetry columns ------------------------------------------------------
# Primary channels used by the PINN.  The processed CSVs already carry a
# pre-filtered (Savitzky-Golay) version in these names AND the original sensor
# readings in the corresponding *_raw columns.  We therefore read the processed
# values directly and DO NOT apply another Savgol pass – double-filtering
# introduces phase error and over-smoothes the slip angles.
TEL_CHANNELS = ["steering", "a_x", "a_y", "u_x", "u_y", "yaw_rate", "time"]
_RAW_SUFFIX = "_raw"          # presence of e.g. "steering_raw" signals processed CSV
_RAW_CHANNELS = [c + _RAW_SUFFIX for c in ["steering", "a_x", "a_y",
                                             "u_x", "u_y", "yaw_rate"]]

# Speed window for telemetry rows used in training.
# Speed window for telemetry rows.  The primary purpose is to exclude very-slow
# entry / exit transients, not to enforce a single target speed.  Three skidpad
# radii are now used:
#   10.5 m → ~11.2 m/s   9 m → ~11.3 m/s   6 m → ~8.5–9.5 m/s
# A bandwidth of ±4.5 m/s around the TTC reference (25 mph = 11.18 m/s) gives
# a window of [6.7, 15.7] m/s, which captures steady-state laps for all three
# radii while still discarding rows where the car is essentially stopped.
_VX_TARGET_MPS: float = 25.0 * 1609.344 / 3600.0   # 11.176 m/s (TTC reference)
_VX_BANDWIDTH_MPS: float = 4.5                       # ± 4.5 m/s  → 6.7 – 15.7 m/s


def _smooth(x: np.ndarray, wlen: int = 15, poly: int = 3) -> np.ndarray:
    if len(x) < wlen + 2:
        return x
    if wlen % 2 == 0:
        wlen += 1
    return savgol_filter(x, wlen, poly)


def load_telemetry(paths: List[str],
                   smooth: bool = True,
                   vx_target: float = _VX_TARGET_MPS,
                   vx_bandwidth: float = _VX_BANDWIDTH_MPS) -> pd.DataFrame:
    """
    Load skidpad telemetry CSVs, preferring pre-processed (filtered) channels.

    Strategy
    --------
    * If the CSV contains ``steering_raw`` (and siblings) the named channels
      (``steering``, ``a_x``, …) are already Savitzky-Golay filtered.  We use
      them as-is to avoid a second filter pass.
    * If only raw channels exist we apply one Savgol pass as before.
    * ``r_dot`` (yaw angular acceleration) is always derived here by finite-
      differencing the (smooth) ``yaw_rate`` signal and then lightly re-filtered
      with a short Savgol window.
    * Rows whose longitudinal speed ``u_x`` falls outside
      ``[vx_target ± vx_bandwidth]`` are dropped to exclude very-slow entry /
      exit transients.  The default window (6.7–15.7 m/s) accommodates all three
      circle radii: 10.5 m (~11.2 m/s), 9 m (~11.3 m/s), and 6 m (~8.5–9.5 m/s).
    """
    frames = []
    vx_lo = vx_target - vx_bandwidth
    vx_hi = vx_target + vx_bandwidth

    for path in paths:
        if not os.path.isfile(path):
            print(f"  [TEL] missing {path}")
            continue
        df_raw = pd.read_csv(path)
        missing = [c for c in TEL_CHANNELS if c not in df_raw.columns]
        if missing:
            print(f"  [TEL] {os.path.basename(path)} missing cols {missing}")
            continue

        df = df_raw[TEL_CHANNELS].copy()
        df["run"] = os.path.basename(path)
        df = df.dropna().reset_index(drop=True)

        # ---- smoothing --------------------------------------------------
        # Only apply if the file does NOT already contain pre-filtered channels
        is_processed = all(c in df_raw.columns for c in _RAW_CHANNELS)
        if is_processed:
            # Channels are already filtered – skip additional smoothing.
            pass
        elif smooth:
            for c in ["a_x", "a_y", "u_x", "u_y", "yaw_rate", "steering"]:
                df[c] = _smooth(df[c].values)

        # ---- yaw angular acceleration (always computed fresh) -----------
        t = df["time"].values
        dt = np.gradient(t)
        dt = np.where(dt < 1e-6, 0.01, dt)
        r_dot_raw = np.gradient(df["yaw_rate"].values) / dt
        # Light Savgol on r_dot regardless of source (differencing adds noise)
        df["r_dot"] = _smooth(r_dot_raw, wlen=21, poly=3)

        # ---- steer conversion -------------------------------------------
        # The 'steering' column is STEERING WHEEL degrees (confirmed by
        # K_us regression test across all 8 runs — see P26Params.steer_ratio).
        # Divide by steering ratio to get road-wheel angle, then → radians.
        ratio = P26.steer_ratio if not P26.steer_is_roadwheel else 1.0
        df["delta_rad"] = df["steering"].values / ratio * np.pi / 180.0
        steer_deg_rw = df["steering"].values / ratio   # for reporting only
        print(f"  [TEL] {os.path.basename(path)}: steering ÷{ratio:.2f} → "
              f"road-wheel [{steer_deg_rw.min():.1f}, {steer_deg_rw.max():.1f}]°")

        # ---- speed filter: keep rows near TTC test speed ----------------
        n_before = len(df)
        df = df[(df["u_x"] >= vx_lo) & (df["u_x"] <= vx_hi)].reset_index(drop=True)
        n_kept = len(df)

        src = "processed" if is_processed else "raw"
        print(f"  [TEL] {os.path.basename(path)} ({src}): "
              f"{n_kept}/{n_before} rows kept  "
              f"({vx_lo:.1f}–{vx_hi:.1f} m/s speed window)  "
              f"u_x=[{df['u_x'].min():.2f},{df['u_x'].max():.2f}] m/s")

        if n_kept < 50:
            print(f"  [TEL] WARNING: very few rows after speed filter, skipping.")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No telemetry data loaded")
    out = pd.concat(frames, ignore_index=True)

    # Build a monotonic global time axis across all concatenated runs.
    # Each run's local 'time' column resets to ~0 at its own start.
    # Plotting against the raw 'time' column produces horizontal line
    # artefacts where matplotlib connects the last sample of run N (large t)
    # to the first sample of run N+1 (t ≈ 0) with a near-horizontal stroke.
    # time_global gives a continuous, ever-increasing axis that eliminates
    # those artefacts without altering any physics quantities.
    dt_nom = float(np.median(np.diff(frames[0]["time"].values[:500])))
    dt_nom = dt_nom if (np.isfinite(dt_nom) and dt_nom > 1e-4) else 0.01
    out["time_global"] = np.arange(len(out)) * dt_nom

    print(f"  [TEL] total: {len(out)} rows across {len(frames)} run(s)  "
          f"(speed filter: {vx_lo:.2f}–{vx_hi:.2f} m/s = "
          f"{vx_lo*3.6:.1f}–{vx_hi*3.6:.1f} km/h)  dt_nom={dt_nom*1000:.1f} ms")
    return out


def load_ttc(paths: List[str]) -> pd.DataFrame:
    """
    Loads raw Calspan TTC CSVs.  Returns a DataFrame with columns
        alpha  [rad]  (ISO: positive alpha -> positive Fy)
        Fz     [N]    (positive)
        Fy     [N]    (ISO)
        gamma  [rad]  (0 for this test)
    """
    frames = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"  [TTC] missing {path}")
            continue
        df = pd.read_csv(path)
        for c in ("SA", "FY", "FZ", "IA"):
            if c not in df.columns:
                print(f"  [TTC] {os.path.basename(path)} missing {c}")
                df = None
                break
        if df is None:
            continue

        # SAE -> ISO (positive alpha <-> positive Fy) : flip Fy
        alpha_rad = df["SA"].values * np.pi / 180.0
        Fy_iso    = -df["FY"].values
        Fz        = np.abs(df["FZ"].values)
        gamma_rad = df["IA"].values * np.pi / 180.0

        # filter obvious transients / near-zero load rows.
        # Threshold is 100 N to accommodate the 50 lbs (≈222 N) load level
        # while still rejecting unloaded-tyre noise at the start of a run.
        valid = (Fz > 100.0) & np.isfinite(alpha_rad) & np.isfinite(Fy_iso)
        out = pd.DataFrame({
            "alpha": alpha_rad[valid],
            "Fz":    Fz[valid],
            "Fy":    Fy_iso[valid],
            "gamma": gamma_rad[valid],
            "source": os.path.basename(path),
        })
        # Assign integer source ID (needed for per-source TTC loss normalisation)
        out["source_id"] = len(frames)  # 0-indexed per file
        frames.append(out)
        print(f"  [TTC] {os.path.basename(path)}: {len(out)} rows  "
              f"Fz_mean={out['Fz'].mean():.0f}N  alpha=[{np.degrees(out['alpha']).min():.1f},"
              f"{np.degrees(out['alpha']).max():.1f}] deg")
    if not frames:
        raise RuntimeError("No TTC data loaded")
    df = pd.concat(frames, ignore_index=True)
    print(f"  [TTC] total: {len(df)} rows")
    return df


# ============================================================================
#  §6  PINN LOSS  &  TRAINER
# ============================================================================

class VehiclePINN(nn.Module):
    """
    Full vehicle-level PINN wrapper.
    Holds the shared MF6.2 tyre model plus a per-axle mu offset so front/rear
    axles can differ (camber / pressure / temperature-induced variations).
    """

    def __init__(self, p: P26Params = P26, Fz0: float = 890.0):
        super().__init__()
        self.p = p
        self.tire = MF62Lateral(Fz0=Fz0)
        self.loader = LoadTransfer(p)
        # Per-axle mu scaling – constrained positive via softplus + offset.
        # softplus(raw) + 0.5  guarantees mu_scale >= 0.5 at all times,
        # preventing the optimizer from flipping tire-force signs.
        # Initialised so that softplus(_raw) + 0.5 == 1.0  =>  _raw = softplus_inv(0.5)
        _sp_inv = math.log(math.exp(0.5) - 1.0)   # ≈ -0.3133
        self._mu_f_raw = nn.Parameter(torch.tensor(_sp_inv, dtype=torch.float64))
        self._mu_r_raw = nn.Parameter(torch.tensor(_sp_inv, dtype=torch.float64))
        # Learnable Vy sensor bias [m/s]. Absorbs systematic lateral-velocity
        # sensor drift that imprints a uniform slip-angle error on all 4 wheels.
        # Initialised at 0.0 (neutral); a strong L2 anchor (weight 30 inside the
        # regulariser, effective = 30*w_reg = 0.03) prevents runaway while still
        # allowing ±1-2 m/s corrections.  Hard asymmetric bounds [-2.5, +1.0]
        # are physically motivated: a sensor reading too HIGH gives a positive Vy
        # bias → correction is negative.  Previous init of -1.0 hard-coded an
        # assumed sign; starting at 0 lets the balance equations discover the
        # correct sign.  PREVIOUS FAILURE MODE: vy_offset flew to the old hard
        # bound of +3.0 m/s (wrong sign, flipped all Vy readings) because the
        # old L2 weight (0.5) was 60x too weak — 5e-4 effective cost for a 3 m/s
        # excursion vs 0.04 balance loss that tolerated the degeneracy.
        self.vy_offset = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        # Iz is FIXED (not learnable): learnable Iz drifts to clamp floor (60)
        # creating 40*r_dot Nm systematic yaw error. Use p.Iz directly.

    @property
    def mu_scale_f(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._mu_f_raw) + 0.5

    @property
    def mu_scale_r(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._mu_r_raw) + 0.5

    # --- per-wheel predictions ------------------------------------------
    def predict_wheels(self, Vx, Vy, r, delta, ax, ay):
        Fz_FL, Fz_FR, Fz_RL, Fz_RR = self.loader(Vx, ax, ay)
        # Correct for learnable Vy sensor bias before computing slip angles.
        Vy_corr = Vy + self.vy_offset
        aFL, aFR, aRL, aRR = slip_angles(Vx, Vy_corr, r, delta, self.p)

        Fy_FL = self.mu_scale_f * self.tire(aFL, Fz_FL)
        Fy_FR = self.mu_scale_f * self.tire(aFR, Fz_FR)
        Fy_RL = self.mu_scale_r * self.tire(aRL, Fz_RL)
        Fy_RR = self.mu_scale_r * self.tire(aRR, Fz_RR)

        return {
            "Fz": (Fz_FL, Fz_FR, Fz_RL, Fz_RR),
            "alpha": (aFL, aFR, aRL, aRR),
            "Fy": (Fy_FL, Fy_FR, Fy_RL, Fy_RR),
        }

    # --- vehicle-level balance residuals --------------------------------
    def vehicle_residuals(self, Vx, Vy, r, r_dot, delta, ax, ay):
        pred = self.predict_wheels(Vx, Vy, r, delta, ax, ay)
        Fy_FL, Fy_FR, Fy_RL, Fy_RR = pred["Fy"]
        cd = torch.cos(delta)
        sd = torch.sin(delta)
        # aero drag (longitudinal only - ignored in lateral balance)
        m = self.p.m_tot
        Iz = self.p.Iz  # fixed yaw inertia [kg*m^2] (learnable Iz was unstable)
        lf = self.p.lf
        lr = self.p.lr
        Tf = self.p.Tf

        Fy_body = (Fy_FL + Fy_FR) * cd + (Fy_RL + Fy_RR)
        # yaw moment about CoG (front wheel Fy rotated into chassis frame)
        Mz = (
            lf * ((Fy_FL + Fy_FR) * cd)
            - lr * (Fy_RL + Fy_RR)
            + 0.5 * Tf * ((Fy_FL - Fy_FR) * sd)
        )

        R_lat = m * ay - Fy_body              # = 0  ideally
        R_yaw = Iz * r_dot - Mz               # = 0  ideally
        return R_lat, R_yaw, pred


# ----------------------------------------------------------------------------

class LossSpec:
    def __init__(self, w_ttc=3.0, w_lat=8.0, w_yaw=6.0, w_reg=1e-3,
                 Mchar=1000.0, r_dot_thresh=1.0):
        #   Weight rationale (normalisation asymmetry):
        #   L_ttc is normalised by (Fz_scale)^2; L_lat by (m*g)^2 = (2534 N)^2;
        #   L_yaw by Mchar^2.
        #   Setting w_ttc=3, w_lat=0.8 gives a true effective ratio of
        #   ~3 / (0.8/6.4) ≈ 24 : 1 in favour of TTC shape fitting, which
        #   is the dominant constraint for correct MF parameter recovery.
        #   w_yaw raised 0.3 → 0.8 to match w_lat.  The yaw constraint was
        #   previously under-weighted relative to its physical importance: a
        #   100 N front-rear Fy error creates an 80 N·m Mz error → 0.8 rad/s²
        #   yaw acceleration error, but L_yaw (normalised by Mchar²) is a
        #   smaller dimensionless number than L_lat, so it needed a higher
        #   weight to receive equal effective gradient influence.
        #   The vehicle equations provide regularisation via physics consistency
        #   but must not overwhelm the direct tyre data signal.
        self.w_ttc = w_ttc
        self.w_lat = w_lat
        self.w_yaw = w_yaw
        self.w_reg = w_reg
        self.Mchar        = Mchar          # yaw-moment normalisation scale [Nm]
        self.r_dot_thresh = r_dot_thresh   # Gaussian kernel width [rad/s^2]


def pinn_loss(model: VehiclePINN, tel_batch, ttc_batch, spec: LossSpec):
    """
    tel_batch: dict of 1-D tensors
        Vx, Vy, r, r_dot, delta, ax, ay
    ttc_batch: dict
        alpha, Fz, Fy
    """
    # 1) TTC tyre fit (supervised, per-source normalised) ------------------
    # Normalise each source's squared residual by its mean Fz² so that each
    # load level contributes equal weight in *friction-coefficient* space.
    # Without this, low-load sources (50 lbs, Fz≈220 N) have tiny absolute
    # errors that are dwarfed by high-load sources (250 lbs, Fz≈1126 N),
    # which biases pDy1 toward the high-mu low-load tyre behaviour.
    Fy_pred_ttc = model.tire(ttc_batch["alpha"], ttc_batch["Fz"])
    src_ids = ttc_batch["source_id"].long()
    n_sources = int(src_ids.max().item()) + 1
    L_ttc_norm = torch.zeros(1, dtype=torch.float64, device=src_ids.device).squeeze()
    n_valid = 0
    for sid in range(n_sources):
        mask = src_ids == sid
        if mask.sum() < 5:
            continue
        Fz_scale = ttc_batch["Fz"][mask].mean().detach()   # characteristic Fz for source
        err_sq = (Fy_pred_ttc[mask] - ttc_batch["Fy"][mask]) ** 2
        L_ttc_norm = L_ttc_norm + torch.mean(err_sq) / (Fz_scale ** 2)
        n_valid += 1
    if n_valid > 0:
        L_ttc_norm = L_ttc_norm / n_valid   # average over sources
    # Direct loss in absolute force space [normalised by (1000 N)^2].
    # Gives high-Fz TTC data proportional gradient on pKy2 / peak shape.
    # Without this, the 1000 N source gets 6x less stiffness gradient
    # than the 400 N source, causing wrong peak slip angles at high Fz.
    L_ttc_direct = torch.mean(
        (Fy_pred_ttc - ttc_batch["Fy"]) ** 2
    ) / (1000.0 ** 2)
    # Blend: 70% mu-space (correct for pDy), 30% force-space (correct for pKy).
    L_ttc = 0.70 * L_ttc_norm + 0.30 * L_ttc_direct

    # 2) vehicle-level residuals ----------------------------------------
    R_lat, R_yaw, _ = model.vehicle_residuals(
        tel_batch["Vx"], tel_batch["Vy"], tel_batch["r"],
        tel_batch["r_dot"], tel_batch["delta"],
        tel_batch["ax"], tel_batch["ay"],
    )
    m = model.p.m_tot
    g = model.p.g
    Fchar = m * g                                   # ~2 534 N  (total weight)
    Mchar = spec.Mchar                              # physically grounded [Nm]

    # Soft Gaussian steady-state weighting (replaces hard binary mask).
    # w_ss = exp(-(r_dot/thresh)^2): at r_dot=0 full weight, at r_dot=thresh
    # weight=0.37, at 2*thresh weight=0.02. Gradient flows continuously;
    # smoother loss landscape for LBFGS, no cliff-edge discontinuities.
    r_dot_t  = tel_batch["r_dot"]
    w_ss     = torch.exp(-(r_dot_t / spec.r_dot_thresh) ** 2)
    w_ss_sum = w_ss.sum().clamp(min=1.0)
    # Huber loss: MSE when |R| < delta, linear beyond (clips spike gradient).
    # delta=0.15 lateral: quadratic zone covers |R_lat| < 0.15*2534 = 380 N
    #   (~15% of m*g); the previous delta=0.5 (1267 N) was so loose that
    #   residuals of several hundred Newtons incurred almost no extra gradient.
    # delta=0.25 yaw: quadratic to |R_yaw| < 0.25*1000 = 250 Nm; peak Mz on
    #   the 10.5 m skidpad is ~280-400 Nm, so this keeps most near-SS samples
    #   in the quadratic zone while clipping transient spikes.
    huber_lat = torch.nn.functional.huber_loss(
        R_lat / Fchar, torch.zeros_like(R_lat), reduction="none", delta=0.15
    )
    huber_yaw = torch.nn.functional.huber_loss(
        R_yaw / Mchar, torch.zeros_like(R_yaw), reduction="none", delta=0.25
    )
    L_lat = torch.sum(w_ss * huber_lat) / w_ss_sum
    L_yaw = torch.sum(w_ss * huber_yaw) / w_ss_sum

    # 3) parameter regulariser ------------------------------------------
    #    Physics-grounded priors that block the main failure modes.
    #    All weights here are INSIDE L_reg; they are multiplied by spec.w_reg
    #    (= 1e-3 by default), so effective weight in total loss = weight * 1e-3.
    #
    # ── (a) pEy3 sign-switch exploit  [weight 150] ────────────────────────
    #    pEy3 → -1 collapses curvature factor for negative aY (all CCW data).
    reg = 0.0
    reg = reg + model.tire.pEy3 ** 2 * 150.0
    reg = reg + model.tire.pEy4 ** 2
    # ── (b) Ey base magnitude — prevent runaway past the ±0.99 clamp ──────
    #    pEy2 WEIGHT RAISED from 0.01 to 3.0 (300× increase) + box [-0.5, +0.5].
    #    PREVIOUS FAILURE: pEy2=+2.745 with effective cost 2.745²×0.01×1e-3=7.5e-5
    #    — essentially free.  At pEy2=2.745: Ey_raw at Fz=1200N is +1.95, clamped
    #    to 0.99, making the high-load MF curve completely flat with NO peak within
    #    ±25° — physically impossible for any real tire.  This also drives enormous
    #    yaw moments from the outer front wheel (Ey≈1 → Fy grows linearly past peak).
    #    Physical range: pEy2 is just a load-sensitivity correction of the curvature
    #    factor, never exceeding ±0.5 for real tires.  The box enforces this hard.
    reg = reg + model.tire.pEy1 ** 2 * 0.1
    reg = reg + model.tire.pEy2 ** 2 * 3.0
    reg = reg + torch.relu( model.tire.pEy2 - 0.5) ** 2 * 200.0
    reg = reg + torch.relu(-0.5 - model.tire.pEy2) ** 2 * 200.0
    # ── (c) SVy load sensitivity ──────────────────────────────────────────
    #    pVy2 WEIGHT RAISED from 10 to 50.  At pVy2=0.141 the old effective
    #    penalty was 0.141²×10×1e-3=2e-4 — negligible.  At weight=50: 1e-3 cost.
    reg = reg + model.tire.pVy2 ** 2 * 50.0
    # ── (d) degressive friction enforcement  pDy2 ≤ 0 ────────────────────
    reg = reg + torch.relu(model.tire.pDy2) ** 2 * 50.0
    # ── (e) pCy1 soft box [1.40, 1.60] ───────────────────────────────────
    #    Lower bound RAISED from 1.2 to 1.40.  At 1.2 the soft penalty was zero
    #    (relu(1.2-1.2)=0) while the hard forward() clamp sat at the same value,
    #    creating a zero-cost attractor at C=1.2.  With the hard clamp also raised
    #    to 1.40, the optimizer now has a continuous gradient pulling pCy1 into
    #    [1.40, 1.60].  Weight raised 200→300 for stronger enforcement.
    reg = reg + torch.relu(model.tire.pCy1 - 1.60) ** 2 * 300.0
    reg = reg + torch.relu(1.40 - model.tire.pCy1) ** 2 * 300.0
    # ── (f) mu_scale tightened box [0.7, 1.3] + central pull toward 1.0 ──
    #    PREVIOUS FAILURE: mu_scale_f/r both converged to the 0.5 floor
    #    (the softplus guarantee), halving all tire forces.  Root cause:
    #    w_lat=1.0 was too weak; the old [0.5, 1.5] box had zero penalty at 0.5.
    #    Fix: (1) tighten box to [0.7, 1.3] — soft penalty outside this range;
    #         (2) add L2 pull toward 1.0 with weight 10 (effective 0.01).
    #    mu_scale physically encodes the ratio of on-track grip to TTC grip.
    #    A well-prepared FSAE tire on a smooth surface should be 0.85–1.15.
    #    Allowing 0.7–1.3 gives 30% headroom on each side for surface/temp
    #    variation while strongly preventing the degenerate 0.5 solution.
    reg = reg + (model.mu_scale_f - 1.0) ** 2 * 10.0
    reg = reg + (model.mu_scale_r - 1.0) ** 2 * 10.0
    reg = reg + torch.relu(model.mu_scale_f - 1.3) ** 2 * 500.0
    reg = reg + torch.relu(0.7 - model.mu_scale_f) ** 2 * 500.0
    reg = reg + torch.relu(model.mu_scale_r - 1.3) ** 2 * 500.0
    reg = reg + torch.relu(0.7 - model.mu_scale_r) ** 2 * 500.0
    # ── (g) Horizontal shift pHy1/pHy2 ───────────────────────────────────
    reg = reg + model.tire.pHy1 ** 2 * 30.0
    reg = reg + model.tire.pHy2 ** 2 * 30.0
    # ── (h) Vertical shift pVy1 ──────────────────────────────────────────
    reg = reg + model.tire.pVy1 ** 2 * 50.0
    # ── (i) vy_offset: strong L2 anchor + tight asymmetric hard bounds ───
    #    PREVIOUS FAILURE: vy_offset flew to +3.0 m/s (wrong sign — added 3 m/s
    #    to Vy instead of subtracting the sensor bias) because old L2 weight=0.5
    #    gave only 5e-4 effective penalty, which was negligible vs the 0.04
    #    balance loss.  New weight=30 gives effective 0.03 — meaningful but not
    #    dominating.  Asymmetric hard bounds [-2.5, +1.0]: the physical sensor
    #    bias is a positive offset on Vy_measured (reads too high), so correction
    #    should be negative; allowing +1.0 m/s headroom for any positive
    #    correction while blocking the catastrophic +3.0 runaway.
    reg = reg + model.vy_offset ** 2 * 30.0
    # Hard bounds relaxed to [-2.5, +2.5] — symmetric, no assumed bias sign.
    # Previous asymmetric bound of +1.0 blocked vy_offset at +1.08 (just over the
    # wall), costing 0.034 in reg while the optimizer clearly needed that value.
    # The L2 anchor (weight=30, effective 0.03) is now the primary restraint:
    # a vy_offset of +1 m/s costs 1²×30×1e-3=0.03, which is meaningful but not
    # prohibitive.  The optimizer can correct a genuine sensor bias of up to ±2.5 m/s.
    reg = reg + torch.relu( model.vy_offset - 2.5) ** 2 * 5000.0   # hard upper: +2.5 m/s
    reg = reg + torch.relu(-model.vy_offset - 2.5) ** 2 * 5000.0   # hard lower: -2.5 m/s
    # ── (j) pDy1 (mu_y) soft box [1.5, 2.5] ─────────────────────────────
    #    Physical constraint: friction coefficient (muy = pDy1 at Fz0, gamma=0)
    #    for FSAE slicks on smooth tarmac: 1.5–2.5.  TTC data (Hoosier 43075)
    #    shows pDy1 ≈ 2.0–2.4 at the test surface; on-track mu varies.
    #    Weight 300 (effective 0.3): strong enough to resist the tendency of
    #    the optimizer to inflate pDy1 beyond 2.5 to explain high balance forces.
    reg = reg + torch.relu(model.tire.pDy1 - 2.5) ** 2 * 300.0
    reg = reg + torch.relu(1.5 - model.tire.pDy1) ** 2 * 300.0
    # ── (k1) pKy2 (load at peak stiffness / Fz0) soft box [0.8, 3.0] ─────
    #    pKy2 sets the normalised load at which Kya is maximum (Fz = pKy2*Fz0).
    #    Converged to 2.673 — peak stiffness at Fz=1777 N, above FSAE range —
    #    physically plausible but needs soft bounding to prevent runaway.
    #    Physical range for racing slicks: 1.0–3.0.  Box below 0.8 blocks cases
    #    where peak stiffness falls below the operating range minimum (~530 N).
    reg = reg + torch.relu(model.tire.pKy2 - 3.0) ** 2 * 20.0
    reg = reg + torch.relu(0.8 - model.tire.pKy2) ** 2 * 20.0
    # ── (k2) pKy1 (cornering stiffness scale) soft box [55, 90] ──────────
    #    pKy1 controls peak alpha via Kya = pKy1*Fz0*sin(2*atan(Fz/(pKy2*Fz0))).
    #    At Fz0≈665 N with pKy2≈1.35: Kya ≈ pKy1 * 635 N/unit.
    #    For peak alpha at 5-7 deg: need B = 1.503/(tan(5.5°)) ≈ 15.6
    #      → Kya = B*C*D = 15.6*1.6*1685 = 42.1 kN/rad → pKy1 ≈ 66.
    #    Box [55, 90] covers peak alpha range [4.5°, 7.5°] — physically correct
    #    for FSAE Hoosier slicks.  Previous init=45 gave pKy1≈44, peak~8° — high.
    #    Weight 5 (effective 5e-3): light nudge; let the TTC data drive the fit.
    reg = reg + torch.relu(model.tire.pKy1 - 90.0) ** 2 * 5.0
    reg = reg + torch.relu(55.0 - model.tire.pKy1) ** 2 * 5.0
    L_reg = reg

    total = spec.w_ttc * L_ttc + spec.w_lat * L_lat + spec.w_yaw * L_yaw + spec.w_reg * L_reg
    parts = dict(total=total.item(), ttc=L_ttc.item(),
                 lat=L_lat.item(), yaw=L_yaw.item(), reg=float(L_reg.detach()))
    return total, parts


# ----------------------------------------------------------------------------

class Trainer:
    def __init__(self, model: VehiclePINN, spec: LossSpec,
                 n_adam: int = 4000, n_lbfgs: int = 600,
                 batch_tel: int = 2048, batch_ttc: int = 1024,
                 lr: float = 3e-3):
        self.model = model
        self.spec = spec
        self.n_adam = n_adam
        self.n_lbfgs = n_lbfgs
        self.batch_tel = batch_tel
        self.batch_ttc = batch_ttc
        self.lr = lr
        self.history = {k: [] for k in ["epoch", "total", "ttc", "lat", "yaw", "reg"]}
        self._best_loss = math.inf
        self._best_state = None

    @staticmethod
    def _df_to_tensors(df: pd.DataFrame, keys: Dict[str, str]) -> Dict[str, torch.Tensor]:
        return {k: torch.as_tensor(df[v].values, dtype=torch.float64, device=DEVICE)
                for k, v in keys.items()}

    @staticmethod
    def _sample_idx(n: int, batch: int) -> torch.Tensor:
        if batch >= n:
            return torch.arange(n, device=DEVICE)
        return torch.randint(0, n, (batch,), device=DEVICE)

    def _tel_batch(self, tel_t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        N = tel_t["Vx"].shape[0]
        idx = self._sample_idx(N, self.batch_tel)
        return {k: v[idx] for k, v in tel_t.items()}

    def _ttc_batch(self, ttc_t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        N = ttc_t["alpha"].shape[0]
        idx = self._sample_idx(N, self.batch_ttc)
        return {k: v[idx] for k, v in ttc_t.items()}

    def _save_if_best(self, loss: float) -> None:
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _log(self, epoch: int, parts: Dict[str, float]) -> None:
        self.history["epoch"].append(epoch)
        for k in ["total", "ttc", "lat", "yaw", "reg"]:
            self.history[k].append(parts[k])
        self._save_if_best(parts["total"])

    # ------------- ADAM -------------
    def _adam(self, tel_t, ttc_t, verbose: bool):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_adam, eta_min=self.lr * 0.05)
        t0 = time.time()
        print(f"\n[Adam]  {self.n_adam} epochs   lr={self.lr:.1e}   batch(tel/ttc)={self.batch_tel}/{self.batch_ttc}")
        for epoch in range(self.n_adam):
            batch_tel = self._tel_batch(tel_t)
            batch_ttc = self._ttc_batch(ttc_t)
            opt.zero_grad(set_to_none=True)
            loss, parts = pinn_loss(self.model, batch_tel, batch_ttc, self.spec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            opt.step()
            sched.step()
            if epoch % 50 == 0:
                self._log(epoch, parts)
            if verbose and epoch % 200 == 0:
                print(f"  epoch {epoch:>5d}  total={parts['total']:.5f}  "
                      f"ttc={parts['ttc']:.5f}  lat={parts['lat']:.5f}  "
                      f"yaw={parts['yaw']:.5f}  t={time.time()-t0:.1f}s")

    # ------------- L-BFGS -------------
    def _lbfgs(self, tel_t, ttc_t, verbose: bool):
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        opt = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=self.n_lbfgs,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        calls = {"n": 0}
        print(f"\n[L-BFGS]  max_iter={self.n_lbfgs}   using FULL batch")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss, parts = pinn_loss(self.model, tel_t, ttc_t, self.spec)
            loss.backward()
            calls["n"] += 1
            if verbose and calls["n"] % 25 == 0:
                print(f"  call {calls['n']:>4d}  total={parts['total']:.6f}  "
                      f"ttc={parts['ttc']:.6f}  lat={parts['lat']:.6f}  yaw={parts['yaw']:.6f}")
            if calls["n"] % 5 == 0:
                self._log(self.n_adam + calls["n"], parts)
            return loss

        opt.step(closure)

    # ------------- Phase-0: TTC-only warmup -------------
    def _ttc_warmup(self, ttc_t: Dict[str, torch.Tensor],
                    n_warmup: int = 800, verbose: bool = True) -> None:
        """Fit MF6.2 directly to TTC data before vehicle constraints are active.

        Rationale: if the Adam phase starts cold with all constraints active,
        the vehicle balance equations (which see only one-sided CCW cornering
        data) can pull the parameters into degenerate basins — specifically the
        pEy3 sign-switch exploit — before the TTC data has had a chance to
        establish a sensible curve shape.  Running TTC-only first gives the
        tyre model a strong initialisation that the subsequent combined loss
        merely refines rather than disrupts.
        """
        if n_warmup <= 0:
            return
        opt = torch.optim.Adam(self.model.tire.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_warmup, eta_min=self.lr * 0.1)
        t0 = time.time()
        print(f"\n[Phase-0 TTC warmup]  {n_warmup} epochs  "
              f"(TTC shape only — no vehicle constraints)")
        for epoch in range(n_warmup):
            idx = self._sample_idx(ttc_t["alpha"].shape[0], self.batch_ttc)
            batch = {k: v[idx] for k, v in ttc_t.items()}
            opt.zero_grad(set_to_none=True)
            Fy_pred = self.model.tire(batch["alpha"], batch["Fz"])
            loss_ttc = torch.mean((Fy_pred - batch["Fy"]) ** 2) / (1000.0 ** 2)
            # Same physics-grounded regularisation as the full loss so that the
            # warmup cannot slip into any degenerate basin.
            # Weights kept in sync with pinn_loss() regulariser block.
            # Note: mu_scale is NOT a tire parameter, so it is not trained
            # here and needs no penalty in the warmup.
            reg = (self.model.tire.pEy3 ** 2 * 150.0               # sign-switch suppression
                   + self.model.tire.pEy1 ** 2 * 0.1
                   + self.model.tire.pEy2 ** 2 * 3.0               # RAISED 0.01→3.0
                   + self.model.tire.pVy1 ** 2 * 50.0              # spurious ay at alpha=0
                   + self.model.tire.pVy2 ** 2 * 50.0              # RAISED 10→50
                   + self.model.tire.pHy1 ** 2 * 30.0              # zero-crossing shift
                   + self.model.tire.pHy2 ** 2 * 30.0
                   + torch.relu(self.model.tire.pDy2) ** 2 * 50.0  # degressive friction
                   # pCy1 soft box [1.40, 1.60] — LOWER BOUND RAISED from 1.2.
                   # At 1.2 the relu(1.2-1.2)=0 created a zero-cost attractor.
                   # With hard clamp also at 1.40, optimizer cannot park at 1.2.
                   + torch.relu(self.model.tire.pCy1 - 1.60) ** 2 * 300.0
                   + torch.relu(1.40 - self.model.tire.pCy1) ** 2 * 300.0
                   # pEy2 — weight RAISED 0.01→3.0 + hard box [-0.5, +0.5] w=200.
                   # Old cost at pEy2=2.74: 7.5e-5 (free). New: 2.74²×3×1e-3=2.25e-2.
                   # Box enforces Ey never clamps to 0.99 at normal FSAE Fz range.
                   + self.model.tire.pEy2 ** 2 * 3.0
                   + torch.relu( self.model.tire.pEy2 - 0.5) ** 2 * 200.0
                   + torch.relu(-0.5 - self.model.tire.pEy2) ** 2 * 200.0
                   # pDy1 (mu_y) soft box [1.5, 2.5] — must match pinn_loss weight=300.
                   + torch.relu(self.model.tire.pDy1 - 2.5) ** 2 * 300.0
                   + torch.relu(1.5 - self.model.tire.pDy1) ** 2 * 300.0
                   # pKy1 soft box [55, 90] — targets peak alpha 5-7 deg.
                   + torch.relu(self.model.tire.pKy1 - 90.0) ** 2 * 5.0
                   + torch.relu(55.0 - self.model.tire.pKy1) ** 2 * 5.0
                   # pVy2 — weight RAISED 10→50. Old: 2e-4 effective. New: 1e-3.
                   + self.model.tire.pVy2 ** 2 * 50.0
                   # pKy2 box [0.8, 3.0] — prevent extreme stiffness load-sensitivity.
                   + torch.relu(self.model.tire.pKy2 - 3.0) ** 2 * 20.0
                   + torch.relu(0.8 - self.model.tire.pKy2) ** 2 * 20.0)
            loss = loss_ttc + self.spec.w_reg * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.tire.parameters(), 5.0)
            opt.step()
            sched.step()
            if verbose and epoch % 200 == 0:
                print(f"  warmup {epoch:>4d}  ttc={loss_ttc.item():.5f}  "
                      f"reg={float(reg.detach()):.4f}  t={time.time()-t0:.1f}s")
        print(f"  [Phase-0] complete  ({time.time()-t0:.1f}s)")

    # ------------- public entry -------
    def train(self, tel_t: Dict[str, torch.Tensor], ttc_t: Dict[str, torch.Tensor],
              verbose: bool = True, n_warmup: int = 800) -> None:
        self._ttc_warmup(ttc_t, n_warmup=n_warmup, verbose=verbose)
        self._adam(tel_t, ttc_t, verbose)
        self._lbfgs(tel_t, ttc_t, verbose)
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        print(f"\n[train] finished - best loss = {self._best_loss:.6f}")


# ============================================================================
#  §7  POST-PROCESSING
# ============================================================================

class PostProcessor:
    DPI = 150

    def __init__(self, model: VehiclePINN, out_dir: str):
        self.m = model
        self.out = out_dir
        os.makedirs(self.out, exist_ok=True)
        self._clear_previous_plots()

    @staticmethod
    def _style_axis(ax, *, legend: bool = False) -> None:
        """Apply publication-style cosmetics to 2D axes only."""
        ax.minorticks_on()
        ax.grid(True, which="major", color="#B0B0B0", alpha=0.35, linewidth=0.9)
        ax.grid(True, which="minor", color="#D9D9D9", alpha=0.25, linewidth=0.6, linestyle=":")
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.0, length=5)
        ax.tick_params(axis="both", which="minor", width=0.8, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#444444")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        if legend:
            leg = ax.legend(frameon=True, framealpha=0.9, fontsize=9)
            if leg is not None:
                leg.get_frame().set_edgecolor("#BBBBBB")
                leg.get_frame().set_linewidth(0.8)

    @staticmethod
    def _style_3d_axis(ax) -> None:
        """Apply consistent styling to 3D surfaces."""
        ax.tick_params(axis="x", which="major", labelsize=9, pad=4)
        ax.tick_params(axis="y", which="major", labelsize=9, pad=4)
        ax.tick_params(axis="z", which="major", labelsize=9, pad=4)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
            axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
            axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.45)
            axis._axinfo["grid"]["linewidth"] = 0.7

    def _clear_previous_plots(self) -> None:
        """
        Remove previous numbered plot PNGs so each run writes a fresh set.
        """
        patt = os.path.join(self.out, "[0-9][0-9]_*.png")
        for p in glob.glob(patt):
            try:
                os.remove(p)
            except OSError:
                # Best effort cleanup; plot save will still overwrite if needed.
                pass

    def _save(self, fig, name):
        p = os.path.join(self.out, name)
        fig.savefig(p, dpi=self.DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] {p}")

    # ---------- 1. Tyre curves ----------
    def plot_tire_curves(self, Fz_levels=(200, 400, 600, 800, 1000, 1200)):
        alpha_deg = np.linspace(-25, 25, 500)
        alpha = torch.tensor(np.deg2rad(alpha_deg), dtype=torch.float64, device=DEVICE)
        fig, ax = plt.subplots(figsize=(9, 6))
        cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(Fz_levels)))
        pt_to_px = fig.dpi / 72.0
        placed_label_boxes = []
        with torch.no_grad():
            for i, (Fz, c) in enumerate(zip(Fz_levels, cols)):
                Fz_t = torch.full_like(alpha, float(Fz))
                Fy = self.m.tire(alpha, Fz_t).cpu().numpy()
                ax.plot(alpha_deg, Fy, color=c, lw=2.2, label=f"Fz = {Fz} N")

                # Mark positive-branch peak only: alpha > 0 and Fy > 0.
                valid = (alpha_deg > 0.0) & (Fy > 0.0)
                if np.any(valid):
                    idx_rel = int(np.argmax(Fy[valid]))
                    idx_peak = int(np.flatnonzero(valid)[idx_rel])
                    a_peak = float(alpha_deg[idx_peak])
                    fy_peak = float(Fy[idx_peak])
                    ax.plot(a_peak, fy_peak, marker="o", ms=5, color=c)
                    ax.axvline(a_peak, color=c, lw=0.9, ls="--", alpha=0.45)
                    ax.axhline(fy_peak, color=c, lw=0.9, ls="--", alpha=0.35)
                    # Keep labels north-east of points; vary distance only.
                    candidate_offsets = [
                        (8, 10),
                        (12, 14),
                        (16, 18),
                        (20, 22),
                        (24, 26),
                    ]
                    label_w_px = 118.0
                    label_h_px = 24.0
                    margin_px = 6.0
                    peak_px = np.array(ax.transData.transform((a_peak, fy_peak)), dtype=float)
                    xoff, yoff = candidate_offsets[0]
                    for cand_xoff, cand_yoff in candidate_offsets:
                        center_px = peak_px + np.array([cand_xoff, cand_yoff]) * pt_to_px
                        half_w = label_w_px / 2.0
                        half_h = label_h_px / 2.0
                        box = (
                            center_px[0] - half_w - margin_px,
                            center_px[0] + half_w + margin_px,
                            center_px[1] - half_h - margin_px,
                            center_px[1] + half_h + margin_px,
                        )
                        overlaps = any(
                            not (
                                box[1] < prev[0]
                                or box[0] > prev[1]
                                or box[3] < prev[2]
                                or box[2] > prev[3]
                            )
                            for prev in placed_label_boxes
                        )
                        if not overlaps:
                            xoff, yoff = cand_xoff, cand_yoff
                            placed_label_boxes.append(box)
                            break
                    else:
                        center_px = peak_px + np.array([xoff, yoff]) * pt_to_px
                        half_w = label_w_px / 2.0
                        half_h = label_h_px / 2.0
                        placed_label_boxes.append(
                            (
                                center_px[0] - half_w - margin_px,
                                center_px[0] + half_w + margin_px,
                                center_px[1] - half_h - margin_px,
                                center_px[1] + half_h + margin_px,
                            )
                        )
                    ax.annotate(
                        f"({a_peak:.2f} deg, {fy_peak:.0f} N)",
                        xy=(a_peak, fy_peak),
                        xytext=(xoff, yoff),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color="#1f2937",
                        fontweight="semibold",
                        bbox={
                            "boxstyle": "round,pad=0.25,rounding_size=0.15",
                            "facecolor": "white",
                            "edgecolor": c,
                            "linewidth": 0.9,
                            "alpha": 0.92,
                        },
                        arrowprops={
                            "arrowstyle": "-",
                            "color": c,
                            "lw": 0.6,
                            "alpha": 0.55,
                            "shrinkA": 6,
                            "shrinkB": 6,
                        },
                    )
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("slip angle  alpha  [deg]")
        ax.set_ylabel("lateral force  Fy  [N]")
        ax.set_title("MF6.2 discovered tyre curves  (Fy vs alpha)")
        self._style_axis(ax, legend=False)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
        self._save(fig, "01_tire_curves.png")

    def plot_mf_vs_nn_kmenas(self, preds: Dict, Fz_levels=tuple(range(200, 1301, 100)), n_clusters: Optional[int] = None):
        """Plot MF6.2 curves and overlay NN-predicted Fy points clustered by Fz using K-means.

        preds: dict with keys 'alpha' (tuple of 4 arrays [rad]), 'Fz' (tuple of 4 arrays [N]), 'Fy' (tuple of 4 arrays [N])
        """
        if n_clusters is None:
            n_clusters = len(Fz_levels)

        # evaluation grid for MF curves
        alpha_deg = np.linspace(-15, 15, 500)
        alpha_rad = torch.tensor(np.deg2rad(alpha_deg), dtype=torch.float64, device=DEVICE)
        fig, ax = plt.subplots(figsize=(9, 6))
        cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(Fz_levels)))

        # Flatten NN predictions across wheels and convert alpha to degrees if needed
        alpha_list = []
        for v in preds["alpha"]:
            arr = np.asarray(v)
            if arr.size == 0:
                continue
            if np.max(np.abs(arr)) < 2 * np.pi:
                alpha_list.append(np.degrees(arr))
            else:
                alpha_list.append(arr)
        if len(alpha_list) == 0:
            alpha_flat = np.array([])
        else:
            alpha_flat = np.concatenate(alpha_list)
        Fz_flat = np.concatenate([np.asarray(v) for v in preds["Fz"]]) if len(preds["Fz"]) > 0 else np.array([])
        Fy_flat = np.concatenate([np.asarray(v) for v in preds["Fy"]]) if len(preds["Fy"]) > 0 else np.array([])

        # run K-means on vertical load Fz
        try:
            centroids, labels = kmeans2(Fz_flat.reshape(-1, 1).astype(np.float64), n_clusters, minit="points")
        except Exception:
            centroids = np.array([np.mean(Fz_flat) if Fz_flat.size else 0.0 for _ in range(n_clusters)])
            labels = np.zeros(Fz_flat.shape[0], dtype=int)

        centroids = np.asarray(centroids).ravel()
        # rank centroids so colors follow increasing load
        ranks = np.argsort(np.argsort(centroids))
        # guard empty case
        if Fz_flat.size:
            point_colors = cols[ranks[labels]]
        else:
            point_colors = cols[0]

        # faint background scatter of NN predictions (cluster colored)
        if alpha_flat.size and Fy_flat.size:
            ax.scatter(alpha_flat, Fy_flat, s=10, c=point_colors, alpha=0.18, marker='.', linewidths=0, rasterized=True, zorder=1)

        # plot MF curves on top
        with torch.no_grad():
            for i, (Fz, c) in enumerate(zip(Fz_levels, cols)):
                Fz_t = torch.full_like(alpha_rad, float(Fz))
                Fy_m = self.m.tire(alpha_rad, Fz_t).cpu().numpy()
                ax.plot(alpha_deg, Fy_m, color=c, lw=2.2, label=f"Fz = {Fz} N", zorder=3)

        # prepare cluster legend entries (sorted by centroid)
        cluster_centers_sorted = np.sort(centroids)
        cluster_handles = []
        cluster_labels = []
        for i in range(n_clusters):
            cluster_handles.append(ax.scatter([], [], c=[cols[i]], alpha=0.6, s=40, marker='o', linewidths=0))
            cluster_labels.append(f"cluster {i+1}: Fz≈{cluster_centers_sorted[i]:.0f} N")

        line_handles, line_labels = ax.get_legend_handles_labels()
        ax.legend(cluster_handles + line_handles, cluster_labels + line_labels, loc="lower right", frameon=True, framealpha=0.9, fontsize=8)

        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("slip angle  alpha  [deg]")
        ax.set_ylabel("lateral force  Fy  [N]")
        ax.set_title("MF6.2 vs NN-predicted Fy  (K-means load clusters)")
        self._style_axis(ax, legend=False)
        self._save(fig, "10_mf_vs_nn_kmenas.png")

    # ---------- 2. Surface ----------
    def plot_tire_surface(self):
        alpha_deg = np.linspace(-15, 15, 80)
        Fz_grid = np.linspace(300, 2200, 80)
        A, F = np.meshgrid(np.deg2rad(alpha_deg), Fz_grid)
        with torch.no_grad():
            Fy = self.m.tire(
                torch.tensor(A.ravel(), device=DEVICE),
                torch.tensor(F.ravel(), device=DEVICE),
            ).cpu().numpy().reshape(A.shape)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        vmax = float(np.percentile(np.abs(Fy), 98))
        surf = ax.plot_surface(
            alpha_deg,
            F,
            Fy,
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.12, label="Fy  [N]")

        ax.set_xlabel("alpha  [deg]")
        ax.set_ylabel("Fz  [N]")
        ax.set_zlabel("Fy  [N]")
        ax.set_title("Fy(alpha, Fz) 3D surface")
        ax.view_init(elev=28, azim=-130)
        self._style_3d_axis(ax)
        self._save(fig, "02_tire_surface.png")

    # ---------- 3. TTC fit ----------
    def plot_ttc_fit(self, ttc_df: pd.DataFrame):
        sources = sorted(ttc_df["source"].unique())
        n = len(sources)
        # Layout: up to 3 columns, as many rows as needed
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        axes_flat = axes.ravel()
        with torch.no_grad():
            for ax, source in zip(axes_flat, sources):
                sub = ttc_df[ttc_df["source"] == source]
                alpha = torch.tensor(sub["alpha"].values, device=DEVICE)
                Fz    = torch.tensor(sub["Fz"].values,    device=DEVICE)
                Fy_m  = self.m.tire(alpha, Fz).cpu().numpy()
                ax.scatter(np.degrees(sub["alpha"]), sub["Fy"], s=6, c="gray",
                           alpha=0.4, label="TTC raw")
                order = np.argsort(sub["alpha"].values)
                ax.plot(np.degrees(sub["alpha"].values[order]), Fy_m[order],
                        "r-", lw=2.0, label="MF6.2 fit")
                ax.set_xlabel("alpha  [deg]")
                ax.set_ylabel("Fy  [N]")
                ax.set_title(f"{source}\nFz ~ {sub['Fz'].mean():.0f} N")
                self._style_axis(ax, legend=True)
        # Hide any unused subplot panels
        for ax in axes_flat[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        self._save(fig, "03_ttc_fit.png")

    # ---------- 4-6. per-wheel time series ----------
    @staticmethod
    def _rolling_mean(x: np.ndarray, w: int = 50) -> np.ndarray:
        """Centred rolling mean; edges are filled with the first/last valid value."""
        if len(x) < w:
            return x
        kern = np.ones(w) / w
        out = np.convolve(x, kern, mode="same")
        # convolve "same" uses zero-padding at the edges which biases the mean;
        # replace edge bands with a cumsum-based causal/anti-causal mean instead.
        half = w // 2
        cs = np.cumsum(x)
        for i in range(half):
            n = i + half + 1
            out[i] = cs[min(n, len(x)) - 1] / min(n, len(x))
        for i in range(len(x) - half, len(x)):
            n = len(x) - (i - half)
            out[i] = (cs[-1] - (cs[i - half - 1] if i - half - 1 >= 0 else 0)) / n
        return out

    def plot_per_wheel(self, tel_df: pd.DataFrame, preds: Dict):
        # Use the monotonic global time axis when available; fall back to the
        # per-run local time (which would show the horizontal-line artefact).
        t = (tel_df["time_global"].values
             if "time_global" in tel_df.columns
             else tel_df["time"].values)
        t_label = "global time [s]" if "time_global" in tel_df.columns else "time [s]"
        labels = ("FL", "FR", "RL", "RR")
        ROLL_W = 50    # rolling-mean window (samples)
        ALPHA_RAW = 0.25  # transparency of raw trace

        # Fz
        fig, ax = plt.subplots(figsize=(12, 5))
        for lab, val in zip(labels, preds["Fz"]):
            ln, = ax.plot(t, val, lw=0.8, alpha=ALPHA_RAW)
            ax.plot(t, self._rolling_mean(val, ROLL_W),
                    color=ln.get_color(), lw=2.0, label=lab)
        ax.set_xlabel(t_label)
        ax.set_ylabel("Fz [N]")
        ax.set_title("Vertical load per wheel  (P26 LoadTransfer_v2)\n"
                     "thin=raw  thick=50-sample rolling mean")
        self._style_axis(ax, legend=True)
        self._save(fig, "04_fz_per_wheel.png")

        # alpha
        if self.m.p.steer_is_roadwheel:
            # Plot-only override requested by user:
            # keep full pipeline behaviour as configured, but render slip angles
            # using the alternative steering interpretation (steer_is_roadwheel=False).
            delta_plot = tel_df["steering"].values / self.m.p.steer_ratio * np.pi / 180.0
            with torch.no_grad():
                aFL_p, aFR_p, aRL_p, aRR_p = slip_angles(
                    torch.as_tensor(tel_df["u_x"].values, dtype=torch.float64, device=DEVICE),
                    torch.as_tensor(tel_df["u_y"].values, dtype=torch.float64, device=DEVICE),
                    torch.as_tensor(tel_df["yaw_rate"].values, dtype=torch.float64, device=DEVICE),
                    torch.as_tensor(delta_plot, dtype=torch.float64, device=DEVICE),
                    self.m.p,
                )
            alpha_plot_deg = tuple(np.degrees(v.cpu().numpy()) for v in (aFL_p, aFR_p, aRL_p, aRR_p))
        else:
            alpha_plot_deg = tuple(np.degrees(v) for v in preds["alpha"])

        fig, ax = plt.subplots(figsize=(12, 5))
        for lab, deg in zip(labels, alpha_plot_deg):
            ln, = ax.plot(t, deg, lw=0.8, alpha=ALPHA_RAW)
            ax.plot(t, self._rolling_mean(deg, ROLL_W),
                    color=ln.get_color(), lw=2.0, label=lab)
        ax.set_xlabel(t_label)
        ax.set_ylabel("alpha [deg]")
        ax.set_title("Slip angle per wheel  (rigid-body kinematics)\n"
                     "thin=raw  thick=50-sample rolling mean")
        self._style_axis(ax, legend=True)
        self._save(fig, "05_alpha_per_wheel.png")

        # Fy
        fig, ax = plt.subplots(figsize=(12, 5))
        for lab, val in zip(labels, preds["Fy"]):
            ln, = ax.plot(t, val, lw=0.8, alpha=ALPHA_RAW)
            ax.plot(t, self._rolling_mean(val, ROLL_W),
                    color=ln.get_color(), lw=2.0, label=lab)
        ax.set_xlabel(t_label)
        ax.set_ylabel("Fy [N]")
        ax.set_title("Lateral force per wheel  (MF6.2)\n"
                     "thin=raw  thick=50-sample rolling mean")
        self._style_axis(ax, legend=True)
        self._save(fig, "06_fy_per_wheel.png")

    # ---------- 7. vehicle balance ----------
    def plot_vehicle_balance(self, tel_df: pd.DataFrame, preds: Dict,
                             r_dot_thresh: float = 2.0):
        """Plot lateral and yaw acceleration balance.

        Two subplots per quantity:
        - Left panel: all samples (full transparency so spikes are visible)
        - Right panel: steady-state only (|r_dot| < r_dot_thresh), which is
          the subset that the loss function actually trained on.

        The steady-state panel is the physically meaningful comparison because
        the PINN is a steady-state model — it correctly cannot represent the
        transient entry/exit dynamics of the skidpad.  Plotting all samples
        conflates model quality with the unmodelled transient regime.
        """
        m = self.m.p.m_tot
        Iz = self.m.p.Iz
        lf = self.m.p.lf
        lr = self.m.p.lr
        Tf = self.m.p.Tf

        delta = tel_df["delta_rad"].values
        cd = np.cos(delta)
        sd = np.sin(delta)
        Fy_FL, Fy_FR, Fy_RL, Fy_RR = preds["Fy"]
        Fy_sum = (Fy_FL + Fy_FR) * cd + (Fy_RL + Fy_RR)
        ay_pred = Fy_sum / m
        Mz = (lf * ((Fy_FL + Fy_FR) * cd) - lr * (Fy_RL + Fy_RR)
              + 0.5 * Tf * ((Fy_FL - Fy_FR) * sd))
        rdot_pred = Mz / Iz

        t = (tel_df["time_global"].values
             if "time_global" in tel_df.columns else tel_df["time"].values)
        t_label = "global time [s]" if "time_global" in tel_df.columns else "time [s]"

        # Steady-state mask — same threshold used during training.
        r_dot_arr = tel_df["r_dot"].values
        ss = np.abs(r_dot_arr) < r_dot_thresh
        n_ss = ss.sum()
        ss_frac = 100.0 * n_ss / max(len(t), 1)

        ay_meas = tel_df["a_y"].values
        ay_bias_ss = float(np.mean(ay_pred[ss] - ay_meas[ss])) if n_ss > 5 else float("nan")
        ay_rmse_ss = float(np.sqrt(np.mean((ay_pred[ss] - ay_meas[ss]) ** 2))) if n_ss > 5 else float("nan")
        rd_bias_ss = float(np.mean(rdot_pred[ss] - r_dot_arr[ss])) if n_ss > 5 else float("nan")
        rd_rmse_ss = float(np.sqrt(np.mean((rdot_pred[ss] - r_dot_arr[ss]) ** 2))) if n_ss > 5 else float("nan")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # ---- Row 0: all samples ----
        ax = axes[0, 0]
        ax.plot(t, ay_meas, "tab:blue", lw=0.8, alpha=0.5, label="measured")
        ax.plot(t, ay_pred, "tab:red",  lw=0.8, alpha=0.5, label="PINN")
        ax.set_xlabel(t_label); ax.set_ylabel("ay [m/s²]")
        ax.set_title("Lateral acceleration — ALL samples (incl. transients)")
        self._style_axis(ax, legend=True)

        ax = axes[0, 1]
        ax.plot(t, r_dot_arr, "tab:blue", lw=0.8, alpha=0.5, label="measured")
        ax.plot(t, rdot_pred, "tab:red",  lw=0.8, alpha=0.5, label="PINN")
        ax.set_xlabel(t_label); ax.set_ylabel("r_dot [rad/s²]")
        ax.set_title("Yaw acceleration — ALL samples (incl. transients)")
        self._style_axis(ax, legend=True)

        # ---- Row 1: steady-state only (training-relevant) ----
        ax = axes[1, 0]
        ax.scatter(t[ss], ay_meas[ss], s=4, c="tab:blue", alpha=0.4, label="measured")
        ax.scatter(t[ss], ay_pred[ss], s=4, c="tab:red",  alpha=0.4, label="PINN")
        ax.set_xlabel(t_label); ax.set_ylabel("ay [m/s²]")
        ax.set_title(
            f"Lateral — STEADY-STATE only  |r_dot|<{r_dot_thresh} rad/s²  "
            f"({ss_frac:.0f}% of data)\n"
            f"bias={ay_bias_ss:+.2f} m/s²  RMSE={ay_rmse_ss:.2f} m/s²"
        )
        self._style_axis(ax, legend=True)

        ax = axes[1, 1]
        ax.scatter(t[ss], r_dot_arr[ss], s=4, c="tab:blue", alpha=0.4, label="measured")
        ax.scatter(t[ss], rdot_pred[ss], s=4, c="tab:red",  alpha=0.4, label="PINN")
        ax.set_xlabel(t_label); ax.set_ylabel("r_dot [rad/s²]")
        ax.set_title(
            f"Yaw — STEADY-STATE only  |r_dot|<{r_dot_thresh} rad/s²  "
            f"({ss_frac:.0f}% of data)\n"
            f"bias={rd_bias_ss:+.2f} rad/s²  RMSE={rd_rmse_ss:.2f} rad/s²"
        )
        self._style_axis(ax, legend=True)

        fig.tight_layout()
        self._save(fig, "07_vehicle_balance.png")

    # ---------- 8. kamm ----------
    def plot_kamm(self, preds: Dict):
        labels = ("FL", "FR", "RL", "RR")
        mu = 1.8
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax, lab, Fy, Fz in zip(axes, labels, preds["Fy"], preds["Fz"]):
            Fx = np.zeros_like(Fy)          # pure lateral in this dataset
            util = np.abs(Fy) / (mu * Fz + 1e-6)
            sc = ax.scatter(Fx, Fy, c=util, cmap="YlOrRd", s=10, vmin=0, vmax=1)
            theta = np.linspace(0, 2 * np.pi, 200)
            Fzm = float(np.mean(Fz))
            ax.plot(mu * Fzm * np.cos(theta), mu * Fzm * np.sin(theta),
                    "k--", lw=1.2, alpha=0.6, label=f"mu={mu}")
            ax.set_aspect("equal")
            lim = mu * Fzm * 1.2
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Fx [N]")
            ax.set_ylabel("Fy [N]")
            ax.set_title(f"wheel {lab}")
            self._style_axis(ax, legend=True)
            plt.colorbar(sc, ax=ax, label="util")
        self._save(fig, "08_kamm_circle.png")

    # ---------- 9. training history ----------
    def plot_history(self, h: Dict[str, List[float]]):
        if not h["epoch"]:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, c in (("total", "k"), ("ttc", "tab:blue"),
                     ("lat", "tab:orange"), ("yaw", "tab:green"), ("reg", "tab:purple")):
            ax.semilogy(h["epoch"], h[k], c, label=k, lw=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (log)")
        ax.set_title("Training history")
        self._style_axis(ax, legend=True)
        self._save(fig, "09_training_history.png")

    # ---------- 10. MF curves vs NN scatter (K-means) ----------
    def plot_mf_vs_nn_kmeans(self, preds: Dict,
                             Fz_levels=tuple(range(200, 1301, 100))):
        """Overlay MF6.2 Fy-vs-alpha curves on top of faded NN-predicted Fy
        scatter points, K-means clustered by vertical load Fz."""
        from matplotlib.lines import Line2D

        n_clusters = len(Fz_levels)
        Fz_levels = np.asarray(Fz_levels, dtype=float)

        # Gather all 4 wheels into flat arrays
        wheels = ("FL", "FR", "RL", "RR")
        alpha_all = np.concatenate([np.degrees(preds["alpha"][i]) for i in range(4)])
        Fz_all    = np.concatenate([preds["Fz"][i]               for i in range(4)])
        Fy_all    = np.concatenate([preds["Fy"][i]               for i in range(4)])

        # K-means on Fz (1-D)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(Fz_all.reshape(-1, 1))
        centroids = km.cluster_centers_.ravel()

        # Assign each cluster to closest MF Fz level (monotonic mapping)
        cluster_to_fz_idx = {}
        sorted_cl = np.argsort(centroids)
        sorted_fz = np.argsort(Fz_levels)
        for ci, fi in zip(sorted_cl, sorted_fz):
            cluster_to_fz_idx[ci] = fi

        # Colours (same viridis slice as plot 01)
        cmap_vals = np.linspace(0.15, 0.9, n_clusters)
        fz_colors = plt.cm.viridis(cmap_vals)

        fig, ax = plt.subplots(figsize=(11, 7))

        # Background: faded scatter per cluster
        for cl_id in range(n_clusters):
            mask = labels == cl_id
            fz_idx = cluster_to_fz_idx[cl_id]
            ax.scatter(alpha_all[mask], Fy_all[mask], s=4,
                       color=fz_colors[fz_idx], alpha=0.12,
                       edgecolors="none", rasterized=True)

        # Foreground: MF6.2 analytic curves
        alpha_deg_line = np.linspace(-15, 15, 500)
        alpha_rad_line = torch.tensor(np.deg2rad(alpha_deg_line),
                                      dtype=torch.float64, device=DEVICE)
        placed_label_boxes = []
        pt_to_px = fig.dpi / 72.0

        with torch.no_grad():
            for i, Fz_val in enumerate(Fz_levels):
                c = fz_colors[i]
                Fz_t = torch.full_like(alpha_rad_line, float(Fz_val))
                Fy_line = self.m.tire(alpha_rad_line, Fz_t).cpu().numpy()
                ax.plot(alpha_deg_line, Fy_line, color=c, lw=2.4,
                        label=f"MF  Fz = {int(Fz_val)} N", zorder=5)

                # Peak annotation (positive branch)
                valid = (alpha_deg_line > 0.0) & (Fy_line > 0.0)
                if np.any(valid):
                    idx_rel = int(np.argmax(Fy_line[valid]))
                    idx_peak = int(np.flatnonzero(valid)[idx_rel])
                    a_pk = float(alpha_deg_line[idx_peak])
                    fy_pk = float(Fy_line[idx_peak])
                    ax.plot(a_pk, fy_pk, marker="o", ms=5, color=c, zorder=6)
                    ax.axvline(a_pk, color=c, lw=0.9, ls="--", alpha=0.35, zorder=3)
                    ax.axhline(fy_pk, color=c, lw=0.9, ls="--", alpha=0.25, zorder=3)

                    candidates = [(8, 10), (12, 14), (16, 18), (20, 22), (24, 26)]
                    lw_px, lh_px, margin = 120.0, 24.0, 6.0
                    peak_px = np.array(ax.transData.transform((a_pk, fy_pk)), dtype=float)
                    xoff, yoff = candidates[0]
                    for cxo, cyo in candidates:
                        ctr = peak_px + np.array([cxo, cyo]) * pt_to_px
                        hw, hh = lw_px / 2.0, lh_px / 2.0
                        box = (ctr[0]-hw-margin, ctr[0]+hw+margin,
                               ctr[1]-hh-margin, ctr[1]+hh+margin)
                        if not any(
                            not (box[1]<p[0] or box[0]>p[1] or box[3]<p[2] or box[2]>p[3])
                            for p in placed_label_boxes
                        ):
                            xoff, yoff = cxo, cyo
                            placed_label_boxes.append(box)
                            break
                    else:
                        ctr = peak_px + np.array([xoff, yoff]) * pt_to_px
                        hw, hh = lw_px / 2.0, lh_px / 2.0
                        placed_label_boxes.append(
                            (ctr[0]-hw-margin, ctr[0]+hw+margin,
                             ctr[1]-hh-margin, ctr[1]+hh+margin)
                        )
                    ax.annotate(
                        f"({a_pk:.1f} deg, {fy_pk:.0f} N)",
                        xy=(a_pk, fy_pk), xytext=(xoff, yoff),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=8, color="#1f2937", fontweight="semibold",
                        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15",
                                  facecolor="white", edgecolor=c,
                                  linewidth=0.9, alpha=0.92),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.6,
                                        alpha=0.55, shrinkA=6, shrinkB=6),
                        zorder=7)

        ax.axhline(0, color="k", lw=0.6, ls=":", zorder=2)
        ax.axvline(0, color="k", lw=0.6, ls=":", zorder=2)
        ax.set_xlabel("slip angle  alpha  [deg]")
        ax.set_ylabel("lateral force  Fy  [N]")
        ax.set_title(
            "MF6.2 tyre curves  vs  NN-predicted operating points\n"
            "(scatter: K-Means clustered by Fz  |  lines: Magic Formula)")

        # Custom legend
        handles = []
        for i, Fz_val in enumerate(Fz_levels):
            cl_id = [k for k, v in cluster_to_fz_idx.items() if v == i][0]
            n_pts = int((labels == cl_id).sum())
            ctr_val = centroids[cl_id]
            handles.append(Line2D([0], [0], color=fz_colors[i], lw=2.4,
                                  label=f"MF  Fz = {int(Fz_val)} N"))
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=fz_colors[i], markersize=6,
                                  alpha=0.55,
                                  label=f"cluster Fz~{ctr_val:.0f} N (n={n_pts})"))
        ax.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.92, fontsize=8.5, ncol=1)
        self._style_axis(ax)
        fig.tight_layout()
        self._save(fig, "10_mf_vs_nn_kmeans.png")


# ============================================================================
#  §8  MAIN
# ============================================================================

def save_coefficients(model: VehiclePINN, out_dir: str) -> None:
    coeffs = model.tire.coefficients()
    coeffs["mu_scale_front"] = float(model.mu_scale_f.detach().cpu().item())
    coeffs["mu_scale_rear"]  = float(model.mu_scale_r.detach().cpu().item())
    coeffs["vy_offset_ms"]   = float(model.vy_offset.detach().cpu().item())
    coeffs["Iz_kgm2"]        = float(model.p.Iz)  # fixed, not learnable
    # Json
    with open(os.path.join(out_dir, "mf62_coefficients.json"), "w") as f:
        json.dump(coeffs, f, indent=2)
    # Csv
    pd.DataFrame({"param": list(coeffs.keys()),
                  "value": list(coeffs.values())}) \
      .to_csv(os.path.join(out_dir, "mf62_coefficients.csv"), index=False)
    print(f"  [save] mf62_coefficients.json / .csv")


def save_metrics(model: VehiclePINN, ttc_df: pd.DataFrame, out_dir: str) -> None:
    """Calculates B, C, D, E and other metrics at nominal load Fz0."""
    with torch.no_grad():
        Fz0 = float(model.tire.Fz0.cpu().item())
        pCy1 = float(model.tire.pCy1.cpu().item())
        pDy1 = float(model.tire.pDy1.cpu().item())
        pKy1 = float(model.tire.pKy1.cpu().item())
        pKy2 = float(model.tire.pKy2.cpu().item())
        pEy1 = float(model.tire.pEy1.cpu().item())

        # MF6.2 -> basic MF at Fz0, gamma=0
        D = pDy1 * Fz0
        C = pCy1
        # Kya = Cornering Stiffness [N/rad]
        Kya = pKy1 * Fz0 * math.sin(2.0 * math.atan(Fz0 / (pKy2 * Fz0 + 1e-9)))
        B = Kya / (C * D + 1e-9)
        E = pEy1

        C_alpha_rad = Kya
        C_alpha_deg = Kya * math.pi / 180.0

        # find peak and alpha_peak via grid search at Fz0
        a_eval = torch.linspace(0, 0.5, 1000, device=DEVICE)
        Fz_eval = torch.full_like(a_eval, Fz0)
        Fy_eval = model.tire(a_eval, Fz_eval).cpu().numpy()
        idx_peak = np.argmax(np.abs(Fy_eval))
        Peak_Fy = float(np.abs(Fy_eval[idx_peak]))
        alpha_peak_deg = float(np.degrees(a_eval[idx_peak].cpu().item()))

        # R2 of TTC fit
        alpha_t = torch.tensor(ttc_df["alpha"].values, device=DEVICE)
        Fz_t    = torch.tensor(ttc_df["Fz"].values,    device=DEVICE)
        Fy_pred = model.tire(alpha_t, Fz_t).cpu().numpy()
        Fy_true = ttc_df["Fy"].values
        ss_res = np.sum((Fy_pred - Fy_true)**2)
        ss_tot = np.sum((Fy_true - np.mean(Fy_true))**2) + 1e-9
        r2_ttc = 1.0 - ss_res / ss_tot

    metrics = {
        "B": B, "C": C, "D": D, "E": E,
        "C_alpha_N_per_deg": C_alpha_deg,
        "C_alpha_N_per_rad": C_alpha_rad,
        "Peak_Fy_N": Peak_Fy,
        "alpha_peak_deg": alpha_peak_deg,
        "Pacejka_R2": r2_ttc,
        "NN_train_R2": 0.0, "NN_val_R2": 0.0, "NN_test_R2": 0.0, "NN_test_RMSE": 0.0,
        "B_CI_lo": 0.0, "B_CI_hi": 0.0,
        "C_CI_lo": 0.0, "C_CI_hi": 0.0,
        "D_CI_lo": 0.0, "D_CI_hi": 0.0,
        "E_CI_lo": 0.0, "E_CI_hi": 0.0,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "tire_model_metrics.csv"), index=False)
    print(f"  [save] tire_model_metrics.csv")


def save_timeseries(tel_df: pd.DataFrame, preds: Dict, out_dir: str) -> None:
    base_cols = ["run", "time", "steering", "u_x", "u_y",
                 "yaw_rate", "r_dot", "a_x", "a_y"]
    if "time_global" in tel_df.columns:
        base_cols = ["run", "time", "time_global", "steering", "u_x", "u_y",
                     "yaw_rate", "r_dot", "a_x", "a_y"]
    out = tel_df[base_cols].copy()
    for lab, v in zip(("FL", "FR", "RL", "RR"), preds["Fz"]):
        out[f"Fz_{lab}"] = v
    for lab, v in zip(("FL", "FR", "RL", "RR"), preds["alpha"]):
        out[f"alpha_{lab}_deg"] = np.degrees(v)
    for lab, v in zip(("FL", "FR", "RL", "RR"), preds["Fy"]):
        out[f"Fy_{lab}"] = v
    out.to_csv(os.path.join(out_dir, "per_wheel_timeseries.csv"), index=False)
    print(f"  [save] per_wheel_timeseries.csv  ({len(out)} rows)")


def write_summary(model, tel_df, preds, ttc_df, loss_parts, out_dir):
    m = model.p.m_tot
    lines = []
    lines.append("=" * 66)
    lines.append(" FSAE P26 - PINN tyre discovery  summary")
    lines.append(f" {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 66)
    lines.append("")
    lines.append("Vehicle (P26):")
    lines.append(f"  m_tot  = {m:.1f} kg   (NSM={model.p.NSM}, SM={model.p.SM})")
    lines.append(f"  Iz     = {model.p.Iz:.1f} kg.m^2")
    lines.append(f"  WB     = {model.p.WB:.3f} m   lf={model.p.lf:.3f}  lr={model.p.lr:.3f}")
    lines.append(f"  Tracks = {model.p.Tf:.3f}/{model.p.Tr:.3f} m   hCG={model.p.hcog_SM:.3f}")
    lines.append("")
    lines.append("Data:")
    lines.append(f"  telemetry rows : {len(tel_df)}")
    lines.append(f"  TTC rows       : {len(ttc_df)}")
    lines.append("")
    lines.append("Final loss components:")
    for k, v in loss_parts.items():
        lines.append(f"  {k:8s} = {v:.6f}")
    lines.append("")

    # MF6.2 metrics
    alpha_t = torch.tensor(ttc_df["alpha"].values, device=DEVICE)
    Fz_t    = torch.tensor(ttc_df["Fz"].values,    device=DEVICE)
    with torch.no_grad():
        Fy_m = model.tire(alpha_t, Fz_t).cpu().numpy()
    Fy_t = ttc_df["Fy"].values
    rmse_ttc = float(np.sqrt(np.mean((Fy_m - Fy_t) ** 2)))
    ss_res   = float(np.sum((Fy_m - Fy_t) ** 2))
    ss_tot   = float(np.sum((Fy_t - Fy_t.mean()) ** 2)) + 1e-9
    r2_ttc   = 1.0 - ss_res / ss_tot
    lines.append("TTC fit quality:")
    lines.append(f"  RMSE = {rmse_ttc:.1f} N")
    lines.append(f"  R^2  = {r2_ttc:.4f}")
    lines.append("")

    # vehicle balance metrics
    ay_meas = tel_df["a_y"].values
    cd = np.cos(tel_df["delta_rad"].values)
    sd = np.sin(tel_df["delta_rad"].values)
    Fy_sum = (preds["Fy"][0] + preds["Fy"][1]) * cd + (preds["Fy"][2] + preds["Fy"][3])
    ay_pred = Fy_sum / m
    rmse_ay = float(np.sqrt(np.mean((ay_pred - ay_meas) ** 2)))
    r2_ay = 1.0 - np.sum((ay_pred - ay_meas) ** 2) / (np.sum((ay_meas - ay_meas.mean()) ** 2) + 1e-9)
    lines.append("Lateral-balance prediction (ay_pred vs ay_meas):")
    lines.append(f"  RMSE = {rmse_ay:.3f} m/s^2")
    lines.append(f"  R^2  = {r2_ay:.4f}")
    lines.append("")
    lines.append("Peak |Fy| per wheel:")
    for lab, Fy in zip(("FL", "FR", "RL", "RR"), preds["Fy"]):
        lines.append(f"  {lab}: {np.max(np.abs(Fy)):.0f} N")
    lines.append("")
    lines.append("Peak Fz per wheel:")
    for lab, Fz in zip(("FL", "FR", "RL", "RR"), preds["Fz"]):
        lines.append(f"  {lab}: {np.max(Fz):.0f} N   min={np.min(Fz):.0f} N")
    lines.append("")
    lines.append("MF6.2 coefficients (pure lateral):")
    for k, v in model.tire.coefficients().items():
        lines.append(f"  {k:12s} = {v:+.6f}")
    lines.append(f"  mu_scale_front = {model.mu_scale_f.detach().cpu().item():+.4f}")
    lines.append(f"  mu_scale_rear  = {model.mu_scale_r.detach().cpu().item():+.4f}")
    lines.append(f"  vy_offset      = {model.vy_offset.detach().cpu().item():+.4f} m/s")
    lines.append(f"  Iz_param       = {model.p.Iz:.2f} kg*m^2  (fixed)")
    lines.append("")
    lines.append("=" * 66)

    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)



# ----------------------------------------------------------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FSAE P26 PINN tyre discovery  (PyTorch, MF6.2 pure lateral)"
    )
    p.add_argument("--tel", nargs="+", required=False,
                   help="Telemetry CSVs (10.5 m, 9 m, and 6 m skidpad runs)")
    p.add_argument("--ttc", nargs="+", required=False,
                   help="Raw Calspan TTC CSVs (pure lateral)")
    p.add_argument("--out", default="pinn_p26_results",
                   help="Output directory")
    p.add_argument("--epochs",   type=int,   default=5000,
                   help="Adam epochs in combined phase (default 5000; was 4000)")
    p.add_argument("--warmup",   type=int,   default=1200,
                   help="Phase-0 TTC-only Adam epochs before combined loss (default 1200; "
                        "was 800). Longer warmup gives tire params a stronger prior so the "
                        "combined phase refines rather than disrupts the TTC fit.")
    p.add_argument("--lbfgs",    type=int,   default=800,
                   help="L-BFGS iterations for final refinement (default 800; was 600)")
    p.add_argument("--batch-tel", type=int,  default=2048)
    p.add_argument("--batch-ttc", type=int,  default=1024)
    p.add_argument("--lr",       type=float, default=3e-3)
    p.add_argument("--w-ttc",    type=float, default=3.0,
                   help="TTC fit weight (70%% mu-normalised + 30%% direct force space)")
    p.add_argument("--w-lat",    type=float, default=8.0,
                   help="Lateral balance weight (default 8.0; was 1.0). "
                        "Previous value was catastrophically too low: the optimizer found a "
                        "degenerate solution (vy_offset=+3 m/s, mu_scale=0.5) that produced "
                        "ay R²=-0.10 while ignoring the balance constraint entirely. "
                        "At 8.0 the balance equations dominate over the vy_offset/mu_scale "
                        "degeneracy, forcing physical solutions.")
    p.add_argument("--w-yaw",    type=float, default=6.0,
                   help="Yaw balance weight (default 6.0; was 1.0). "
                        "Yaw moment at 1.2 g skidpad: Mz ~ lf*Fy_f - lr*Fy_r ~ 300-400 Nm. "
                        "With Mchar=1000 Nm, the dimensionless yaw loss is ~0.1 per 100 Nm "
                        "residual — needs w_yaw=6 to match the lateral constraint gradient.")
    p.add_argument("--w-reg",    type=float, default=1e-3)
    p.add_argument("--r-dot-thresh", type=float, default=1.0,
                   help="Gaussian kernel width [rad/s^2] for soft SS weighting (default 1.0; "
                        "was 2.0). At r_dot=1.0 the Gaussian weight = exp(-1)=0.37; at 2.0 "
                        "it is 0.02 — effectively excluded. Tighter threshold concentrates "
                        "the balance loss on true steady-state laps, giving cleaner gradients "
                        "for vy_offset and mu_scale.")
    p.add_argument("--ttc-scale", type=float, default=1.0,
                   help="Global Fy scale applied to ALL TTC data before training "
                        "(default 1.0 = no scaling).  Use values < 1 to down-weight "
                        "the TTC force level (e.g. 0.9 if TTC mu is known to be ~10%% "
                        "higher than on-track), or > 1 to scale up.  Only Fy is "
                        "scaled; Fz and alpha are unchanged.")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


def main():
    args = _cli()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- automatic data discovery if not provided ---
    if not args.tel:
        # Collect all three skidpad radius datasets: 10.5 m, 9 m, 6 m
        tel_patterns = [
            os.path.join("data", "Testing_Data", "processed_10_5m_*.csv"),
            os.path.join("data", "Testing_Data", "processed_9m*.csv"),
            os.path.join("data", "Testing_Data", "processed_6m_*.csv"),
        ]
        args.tel = []
        for patt in tel_patterns:
            args.tel.extend(glob.glob(patt))
        args.tel = sorted(args.tel)
        if args.tel:
            print(f"[auto] found {len(args.tel)} telemetry files in data/Testing_Data/")
        else:
            print(f"Error: No telemetry files found matching patterns in data/Testing_Data/")
            sys.exit(1)

    if not args.ttc:
        pattern = os.path.join("data", "TTC data", "*.csv")
        args.ttc = sorted(glob.glob(pattern))
        if args.ttc:
            print(f"[auto] found {len(args.ttc)} TTC files in data/TTC data/")
        else:
            print(f"Error: No TTC files found matching {pattern}")
            sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print("=" * 66)
    print("P26  PINN TYRE DISCOVERY  (PyTorch, MF6.2 pure lateral)")
    print(f" device  : {DEVICE}")
    print(f" out dir : {args.out}")
    print("=" * 66)

    # --- load data ------------------------------------------------------
    print("\n[load] telemetry")
    tel_df = load_telemetry(args.tel)
    print("\n[load] TTC")
    ttc_df = load_ttc(args.ttc)

    # --- TTC Fy scaling (--ttc-scale) ------------------------------------
    # Applies a global scalar to all TTC Fy values before any further
    # processing.  Useful when the TTC test conditions (surface mu, tyre
    # temperature, inflation pressure) are known to differ systematically
    # from the on-track conditions captured in the telemetry.
    # Only Fy is touched — Fz and alpha are physical measurements that
    # should not be altered.
    if args.ttc_scale != 1.0:
        ttc_df = ttc_df.copy()
        ttc_df["Fy"] = ttc_df["Fy"] * args.ttc_scale
        print(f"  [TTC] Fy scaled by {args.ttc_scale:.4f}  "
              f"(Fy range after: [{ttc_df['Fy'].min():.0f}, {ttc_df['Fy'].max():.0f}] N)")

    # --- build tensors --------------------------------------------------
    tel_tensors = {
        "Vx":    torch.tensor(tel_df["u_x"].values,       device=DEVICE),
        "Vy":    torch.tensor(tel_df["u_y"].values,       device=DEVICE),
        "r":     torch.tensor(tel_df["yaw_rate"].values,  device=DEVICE),
        "r_dot": torch.tensor(tel_df["r_dot"].values,     device=DEVICE),
        "delta": torch.tensor(tel_df["delta_rad"].values, device=DEVICE),
        "ax":    torch.tensor(tel_df["a_x"].values,       device=DEVICE),
        "ay":    torch.tensor(tel_df["a_y"].values,       device=DEVICE),
    }
    ttc_tensors = {
        "alpha":     torch.tensor(ttc_df["alpha"].values,     device=DEVICE),
        "Fz":        torch.tensor(ttc_df["Fz"].values,        device=DEVICE),
        "Fy":        torch.tensor(ttc_df["Fy"].values,        device=DEVICE),
        "gamma":     torch.tensor(ttc_df["gamma"].values,     device=DEVICE),
        "source_id": torch.tensor(ttc_df["source_id"].values, device=DEVICE),
    }

    # MF6.2 nominal reference load Fz0.
    # Equal-weight mean over source files so each load level contributes equally.
    if "source" in ttc_df.columns:
        per_source_fz = ttc_df.groupby("source")["Fz"].mean().abs()
        Fz0 = float(per_source_fz.mean())
    else:
        Fz0 = 1000.9
    print(f"\n[init] MF6.2 reference load  Fz0 = {Fz0:.0f} N  "
          f"(equal-weight mean across TTC files)")

    # Yaw-moment normalisation: fixed at 1000 Nm.
    Mchar = 1000.0
    print(f"[init] yaw moment scale  Mchar = {Mchar:.0f} Nm  (fixed; was ~1971 Nm)")

    # --- build model ----------------------------------------------------
    model = VehiclePINN(P26, Fz0=Fz0).to(DEVICE)
    spec = LossSpec(w_ttc=args.w_ttc, w_lat=args.w_lat,
                    w_yaw=args.w_yaw, w_reg=args.w_reg,
                    Mchar=Mchar, r_dot_thresh=args.r_dot_thresh)

    # --- train ----------------------------------------------------------
    trainer = Trainer(model, spec,
                      n_adam=args.epochs, n_lbfgs=args.lbfgs,
                      batch_tel=args.batch_tel, batch_ttc=args.batch_ttc,
                      lr=args.lr)
    trainer.train(tel_tensors, ttc_tensors, verbose=True, n_warmup=args.warmup)

    # --- inference on full telemetry ------------------------------------
    model.eval()
    with torch.no_grad():
        preds_t = model.predict_wheels(
            tel_tensors["Vx"], tel_tensors["Vy"],
            tel_tensors["r"],  tel_tensors["delta"],
            tel_tensors["ax"], tel_tensors["ay"],
        )
    preds_np = {
        "Fz":    tuple(v.cpu().numpy() for v in preds_t["Fz"]),
        "alpha": tuple(v.cpu().numpy() for v in preds_t["alpha"]),
        "Fy":    tuple(v.cpu().numpy() for v in preds_t["Fy"]),
    }

    # final loss for summary
    with torch.no_grad():
        _, final_parts = pinn_loss(model, tel_tensors, ttc_tensors, spec)

    # --- save + plots ---------------------------------------------------
    save_coefficients(model, args.out)
    save_metrics(model, ttc_df, args.out)
    save_timeseries(tel_df, preds_np, args.out)
    pp = PostProcessor(model, args.out)
    pp.plot_tire_curves()
    pp.plot_tire_surface()
    pp.plot_ttc_fit(ttc_df)
    pp.plot_per_wheel(tel_df, preds_np)
    pp.plot_vehicle_balance(tel_df, preds_np, r_dot_thresh=spec.r_dot_thresh)
    pp.plot_kamm(preds_np)
    pp.plot_history(trainer.history)
    pp.plot_mf_vs_nn_kmeans(preds_np)

    torch.save(model.state_dict(), os.path.join(args.out, "pinn_model.pt"))
    print(f"  [save] pinn_model.pt")

    write_summary(model, tel_df, preds_np, ttc_df, final_parts, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
