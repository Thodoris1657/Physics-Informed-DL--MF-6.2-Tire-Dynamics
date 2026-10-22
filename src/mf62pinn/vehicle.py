"""
Vehicle physics: wheel geometry, rigid-body slip kinematics, load transfer
and the steady-state force/moment balances.

Frame
-----
All vehicle quantities use one right-handed SAE body frame: ``x`` forward,
``y`` to the right, ``z`` down. Consequently

* the left wheels sit at ``y = -T/2`` and the right wheels at ``y = +T/2``;
* ``r < 0`` and ``ay < 0`` is a left-hand turn (the outer = right wheels load up);
* the slip angle used by the tyre model is ``alpha = delta - atan(v_y/v_x)`` of
  the contact point, i.e. ``alpha = -alpha_SAE``, and a positive ``alpha``
  produces a positive (rightward, SAE) lateral force.

All three blocks below (kinematics, load transfer, balances) read the wheel
positions from :func:`wheel_positions`, so the geometry is defined exactly once.
This removes the class of sign inconsistencies identified in the v1 audit
(steer-induced yaw moment with FL on the wrong side).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import VehicleParams

WHEELS = ("FL", "FR", "RL", "RR")


def wheel_positions(p: VehicleParams) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Contact-patch positions ``(x_i, y_i)`` relative to the CoG, SAE frame."""
    x = (p.lf, p.lf, -p.lr, -p.lr)
    y = (-0.5 * p.Tf, 0.5 * p.Tf, -0.5 * p.Tr, 0.5 * p.Tr)
    return x, y


def _smooth_floor(x: torch.Tensor, floor: float, sharpness: float = 0.05) -> torch.Tensor:
    """C-infinity ``max(x, floor)``; identical to ``x`` a few 1/sharpness above the floor."""
    return floor + F.softplus(x - floor, beta=sharpness)


# --------------------------------------------------------------------------- #
#  Load transfer  (faithful port of MATLAB LoadTransfer_v2)
# --------------------------------------------------------------------------- #


def load_transfer(Vx: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor,
                  p: VehicleParams, smooth_floor_N: float = 0.0
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wheel loads ``(Fz_FL, Fz_FR, Fz_RL, Fz_RR)`` [N].

    Static loads -> longitudinal transfer (sprung + unsprung) -> Milliken's
    lateral transfer (elastic + geometric + unsprung) -> aero downforce with a
    roll-dependent ClA. Identical to the v1 implementation (which reproduces the
    team MATLAB routine); the only change is an optional *smooth* floor in place
    of the hard ``max(Fz, 0)``, so that gradients never vanish.
    """
    m_tot, SM, NSM = p.m_tot, p.SM, p.NSM
    WD, WD_NSM, g, WB = p.WD_tot, p.WD_NSM, p.g, p.WB
    a = WB * (1.0 - WD)
    b = WB - a
    hRC_at_CoG = WD * p.hRC_f + (1.0 - WD) * p.hRC_r

    Fz_f_s = 0.5 * m_tot * WD * g
    Fz_r_s = 0.5 * m_tot * (1.0 - WD) * g

    LT_x = 0.5 * (NSM * p.hcog_NSM + SM * p.hcog_SM) * ax / WB   # + to rear

    q = p.hcog_SM - hRC_at_CoG
    susp_roll_stif = SM * q / p.roll_rate_deg_per_g - SM * q          # [N m/deg]*
    Kf = susp_roll_stif * p.MB * 180.0 / math.pi
    Kr = susp_roll_stif * (1.0 - p.MB) * 180.0 / math.pi
    KappaF = Kf - b * SM * q / WB
    KappaR = Kr - a * SM * q / WB
    NSM_f = NSM * WD_NSM
    NSM_r = NSM * (1.0 - WD_NSM)
    denom = Kf + Kr - SM * q
    LT_y_f = (SM / p.Tf * (q * KappaF / denom + WD * p.hRC_f) + NSM_f / p.Tf * p.hcog_NSM) * ay
    LT_y_r = (SM / p.Tr * (q * KappaR / denom + WD * p.hRC_r) + NSM_r / p.Tr * p.hcog_NSM) * ay

    roll_deg = torch.abs(ay) / g * p.roll_rate_deg_per_g
    clA = p.clA_nom - p.roll_loss * roll_deg
    q_dyn = 0.5 * p.rho * clA * Vx * Vx
    F_aero_f = p.AD_lat * p.AD_long * q_dyn
    F_aero_r = p.AD_lat * (1.0 - p.AD_long) * q_dyn

    # ay < 0 (left turn, SAE) -> LT_y < 0 -> left wheels unload, right wheels load
    Fz_FL = Fz_f_s - LT_x + LT_y_f + F_aero_f
    Fz_FR = Fz_f_s - LT_x - LT_y_f + F_aero_f
    Fz_RL = Fz_r_s + LT_x + LT_y_r + F_aero_r
    Fz_RR = Fz_r_s + LT_x - LT_y_r + F_aero_r

    if smooth_floor_N > 0.0:
        return tuple(_smooth_floor(z, smooth_floor_N) for z in (Fz_FL, Fz_FR, Fz_RL, Fz_RR))  # type: ignore
    zero = torch.zeros((), dtype=Vx.dtype, device=Vx.device)
    return tuple(torch.maximum(z, zero) for z in (Fz_FL, Fz_FR, Fz_RL, Fz_RR))  # type: ignore


# --------------------------------------------------------------------------- #
#  Kinematics
# --------------------------------------------------------------------------- #


def slip_angles(Vx: torch.Tensor, Vy: torch.Tensor, r: torch.Tensor,
                delta: torch.Tensor, p: VehicleParams
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rigid-body slip angle of every contact patch (tyre convention, see module doc).

    The contact-point velocity is ``(Vx - r*y_i, Vy + r*x_i)``; front wheels are
    steered by ``delta`` (parallel steer), rear wheels are not.
    """
    xs, ys = wheel_positions(p)
    Vxs = torch.where(torch.abs(Vx) < 1e-3, torch.full_like(Vx, 1e-3), Vx)
    out = []
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        steer = delta if i < 2 else torch.zeros_like(delta)
        out.append(steer - torch.atan2(Vy + r * xi, Vxs - r * yi))
    return tuple(out)  # type: ignore


def body_forces(Fy: Tuple[torch.Tensor, ...], delta: torch.Tensor, p: VehicleParams
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Body-frame lateral force and yaw moment of the four tyre lateral forces.

    A steered wheel's lateral force ``F_i`` has body components
    ``(-F_i sin(delta), F_i cos(delta))``; its moment about the CoG is
    ``x_i*F_y - y_i*F_x = F_i (x_i cos(delta) + y_i sin(delta))``.
    """
    xs, ys = wheel_positions(p)
    cd, sd = torch.cos(delta), torch.sin(delta)
    Fy_body = 0.0
    Mz = 0.0
    for i, F_i in enumerate(Fy):
        if i < 2:
            Fy_body = Fy_body + F_i * cd
            Mz = Mz + F_i * (xs[i] * cd + ys[i] * sd)
        else:
            Fy_body = Fy_body + F_i
            Mz = Mz + F_i * xs[i]
    return Fy_body, Mz  # type: ignore


def rear_drive_force(Vx: torch.Tensor, ax: torch.Tensor, p: VehicleParams
                     ) -> Tuple[torch.Tensor, ...]:
    """
    Per-wheel longitudinal force needed to produce ``ax`` against aero drag
    (open-differential split, used only for friction-ellipse de-rating).
    Returns ``(Fx_FL, Fx_FR, Fx_RL, Fx_RR)``.
    """
    Fx_tot = p.m_tot * ax + 0.5 * p.rho * p.CdA * Vx * Vx
    z = torch.zeros_like(Fx_tot)
    if p.drive == "front":
        return 0.5 * Fx_tot, 0.5 * Fx_tot, z, z
    if p.drive == "awd":
        return (0.25 * Fx_tot,) * 4
    return z, z, 0.5 * Fx_tot, 0.5 * Fx_tot


def static_summary(p: VehicleParams) -> Dict[str, float]:
    """Handy derived numbers for reports."""
    return {
        "m_tot_kg": p.m_tot, "lf_m": p.lf, "lr_m": p.lr, "WB_m": p.WB,
        "Fz_front_static_N": 0.5 * p.m_tot * p.WD_tot * p.g,
        "Fz_rear_static_N": 0.5 * p.m_tot * (1 - p.WD_tot) * p.g,
        "Iz_kgm2": p.Iz,
    }
