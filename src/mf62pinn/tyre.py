"""
Differentiable MF6.2 pure-lateral tyre model with a constrained,
identifiability-aware parameterisation.

Equations (Pacejka 2012, Besselink et al. 2010), scaling factors included::

    dfz  = (Fz - Fz0) / Fz0
    mu_y = (pDy1 + pDy2 dfz)(1 - pDy3 g^2) * LMUY
    Dy   = mu_y Fz
    Cy   = pCy1
    Kya  = pKy1 Fz0 sin(2 atan(Fz / (pKy2 Fz0))) * LKY
    By   = Kya / (Cy Dy)
    SHy  = pHy1 + pHy2 dfz + pHy3 g
    a_y  = alpha + SHy
    Ey   = (pEy1 + pEy2 dfz)(1 - (pEy3 + pEy4 g) sgn(a_y))         (<= 1)
    SVy  = Fz (pVy1 + pVy2 dfz + (pVy3 + pVy4 dfz) g) * LMUY'
    Fy   = Dy sin(Cy atan(By a_y - Ey (By a_y - atan(By a_y)))) + SVy

with ``LMUY' = 10 LMUY / (1 + 9 LMUY)`` (MF6.1/6.2 degressive friction
scaling, A_mu = 10). Internal sign convention: positive alpha -> positive Fy
(see :mod:`mf62pinn.vehicle`); :mod:`mf62pinn.export.tir` converts to ISO-W.

Design choices (v2)
-------------------
* **No hard clamps.** Every coefficient lives on an unconstrained real line and
  is mapped into its physical box with a scaled sigmoid, ``Ey <= 1`` is a smooth
  cap and ``sgn`` is a narrow ``tanh``. The optimiser therefore never meets a
  zero-gradient wall (v1 had attractors on the C_y and E_y clamps).
* **Identifiability-aware.** Parameters with no data gradient (camber terms
  when the data have no camber) are frozen automatically and reported as such.
* **Weakly-informative priors.** Each coefficient carries a Gaussian prior
  (``ParamSpec``); priors are expressed in physical units so their strength is
  interpretable, unlike v1's hand-tuned penalty walls.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CAMBER_PARAMS, ParamSpec, TyreConfig

PARAM_NAMES: List[str] = [
    "pCy1", "pDy1", "pDy2", "pDy3",
    "pEy1", "pEy2", "pEy3", "pEy4",
    "pKy1", "pKy2",
    "pHy1", "pHy2", "pHy3",
    "pVy1", "pVy2", "pVy3", "pVy4",
]

A_MU = 10.0  # MF6.x degressive-friction constant for LMUY'


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def lmuy_prime(lmuy: torch.Tensor) -> torch.Tensor:
    return A_MU * lmuy / (1.0 + (A_MU - 1.0) * lmuy)


class MF62Lateral(nn.Module):
    """MF6.2 pure-lateral force with bounded, prior-equipped coefficients."""

    def __init__(self, Fz0: float, cfg: Optional[TyreConfig] = None,
                 frozen: Iterable[str] = ()):
        super().__init__()
        cfg = cfg or TyreConfig()
        self.cfg = cfg
        self.specs: Dict[str, ParamSpec] = {k: v for k, v in cfg.params.items()}
        self.register_buffer("Fz0", torch.tensor(float(Fz0), dtype=torch.float64))
        frozen = set(frozen) | set(cfg.freeze) | {k for k, s in self.specs.items() if s.fixed}
        self.frozen = sorted(frozen)
        lo = torch.tensor([self.specs[n].lo for n in PARAM_NAMES], dtype=torch.float64)
        hi = torch.tensor([self.specs[n].hi for n in PARAM_NAMES], dtype=torch.float64)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        u0 = [_logit((self.specs[n].init - self.specs[n].lo) / (self.specs[n].hi - self.specs[n].lo))
              for n in PARAM_NAMES]
        self.u = nn.Parameter(torch.tensor(u0, dtype=torch.float64))
        mask = torch.tensor([0.0 if n in frozen else 1.0 for n in PARAM_NAMES], dtype=torch.float64)
        self.register_buffer("active_mask", mask)
        self.register_buffer("u_frozen", self.u.detach().clone())
        self.sign_eps = math.radians(cfg.sign_eps_deg)
        self.ey_beta = cfg.ey_cap_sharpness

    # ------------------------------------------------------------------ params
    @property
    def active_names(self) -> List[str]:
        return [n for n in PARAM_NAMES if n not in self.frozen]

    def u_effective(self, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        u = self.u if u is None else u
        m = self.active_mask
        return m * u + (1.0 - m) * self.u_frozen

    def theta(self, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Physical coefficient vector (ordered as ``PARAM_NAMES``)."""
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.u_effective(u))

    def values(self) -> Dict[str, float]:
        th = self.theta().detach().cpu().numpy()
        return {n: float(v) for n, v in zip(PARAM_NAMES, th)}

    def set_values(self, values: Dict[str, float]) -> None:
        """Set physical values (clipped strictly inside the box)."""
        with torch.no_grad():
            u = self.u.detach().clone()
            for i, n in enumerate(PARAM_NAMES):
                if n in values:
                    s = self.specs[n]
                    frac = (float(values[n]) - s.lo) / (s.hi - s.lo)
                    u[i] = _logit(min(max(frac, 1e-6), 1 - 1e-6))
            self.u.copy_(u)
            self.u_frozen.copy_(torch.where(self.active_mask > 0, self.u_frozen, u))

    def freeze(self, names: Iterable[str]) -> None:
        names = set(names)
        self.frozen = sorted(set(self.frozen) | names)
        with torch.no_grad():
            for i, n in enumerate(PARAM_NAMES):
                if n in names:
                    self.active_mask[i] = 0.0
                    self.u_frozen[i] = self.u[i]

    def prior_nll(self, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        th = self.theta(u)
        mu = torch.tensor([self.specs[n].prior_mu for n in PARAM_NAMES], dtype=th.dtype, device=th.device)
        sd = torch.tensor([self.specs[n].prior_sd for n in PARAM_NAMES], dtype=th.dtype, device=th.device)
        z = (th - mu) / sd
        return 0.5 * torch.sum(self.active_mask * z * z)

    # ------------------------------------------------------------------ physics
    def forward(self, alpha: torch.Tensor, Fz: torch.Tensor,
                gamma: Optional[torch.Tensor] = None,
                lmuy: torch.Tensor | float = 1.0, lky: torch.Tensor | float = 1.0,
                u: Optional[torch.Tensor] = None) -> torch.Tensor:
        return mf62_fy(self.theta(u), alpha, Fz, self.Fz0, gamma, lmuy, lky,
                       sign_eps=self.sign_eps, ey_beta=self.ey_beta)


def mf62_fy(theta: torch.Tensor, alpha: torch.Tensor, Fz: torch.Tensor, Fz0: torch.Tensor,
            gamma: Optional[torch.Tensor] = None, lmuy: torch.Tensor | float = 1.0,
            lky: torch.Tensor | float = 1.0, sign_eps: float = math.radians(0.25),
            ey_beta: float = 25.0) -> torch.Tensor:
    """Functional MF6.2 lateral force for a coefficient vector ``theta``."""
    (pCy1, pDy1, pDy2, pDy3, pEy1, pEy2, pEy3, pEy4,
     pKy1, pKy2, pHy1, pHy2, pHy3, pVy1, pVy2, pVy3, pVy4) = theta.unbind(-1)
    if gamma is None:
        gamma = torch.zeros_like(alpha)
    Fz = 30.0 + F.softplus(Fz - 30.0, beta=0.05)     # smooth Fz >= 30 N
    dfz = (Fz - Fz0) / Fz0
    lmuy = torch.as_tensor(lmuy, dtype=alpha.dtype, device=alpha.device)
    lky = torch.as_tensor(lky, dtype=alpha.dtype, device=alpha.device)

    mu = (pDy1 + pDy2 * dfz) * (1.0 - pDy3 * gamma * gamma) * lmuy
    Dy = mu * Fz
    Cy = pCy1
    Kya = pKy1 * Fz0 * torch.sin(2.0 * torch.atan(Fz / (pKy2 * Fz0))) * lky
    By = Kya / (Cy * Dy + 1e-9)
    SHy = pHy1 + pHy2 * dfz + pHy3 * gamma
    ay = alpha + SHy
    sgn = torch.tanh(ay / sign_eps)
    Ey_raw = (pEy1 + pEy2 * dfz) * (1.0 - (pEy3 + pEy4 * gamma) * sgn)
    Ey = 1.0 - F.softplus(1.0 - Ey_raw, beta=ey_beta)
    SVy = Fz * (pVy1 + pVy2 * dfz + (pVy3 + pVy4 * dfz) * gamma) * lmuy_prime(lmuy)
    Bx = By * ay
    phi = Bx - Ey * (Bx - torch.atan(Bx))
    return Dy * torch.sin(Cy * torch.atan(phi)) + SVy


# --------------------------------------------------------------------------- #
#  Characteristic values (differentiable)
# --------------------------------------------------------------------------- #


def characteristics(theta: torch.Tensor, Fz: torch.Tensor, Fz0: torch.Tensor,
                    lmuy: torch.Tensor | float = 1.0, lky: torch.Tensor | float = 1.0,
                    side: int = 1, ey_beta: float = 25.0, newton_iters: int = 30
                    ) -> Dict[str, torch.Tensor]:
    """
    B, C, D, E, cornering stiffness, peak force and peak slip angle at load
    ``Fz`` on the ``side`` (+1 / -1) of the curve. Fully differentiable, so the
    Laplace covariance can be pushed through by the delta method.

    The peak satisfies ``Cy atan(phi(u)) = pi/2`` with ``u = By a_y``; it is
    solved by Newton's method on ``phi(u) = tan(pi/(2 Cy))``.
    """
    (pCy1, pDy1, pDy2, pDy3, pEy1, pEy2, pEy3, pEy4,
     pKy1, pKy2, pHy1, pHy2, pHy3, pVy1, pVy2, pVy3, pVy4) = theta.unbind(-1)
    dfz = (Fz - Fz0) / Fz0
    lmuy = torch.as_tensor(lmuy, dtype=theta.dtype)
    lky = torch.as_tensor(lky, dtype=theta.dtype)
    mu = (pDy1 + pDy2 * dfz) * lmuy
    D = mu * Fz
    C = pCy1
    K = pKy1 * Fz0 * torch.sin(2.0 * torch.atan(Fz / (pKy2 * Fz0))) * lky
    B = K / (C * D)
    Ey_raw = (pEy1 + pEy2 * dfz) * (1.0 - pEy3 * side)
    E = 1.0 - F.softplus(1.0 - Ey_raw, beta=ey_beta)
    SHy = pHy1 + pHy2 * dfz
    SVy = Fz * (pVy1 + pVy2 * dfz) * lmuy_prime(lmuy)
    target = torch.tan(math.pi / (2.0 * C))
    u = target.clone()
    for _ in range(newton_iters):
        f = u - E * (u - torch.atan(u)) - target
        df = 1.0 - E * (1.0 - 1.0 / (1.0 + u * u))
        u = u - f / df
    alpha_peak = side * u / B - SHy
    Fy_peak = side * D + SVy
    return {
        "B": B, "C": C, "D": D, "E": E, "Kya_N_per_rad": K, "Kya_N_per_deg": K * math.pi / 180.0,
        "mu_peak": Fy_peak / Fz * side, "Fy_peak_N": Fy_peak, "alpha_peak_deg": alpha_peak * 180.0 / math.pi,
        "SHy_deg": SHy * 180.0 / math.pi, "SVy_N": SVy,
    }


def active_camber_check(gamma_rad: torch.Tensor, threshold_deg: float) -> bool:
    """True if the data excite camber enough to identify camber terms."""
    if gamma_rad.numel() == 0:
        return False
    span = float(gamma_rad.max() - gamma_rad.min())
    return math.degrees(span) >= threshold_deg


__all__ = ["MF62Lateral", "mf62_fy", "characteristics", "PARAM_NAMES", "CAMBER_PARAMS", "lmuy_prime"]
