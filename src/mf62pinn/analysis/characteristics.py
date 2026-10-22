"""Characteristic values and confidence bands of identified tyres (delta method)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..fit.common import DTYPE, delta_method
from ..tyre import PARAM_NAMES, MF62Lateral, characteristics, mf62_fy

KEYS = ("Kya_N_per_deg", "mu_peak", "Fy_peak_N", "alpha_peak_deg", "B", "C", "D", "E")


def _theta_from_active(model: MF62Lateral, x: torch.Tensor, act_idx: List[int]) -> torch.Tensor:
    u = model.u.detach().clone().index_put((torch.tensor(act_idx),), x)
    return model.theta(u)


def table(model: MF62Lateral, loads: Sequence[float], cov_u: Optional[np.ndarray] = None,
          lmuy: float = 1.0, lky: float = 1.0, lmuy_sd: float = 0.0) -> pd.DataFrame:
    """Per-load characteristic values (both curve sides) with 1-sd uncertainties."""
    act_idx = [i for i, n in enumerate(PARAM_NAMES) if n not in model.frozen]
    x0 = model.u.detach()[act_idx]
    cov = torch.tensor(cov_u, dtype=DTYPE) if cov_u is not None else torch.zeros(len(act_idx), len(act_idx), dtype=DTYPE)
    rows = []
    for Fz in loads:
        Fz_t = torch.tensor(float(Fz), dtype=DTYPE)
        for side in (+1, -1):
            def f(x: torch.Tensor) -> torch.Tensor:
                c = characteristics(_theta_from_active(model, x, act_idx), Fz_t, model.Fz0,
                                    lmuy=lmuy, lky=lky, side=side, ey_beta=model.ey_beta)
                return torch.stack([c[k] for k in KEYS])
            mean, sd = delta_method(f, x0, cov)
            row = {"Fz_N": float(Fz), "side": "+" if side > 0 else "-"}
            for k, m, s in zip(KEYS, mean, sd):
                row[k] = float(m)
                row[k + "_sd"] = float(s)
            if lmuy_sd > 0:   # friction-scale uncertainty (independent, first order)
                for k in ("mu_peak", "Fy_peak_N"):
                    rel = lmuy_sd / lmuy
                    row[k + "_sd"] = float(np.hypot(row[k + "_sd"], abs(row[k]) * rel))
            rows.append(row)
    return pd.DataFrame(rows)


def curve_band(model: MF62Lateral, alpha_deg: np.ndarray, Fz: float, cov_u: Optional[np.ndarray],
               lmuy: float = 1.0, lky: float = 1.0):
    """Mean and 1-sd band of ``Fy(alpha)`` at one load."""
    act_idx = [i for i, n in enumerate(PARAM_NAMES) if n not in model.frozen]
    x0 = model.u.detach()[act_idx]
    a = torch.tensor(np.radians(alpha_deg), dtype=DTYPE)
    Fz_t = torch.full_like(a, float(Fz))

    def f(x: torch.Tensor) -> torch.Tensor:
        return mf62_fy(_theta_from_active(model, x, act_idx), a, Fz_t, model.Fz0, lmuy=lmuy, lky=lky,
                       sign_eps=model.sign_eps, ey_beta=model.ey_beta)
    if cov_u is None:
        with torch.no_grad():
            return f(x0).numpy(), np.zeros_like(alpha_deg)
    return delta_method(f, x0, torch.tensor(cov_u, dtype=DTYPE))


def coefficient_table(values: Dict[str, float], std: Dict[str, float], frozen: Sequence[str],
                      specs) -> pd.DataFrame:
    """Coefficient, 1-sd, relative error and identifiability status."""
    rows = []
    for n in PARAM_NAMES:
        v = values[n]
        s = std.get(n, float("nan"))
        sp = specs[n]
        if n in frozen:
            status = "frozen (no data gradient)"
        else:
            span = sp.hi - sp.lo
            near = min(v - sp.lo, sp.hi - v) / span < 0.02
            prior_dom = np.isfinite(s) and s > 0.5 * sp.prior_sd
            status = "near bound" if near else ("prior-dominated" if prior_dom else "identified")
        rows.append({"param": n, "value": v, "sd": s,
                     "rel_sd_%": 100 * s / abs(v) if (np.isfinite(s) and abs(v) > 1e-12) else float("nan"),
                     "lo": sp.lo, "hi": sp.hi, "status": status})
    return pd.DataFrame(rows)
