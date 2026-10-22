"""
Steady-state cornering with the identified tyre: handling diagram, understeer
gradient and lateral grip limit on a constant radius.

For a given radius ``R`` and speed ``V`` (right-hand turn in the SAE frame,
``r = V/R``, ``ay = V^2/R``) we solve the two steady-state balances

    m ay = sum_i Fy_i cos(delta_i)            (lateral)
    0    = sum_i Fy_i (x_i cos d_i + y_i sin d_i)   (yaw)

for the CoG sideslip ``beta`` and the road-wheel steer ``delta`` with a damped
Newton method (exact 2x2 Jacobian by autograd), continuing in speed from the
kinematic solution. The last converged speed bounds the steady-state grip limit
at that radius; it is refined by bisection.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
import torch

from ..vehicle import WHEELS, body_forces

D2R = math.pi / 180.0


def _residual(model, V: torch.Tensor, R: float, beta: torch.Tensor, delta: torch.Tensor):
    veh = model.veh
    r = V / R
    ay = V * V / R
    Vy = V * torch.tan(beta)
    ax = torch.zeros_like(V)
    w = model.wheel_forces(V, Vy, r, delta, ax, ay)
    Fy_body, Mz = body_forces(w["Fy"], delta, veh)
    return torch.stack([(veh.m_tot * ay - Fy_body) / (veh.m_tot * veh.g), Mz / 1000.0]), w


def _solve_point(model, V: float, R: float, x0: torch.Tensor, iters: int = 40, tol: float = 1e-9):
    x = x0.clone()
    Vt = torch.tensor([V], dtype=torch.float64)
    for _ in range(iters):
        xr = x.clone().requires_grad_(True)
        F, _ = _residual(model, Vt, R, xr[0:1], xr[1:2])
        F = F.reshape(-1)
        J = torch.stack([torch.autograd.grad(F[i], xr, retain_graph=True)[0] for i in range(2)])
        Fd = F.detach()
        if float(Fd.abs().max()) < tol:
            return x, True
        try:
            step = torch.linalg.solve(J, -Fd)
        except RuntimeError:
            return x, False
        # damped update with backtracking
        lam = 1.0
        f0 = float(Fd.norm())
        for _ in range(12):
            xn = x + lam * step
            with torch.no_grad():
                Fn, _ = _residual(model, Vt, R, xn[0:1], xn[1:2])
            if float(Fn.norm()) < f0:
                break
            lam *= 0.5
        x = xn.detach()
        if abs(float(x[1])) > 45 * D2R or abs(float(x[0])) > 30 * D2R:
            return x, False
    with torch.no_grad():
        F, _ = _residual(model, Vt, R, x[0:1], x[1:2])
    return x, bool(F.abs().max() < 1e-6)


@torch.no_grad()
def _state(model, V: float, R: float, x: torch.Tensor) -> Dict[str, float]:
    Vt = torch.tensor([V], dtype=torch.float64)
    _, w = _residual(model, Vt, R, x[0:1], x[1:2])
    veh = model.veh
    out = {"V_ms": V, "ay_g": V * V / R / veh.g, "beta_deg": float(x[0]) / D2R,
           "delta_deg": float(x[1]) / D2R,
           "understeer_deg": (float(x[1]) - veh.WB / R) / D2R}
    for i, wn in enumerate(WHEELS):
        out[f"alpha_{wn}_deg"] = float(w["alpha"][i]) / D2R
        out[f"Fz_{wn}"] = float(w["Fz"][i])
        out[f"Fy_{wn}"] = float(w["Fy"][i])
    return out


def handling_diagram(model, R: float, vmin: float = 4.0, vmax: float = 16.0, n: int = 60
                     ) -> Dict[str, object]:
    """Constant-radius sweep. Returns the curve, the grip limit and Kus."""
    veh = model.veh
    speeds = np.linspace(vmin, vmax, n)
    x = torch.tensor([veh.lr / R, veh.WB / R], dtype=torch.float64)
    rows, last_ok = [], None
    for V in speeds:
        xs, ok = _solve_point(model, float(V), R, x)
        if not ok:
            break
        rows.append(_state(model, float(V), R, xs))
        x, last_ok = xs, float(V)
    curve = pd.DataFrame(rows)
    v_lim = last_ok
    if last_ok is not None and last_ok < speeds[-1]:
        lo, hi, x_lo = last_ok, float(speeds[min(len(rows), len(speeds) - 1)]), x
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            xm, ok = _solve_point(model, mid, R, x_lo)
            if ok:
                lo, x_lo = mid, xm
            else:
                hi = mid
        v_lim = lo
        rows.append(_state(model, lo, R, x_lo))
        curve = pd.DataFrame(rows)
    kus = float("nan")
    if len(curve) >= 4:
        lin = curve[curve["ay_g"] <= 0.5]
        if len(lin) >= 3:
            kus = float(np.polyfit(lin["ay_g"], lin["understeer_deg"], 1)[0])
    ay_lim = (v_lim ** 2 / R / veh.g) if v_lim else float("nan")
    return {"curve": curve, "radius_m": R, "v_limit_ms": v_lim, "ay_limit_g": ay_lim,
            "Kus_linear_deg_per_g": kus, "limited_by_sweep": bool(v_lim is not None and v_lim >= vmax - 1e-9)}


def measured_handling_points(seg: pd.DataFrame, WB: float, g: float) -> pd.DataFrame:
    """Measured steady-state points mapped to the right-turn convention."""
    turn = np.sign(seg["r"].to_numpy())
    us = (seg["delta"].to_numpy() - WB * seg["r"].to_numpy() / seg["Vx"].to_numpy()) * turn
    return pd.DataFrame({"run": seg["run"], "ay_g": np.abs(seg["ay"]) / g, "understeer_deg": us / D2R,
                         "radius_m": seg["Vx"] / np.abs(seg["r"])})


def grip_limit_by_run(model, seg: pd.DataFrame, g: float) -> pd.DataFrame:
    """Predicted steady-state grip limit at each run's own radius vs. what was driven."""
    rows = []
    for run, s in seg.groupby("run"):
        R = float(np.median(s["Vx"] / np.abs(s["r"])))
        hd = handling_diagram(model, R, vmin=4.0, vmax=25.0, n=50)
        rows.append({"run": run, "radius_m": R, "ay_limit_pred_g": hd["ay_limit_g"],
                     "ay_driven_p90_g": float(np.quantile(np.abs(s["ay"]), 0.9)) / g,
                     "ay_driven_mean_g": float(np.mean(np.abs(s["ay"]))) / g})
    return pd.DataFrame(rows)
