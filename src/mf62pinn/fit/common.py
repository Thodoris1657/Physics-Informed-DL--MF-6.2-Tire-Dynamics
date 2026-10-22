"""Shared numerics: robust likelihoods, optimiser loops, Laplace covariance."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

DTYPE = torch.float64


def nll_residual(z: torch.Tensor, kind: str = "student_t", nu: float = 4.0) -> torch.Tensor:
    """Per-sample negative log-density of a *standardised* residual (no log-sigma term)."""
    if kind == "gaussian":
        return 0.5 * z * z
    if kind == "student_t":
        return 0.5 * (nu + 1.0) * torch.log1p(z * z / nu)
    if kind == "huber":
        a = torch.abs(z)
        return torch.where(a < 1.345, 0.5 * z * z, 1.345 * (a - 0.5 * 1.345))
    raise ValueError(kind)


def ar1_inflation(resid: np.ndarray, groups: Optional[np.ndarray] = None) -> float:
    """
    Variance-inflation factor ``N / N_eff`` for serially correlated residuals
    (AR(1) approximation, pooled over groups). Used to de-bias Laplace
    covariances computed from 100 Hz rig sweeps, whose samples are far from
    independent.
    """
    resid = np.asarray(resid, float)
    if groups is None:
        groups = np.zeros(len(resid), int)
    num = den = 0.0
    for g in np.unique(groups):
        e = resid[groups == g]
        e = e - e.mean()
        if len(e) < 3:
            continue
        num += float(np.dot(e[1:], e[:-1]))
        den += float(np.dot(e, e))
    rho = float(np.clip(num / max(den, 1e-30), 0.0, 0.995))
    return (1.0 + rho) / (1.0 - rho)


def run_lbfgs(params: List[torch.Tensor], closure_loss: Callable[[], torch.Tensor],
              max_iter: int = 500, history: int = 50, tol: float = 1e-12,
              log: Optional[List[float]] = None) -> float:
    """Full-batch L-BFGS with strong-Wolfe line search. Returns the final loss."""
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, history_size=history,
                            tolerance_grad=1e-10, tolerance_change=tol,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = closure_loss()
        loss.backward()
        if log is not None:
            log.append(float(loss.detach()))
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(closure_loss())


def run_adam(params: List[torch.Tensor], closure_loss: Callable[[], torch.Tensor],
             iters: int, lr: float = 0.05, log: Optional[List[float]] = None,
             log_every: int = 10) -> None:
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=lr * 0.02)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = closure_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        if log is not None and it % log_every == 0:
            log.append(float(loss.detach()))


def hessian(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Dense Hessian of a scalar function (small problems, float64)."""
    x = x.detach().clone().requires_grad_(True)
    return torch.autograd.functional.hessian(fn, x)


def laplace_cov(H: torch.Tensor, jitter: float = 1e-10) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Covariance from a (possibly indefinite) Hessian via a symmetric eigen
    decomposition. Non-positive / tiny eigenvalues are lifted and reported -
    they are the non-identifiable directions.
    """
    H = 0.5 * (H + H.T)
    # rows that are identically zero belong to masked (frozen) entries: give them
    # unit curvature so they neither pollute the spectrum nor the covariance
    dead = torch.all(H.abs() < 1e-300, dim=1)
    if bool(dead.any()):
        H = H.clone()
        H[dead, dead] = 1.0
    w, V = torch.linalg.eigh(H)
    floor = max(float(w.abs().max()) * 1e-12, jitter)
    flagged = int((w <= floor).sum())
    w_c = torch.clamp(w, min=floor)
    cov = (V / w_c) @ V.T
    info = {"eigvals": w.detach().cpu().numpy(), "n_nonpositive": flagged,
            "condition": float(w_c.max() / w_c.min())}
    return cov, info


def delta_method(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                 cov: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of ``fn(x)`` under ``x ~ N(x, cov)`` (first order)."""
    x = x.detach().clone()
    y = fn(x)
    J = torch.autograd.functional.jacobian(fn, x)
    J = J.reshape(y.numel(), x.numel())
    var = torch.einsum("ij,jk,ik->i", J, cov, J)
    return y.detach().cpu().numpy().reshape(y.shape), torch.sqrt(torch.clamp(var, min=0)).cpu().numpy().reshape(y.shape)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else float("nan")


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def sobol(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc
    m = int(math.ceil(math.log2(max(n, 2))))
    s = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)
    return s[:n]
