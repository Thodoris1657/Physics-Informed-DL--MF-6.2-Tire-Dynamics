"""
Stage A - rig identification of the MF6.2 lateral coefficients from TTC sweeps.

* **Objective**: robust (Student-t) likelihood of the residual in friction
  space, ``(Fy_hat - Fy)/Fz_load``, with every load level weighted equally,
  a learned noise scale, and the weak physical priors of the coefficients.
* **Serial correlation**: rig sweeps are logged at 100 Hz, so residuals are
  strongly autocorrelated. The data term is tempered by the AR(1)
  variance-inflation factor ``N/N_eff`` so that the posterior (and the Laplace
  covariance) reflects the *effective* amount of information.
* **Global search**: a batched Sobol multi-start (Adam on all starts in
  parallel) followed by full-batch L-BFGS on the best candidates. The spread
  of near-optimal starts is kept as an empirical picture of what the rig data
  do *not* pin down.
* **Uncertainty**: Laplace approximation (exact autograd Hessian) on the
  unconstrained active parameters, mapped to physical units.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..config import CAMBER_PARAMS, RunConfig
from ..data.ttc import TTCData
from ..tyre import PARAM_NAMES, MF62Lateral, active_camber_check, mf62_fy
from .common import DTYPE, ar1_inflation, hessian, laplace_cov, nll_residual, r2, rmse, run_adam, run_lbfgs, sobol

Progress = Optional[Callable[[str, float], None]]


@dataclass
class TTCTensors:
    alpha: torch.Tensor
    Fz: torch.Tensor
    Fy: torch.Tensor
    gamma: torch.Tensor
    Fz_src: torch.Tensor     # mean load of the sample's source
    w: torch.Tensor          # per-sample weight (each source sums to N/S)
    src: np.ndarray

    @classmethod
    def from_data(cls, ttc: TTCData) -> "TTCTensors":
        df = ttc.df
        t = lambda c: torch.tensor(df[c].to_numpy(), dtype=DTYPE)  # noqa: E731
        src = df["source_id"].to_numpy()
        counts = np.bincount(src)
        S = int((counts > 0).sum())
        w = len(df) / (S * counts[src])
        Fz_src = df.groupby("source_id")["Fz"].transform("mean").to_numpy()
        return cls(t("alpha"), t("Fz"), t("Fy"), t("gamma"),
                   torch.tensor(Fz_src, dtype=DTYPE), torch.tensor(w, dtype=DTYPE), src)


@dataclass
class TTCFitResult:
    model: MF62Lateral
    values: Dict[str, float]
    active: List[str]
    frozen: List[str]
    Fz0: float
    sigma_mu: float
    objective: float
    inflation: float
    cov_u: np.ndarray                    # active, unconstrained
    cov_theta: np.ndarray                # active, physical
    std: Dict[str, float]
    corr: pd.DataFrame
    hess_eig: np.ndarray
    starts: pd.DataFrame                 # every multi-start end point
    ensemble: List[Dict[str, float]]     # near-optimal coefficient sets
    metrics: Dict[str, float]
    per_source: pd.DataFrame
    history: List[float] = field(default_factory=list)
    seconds: float = 0.0


def build_tyre(ttc: TTCData, cfg: RunConfig) -> MF62Lateral:
    Fz0 = cfg.tyre.Fz0 or ttc.Fz0_equal_weight
    frozen: List[str] = []
    gamma = torch.tensor(ttc.df["gamma"].to_numpy(), dtype=DTYPE)
    if cfg.tyre.auto_freeze_camber and not active_camber_check(gamma, cfg.ttc.camber_identifiable_deg):
        frozen += list(CAMBER_PARAMS)
    return MF62Lateral(Fz0=Fz0, cfg=cfg.tyre, frozen=frozen)


class _Objective:
    """Batched tempered negative log-posterior of the rig data."""

    def __init__(self, model: MF62Lateral, T: TTCTensors, kind: str, nu: float):
        self.m, self.T, self.kind, self.nu = model, T, kind, nu
        sp = model.specs
        self.mu = torch.tensor([sp[n].prior_mu for n in PARAM_NAMES], dtype=DTYPE)
        self.sd = torch.tensor([sp[n].prior_sd for n in PARAM_NAMES], dtype=DTYPE)
        self.N = float(T.alpha.numel())
        self.c = 1.0   # tempering = AR(1) variance inflation

    def residual(self, u: torch.Tensor) -> torch.Tensor:
        th = self.m.theta(u)
        Fy = mf62_fy(th.unsqueeze(-2), self.T.alpha, self.T.Fz, self.m.Fz0, self.T.gamma,
                     sign_eps=self.m.sign_eps, ey_beta=self.m.ey_beta)
        return (Fy - self.T.Fy) / self.T.Fz_src

    def __call__(self, u: torch.Tensor, log_sig: torch.Tensor) -> torch.Tensor:
        r = self.residual(u)
        z = r / torch.exp(log_sig).unsqueeze(-1)
        data = (self.T.w * (nll_residual(z, self.kind, self.nu) + log_sig.unsqueeze(-1))).sum(-1)
        th = self.m.theta(u)
        prior = 0.5 * (self.m.active_mask * ((th - self.mu) / self.sd) ** 2).sum(-1)
        return data / self.c + prior


def fit_ttc(ttc: TTCData, cfg: RunConfig, progress: Progress = None) -> TTCFitResult:
    t0 = time.time()
    fc = cfg.ttc_fit
    torch.manual_seed(fc.seed)
    model = build_tyre(ttc, cfg)
    T = TTCTensors.from_data(ttc)
    obj = _Objective(model, T, fc.likelihood, fc.student_nu)
    act_idx = [i for i, n in enumerate(PARAM_NAMES) if n not in model.frozen]
    history: List[float] = []

    # ---------------- batched multi-start -------------------------------- #
    S = max(int(fc.n_starts), 1)
    u0 = model.u.detach().clone()
    U = u0.repeat(S, 1)
    if S > 1:
        q = sobol(S - 1, len(act_idx), fc.seed)
        for j, i in enumerate(act_idx):
            s = model.specs[PARAM_NAMES[i]]
            lo = max(s.lo, s.prior_mu - 2 * s.prior_sd)
            hi = min(s.hi, s.prior_mu + 2 * s.prior_sd)
            vals = lo + (hi - lo) * q[:, j]
            frac = np.clip((vals - s.lo) / (s.hi - s.lo), 1e-4, 1 - 1e-4)
            U[1:, i] = torch.tensor(np.log(frac / (1 - frac)), dtype=DTYPE)
    U.requires_grad_(True)
    LS = torch.full((S,), math.log(0.05), dtype=DTYPE, requires_grad=True)
    if progress:
        progress(f"TTC multi-start: {S} starts", 0.05)
    run_adam([U, LS], lambda: obj(U, LS).sum() / obj.N, iters=fc.adam_iters, lr=0.08,
             log=history)
    with torch.no_grad():
        f_adam = obj(U, LS).numpy()

    # ---------------- L-BFGS polish of the best candidates ---------------- #
    order = np.argsort(f_adam)
    n_polish = min(S, 6)
    finals = np.full(S, np.nan)
    U_fin = U.detach().clone()
    LS_fin = LS.detach().clone()
    for rank, s_i in enumerate(order):
        u = U.detach()[s_i].clone().requires_grad_(True)
        ls = LS.detach()[s_i].clone().requires_grad_(True)
        iters = fc.lbfgs_iters if rank < n_polish else max(fc.lbfgs_iters // 4, 50)
        f = run_lbfgs([u, ls], lambda: obj(u, ls) / obj.N, max_iter=iters)
        finals[s_i] = f * obj.N
        U_fin[s_i], LS_fin[s_i] = u.detach(), ls.detach()
        if progress:
            progress(f"TTC polish {rank + 1}/{S}", 0.05 + 0.45 * (rank + 1) / S)
    best = int(np.nanargmin(finals))

    # ---------------- serial-correlation tempering + final refine --------- #
    with torch.no_grad():
        res = obj.residual(U_fin[best]).numpy()
    obj.c = ar1_inflation(res, T.src)
    u = U_fin[best].clone().requires_grad_(True)
    ls = LS_fin[best].clone().requires_grad_(True)
    f_best = run_lbfgs([u, ls], lambda: obj(u, ls), max_iter=fc.lbfgs_iters, log=history)
    with torch.no_grad():
        model.u.copy_(u.detach())

    # ---------------- Laplace on (u_active, log sigma) -------------------- #
    def f_act(x: torch.Tensor) -> torch.Tensor:
        uu = u.detach().clone()
        uu = uu.index_put((torch.tensor(act_idx),), x[:-1])
        return obj(uu, x[-1])

    x_hat = torch.cat([u.detach()[act_idx], ls.detach().reshape(1)])
    H = hessian(f_act, x_hat)
    cov_x, hinfo = laplace_cov(H)
    cov_u = cov_x[:-1, :-1]
    s = torch.sigmoid(u.detach()[act_idx])
    jac = (model.hi - model.lo)[act_idx] * s * (1 - s)
    cov_theta = (jac[:, None] * cov_u * jac[None, :]).numpy()
    std_vec = np.sqrt(np.clip(np.diag(cov_theta), 0, None))
    names = [PARAM_NAMES[i] for i in act_idx]
    d = np.sqrt(np.clip(np.diag(cov_theta), 1e-300, None))
    corr = pd.DataFrame(cov_theta / np.outer(d, d), index=names, columns=names)

    # ---------------- ensemble of near-optimal solutions ------------------ #
    # compare all starts on the same (tempered) objective
    with torch.no_grad():
        f_all = obj(U_fin, LS_fin).numpy()
    rows = []
    for k in range(S):
        vals = model.theta(U_fin[k]).detach().numpy()
        rows.append({"start": k, "objective": float(f_all[k]),
                     **{n: float(vals[i]) for i, n in enumerate(PARAM_NAMES)}})
    starts = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    thresh = float(starts["objective"].min()) + 3.0   # ~ 95% for the leading directions
    ensemble = starts[starts["objective"] <= thresh][PARAM_NAMES].to_dict("records")

    # ---------------- metrics --------------------------------------------- #
    with torch.no_grad():
        Fy_hat = model(T.alpha, T.Fz, T.gamma).numpy()
    df = ttc.df.assign(Fy_hat=Fy_hat)
    per = []
    for sid, g in df.groupby("source_id"):
        per.append({"source": g["source"].iloc[0], "Fz_mean_N": g["Fz"].mean(), "rows": len(g),
                    "rmse_N": rmse(g["Fy"], g["Fy_hat"]), "r2": r2(g["Fy"], g["Fy_hat"]),
                    "mu_rmse": rmse(g["Fy"] / g["Fz"], g["Fy_hat"] / g["Fz"])})
    metrics = {"rmse_N": rmse(df["Fy"], df["Fy_hat"]), "r2": r2(df["Fy"], df["Fy_hat"]),
               "mae_N": float(np.mean(np.abs(df["Fy"] - df["Fy_hat"]))),
               "sigma_mu": float(torch.exp(ls.detach())), "n_rows": int(len(df)),
               "ar1_inflation": float(obj.c), "n_eff": float(len(df) / obj.c),
               "hessian_condition": float(hinfo["condition"]),
               "n_starts": S, "n_near_optimal": len(ensemble)}
    if progress:
        progress("TTC fit done", 0.5)
    return TTCFitResult(
        model=model, values=model.values(), active=names, frozen=list(model.frozen),
        Fz0=float(model.Fz0), sigma_mu=float(torch.exp(ls.detach())), objective=float(f_best),
        inflation=float(obj.c), cov_u=cov_u.numpy(), cov_theta=cov_theta,
        std={n: float(v) for n, v in zip(names, std_vec)}, corr=corr, hess_eig=hinfo["eigvals"],
        starts=starts, ensemble=ensemble, metrics=metrics,
        per_source=pd.DataFrame(per), history=history, seconds=time.time() - t0)
