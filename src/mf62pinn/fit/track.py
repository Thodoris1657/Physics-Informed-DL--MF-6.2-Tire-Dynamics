"""
Stage B - track calibration as a hierarchical errors-in-variables (EIV) problem.

Model
-----
For every quasi-steady segment ``k`` (window means of one run ``j(k)``)::

    beta_k   = beta_meas_k + b_global + b_run[j(k)] + e_k          (latent sideslip)
    Vy_k     = Vx_k tan(beta_k)
    alpha_ik = rigid-body slip of wheel i                             (vehicle.slip_angles)
    Fz_ik    = LoadTransfer_v2(Vx, ax, ay)                            (vehicle.load_transfer)
    Fy_ik    = MF6.2(alpha_ik, Fz_ik; theta, LMUY_axle, LKY) * ellipse(Fx_ik)
    m ay_k        = sum_i Fy_ik cos(delta_i)                          (lateral balance)
    Iz r_dot_k    = sum_i Fy_ik (x_i cos(delta_i) + y_i sin(delta_i)) (yaw balance)

with priors

    b_global ~ N(0, s_g^2),  b_run ~ N(0, s_r^2),  e_k ~ N(0, s_s^2 + sem_k^2)
    log LMUY ~ N(log m_mu, sd_mu^2),  log LKY ~ N(0, sd_K^2)
    theta    ~ N(theta_TTC, Sigma_TTC)       (Laplace posterior of stage A)

Why this removes the v1 degeneracy
----------------------------------
In v1 the measured slip angle entered the balances at face value, apart from
one global offset. With 1-4 deg of slip noise the least-squares optimum moves
every wheel onto the flat post-peak plateau, where the predicted force no
longer depends on the noisy slip angle (errors-in-variables attenuation). Here
the slip error of each segment is an explicit latent variable with a
measurement prior, so noise is absorbed where it occurs and the plateau no
longer buys anything. What identifies the axle grip and balance is the
*difference* of front and rear slip, ``alpha_f - alpha_r = delta - L r/Vx``,
which does not depend on the lateral-velocity sensor at all.

Grip scaling follows the MF6.2 convention: ``LMUY`` scales friction (D_y and,
through LMUY', S_Vy) while the cornering stiffness is scaled separately by
``LKY`` with a tight prior - the abrasive TTC belt inflates friction far more
than carcass stiffness.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RunConfig, VehicleParams
from ..tyre import PARAM_NAMES, MF62Lateral, mf62_fy
from ..vehicle import WHEELS, body_forces, load_transfer, rear_drive_force, slip_angles
from .common import DTYPE, hessian, laplace_cov, nll_residual, r2, rmse, run_adam, run_lbfgs
from .ttc import TTCFitResult

Progress = Optional[Callable[[str, float], None]]
D2R = math.pi / 180.0


@dataclass
class SegTensors:
    Vx: torch.Tensor
    r: torch.Tensor
    r_dot: torch.Tensor
    delta: torch.Tensor
    ax: torch.Tensor
    ay: torch.Tensor
    beta_meas: torch.Tensor
    beta_sem: torch.Tensor
    run_id: torch.Tensor
    n_runs: int

    @classmethod
    def from_df(cls, seg: pd.DataFrame, n_runs: Optional[int] = None) -> "SegTensors":
        t = lambda c: torch.tensor(seg[c].to_numpy(), dtype=DTYPE)  # noqa: E731
        rid = torch.tensor(seg["run_id"].to_numpy(), dtype=torch.long)
        return cls(t("Vx"), t("r"), t("r_dot"), t("delta"), t("ax"), t("ay"), t("beta_meas"),
                   t("beta_sem"), rid, int(n_runs if n_runs is not None else rid.max().item() + 1))

    def subset(self, mask: np.ndarray) -> "SegTensors":
        m = torch.tensor(mask)
        return SegTensors(self.Vx[m], self.r[m], self.r_dot[m], self.delta[m], self.ax[m], self.ay[m],
                          self.beta_meas[m], self.beta_sem[m], self.run_id[m], self.n_runs)


class TrackModel(nn.Module):
    """Vehicle-level model with MF6.2 scaling factors and latent sideslip."""

    def __init__(self, tyre: MF62Lateral, ttc: TTCFitResult, veh: VehicleParams, cfg: RunConfig,
                 n_runs: int, n_seg: int):
        super().__init__()
        tc = cfg.track_fit
        self.tc, self.veh = tc, veh
        self.tyre = copy.deepcopy(tyre)
        self.act_idx = [i for i, n in enumerate(PARAM_NAMES) if n not in tyre.frozen]
        u_hat = self.tyre.u.detach().clone()
        self.register_buffer("u_hat", u_hat)
        self.register_buffer("prec_u", torch.linalg.pinv(torch.tensor(ttc.cov_u, dtype=DTYPE)))
        self.tyre.u.requires_grad_(tc.shape_coupling != "fixed")
        n_ax = 2 if tc.per_axle else 1
        mu0 = math.log(tc.lambda_mu_median)
        self.log_lmu = nn.Parameter(torch.full((n_ax,), mu0, dtype=DTYPE))
        self.log_lky = nn.Parameter(torch.zeros(1, dtype=DTYPE), requires_grad=tc.learn_stiffness_scale)
        self.z_g = nn.Parameter(torch.zeros(1, dtype=DTYPE))
        eiv = tc.formulation == "eiv"
        self.z_run = nn.Parameter(torch.zeros(n_runs, dtype=DTYPE), requires_grad=eiv)
        self.z_seg = nn.Parameter(torch.zeros(n_seg, dtype=DTYPE), requires_grad=eiv)
        self.eiv = eiv
        # One shared noise scale: the *ratio* of lateral to yaw noise is fixed by the
        # config. Learning the two scales independently is degenerate in an EIV
        # model - one latent per segment can zero the yaw residual exactly, its
        # scale then collapses and the lateral balance gets ignored.
        self.log_noise = nn.Parameter(torch.zeros((), dtype=DTYPE), requires_grad=tc.learn_noise)
        # latent steering zero offsets (global + per run); disabled when sigma = 0
        self.sd_g = tc.sigma_steer_global_deg * D2R
        self.sd_r = tc.sigma_steer_run_deg * D2R
        self.zd_g = nn.Parameter(torch.zeros(1, dtype=DTYPE), requires_grad=self.sd_g > 0)
        self.zd_run = nn.Parameter(torch.zeros(n_runs, dtype=DTYPE), requires_grad=self.sd_r > 0)
        self.s_g = tc.sigma_beta_global_deg * D2R
        self.s_r = tc.sigma_beta_run_deg * D2R
        self.s_s = tc.sigma_beta_seg_deg * D2R

    # ------------------------------------------------------------- scalings
    @property
    def lmu(self) -> torch.Tensor:          # (front, rear)
        v = torch.exp(self.log_lmu)
        return v if v.numel() == 2 else v.repeat(2)

    @property
    def lky(self) -> torch.Tensor:
        return torch.exp(self.log_lky)[0]

    def beta_offset(self, T: SegTensors, include_seg: bool = True) -> torch.Tensor:
        off = self.s_g * self.z_g[0] + torch.zeros_like(T.Vx)
        if self.eiv:
            off = off + self.s_r * self.z_run[T.run_id]
            if include_seg:
                sd = torch.sqrt(self.s_s ** 2 + T.beta_sem ** 2)
                off = off + sd * self.z_seg
        return off

    def steer_offset(self, T: SegTensors) -> torch.Tensor:
        return self.sd_g * self.zd_g[0] + self.sd_r * self.zd_run[T.run_id]

    # ------------------------------------------------------------- physics
    def wheel_forces(self, Vx, Vy, r, delta, ax, ay):
        veh = self.veh
        alpha = slip_angles(Vx, Vy, r, delta, veh)
        Fz = load_transfer(Vx, ax, ay, veh, smooth_floor_N=20.0)
        Fx = rear_drive_force(Vx, ax, veh)
        lmu, lky = self.lmu, self.lky
        th = self.tyre.theta()
        out_Fy = []
        for i in range(4):
            ax_i = 0 if i < 2 else 1
            if self.tc.grip_scaling == "friction":
                Fy0 = mf62_fy(th, alpha[i], Fz[i], self.tyre.Fz0, lmuy=lmu[ax_i], lky=lky,
                              sign_eps=self.tyre.sign_eps, ey_beta=self.tyre.ey_beta)
                mu_pk = lmu[ax_i] * (th[1] + th[2] * (Fz[i] - self.tyre.Fz0) / self.tyre.Fz0)
            else:  # v1: whole-force scale
                Fy0 = lmu[ax_i] * mf62_fy(th, alpha[i], Fz[i], self.tyre.Fz0,
                                          sign_eps=self.tyre.sign_eps, ey_beta=self.tyre.ey_beta)
                mu_pk = lmu[ax_i] * (th[1] + th[2] * (Fz[i] - self.tyre.Fz0) / self.tyre.Fz0)
            if self.tc.combined_slip == "ellipse":
                u = Fx[i] / (mu_pk * Fz[i])
                fac = torch.sqrt(0.02 + F.softplus(0.98 - u * u, beta=40.0))
                Fy0 = Fy0 * fac
            out_Fy.append(Fy0)
        return {"alpha": alpha, "Fz": Fz, "Fx": Fx, "Fy": tuple(out_Fy)}

    def forward(self, T: SegTensors, include_seg: bool = True,
                beta: Optional[torch.Tensor] = None) -> Dict[str, object]:
        if beta is None:
            beta = T.beta_meas + self.beta_offset(T, include_seg)
        Vy = T.Vx * torch.tan(beta)
        delta = T.delta + self.steer_offset(T)
        w = self.wheel_forces(T.Vx, Vy, T.r, delta, T.ax, T.ay)
        Fy_body, Mz = body_forces(w["Fy"], delta, self.veh)
        veh = self.veh
        R_lat = veh.m_tot * T.ay - Fy_body
        R_yaw = veh.Iz * T.r_dot - Mz
        return {**w, "beta": beta, "delta": delta, "Fy_body": Fy_body, "Mz": Mz, "R_lat": R_lat, "R_yaw": R_yaw}

    # ------------------------------------------------------------- posterior
    def sigmas(self):
        k = torch.exp(self.log_noise)
        return self.tc.sigma_lat_g * k, self.tc.sigma_yaw_Nm * k

    def neg_log_post(self, T: SegTensors) -> torch.Tensor:
        tc, veh = self.tc, self.veh
        out = self.forward(T)
        s_lat, s_yaw = self.sigmas()
        z_lat = out["R_lat"] / (veh.m_tot * veh.g * s_lat)
        z_yaw = out["R_yaw"] / s_yaw
        K = T.Vx.numel()
        nll = (nll_residual(z_lat, tc.likelihood, tc.student_nu).sum() + K * torch.log(s_lat)
               + nll_residual(z_yaw, tc.likelihood, tc.student_nu).sum() + K * torch.log(s_yaw))
        # priors
        pri = 0.5 * ((self.log_lmu - math.log(tc.lambda_mu_median)) / tc.lambda_mu_sdlog).pow(2).sum()
        pri = pri + 0.5 * (self.log_lky / tc.lambda_K_sdlog).pow(2).sum()
        if tc.formulation == "eiv":
            pri = pri + 0.5 * self.z_g.pow(2).sum() + 0.5 * self.z_run.pow(2).sum() + 0.5 * self.z_seg.pow(2).sum()
        else:  # global offset: weak prior (v1 style)
            pri = pri + 0.5 * self.z_g.pow(2).sum() / 9.0
        if tc.shape_coupling == "laplace":
            du = (self.tyre.u - self.u_hat)[self.act_idx]
            pri = pri + 0.5 * du @ self.prec_u @ du
        pri = pri + 0.5 * self.zd_g.pow(2).sum() + 0.5 * self.zd_run.pow(2).sum()
        # weak log-normal prior on the shared noise scale
        pri = pri + 0.5 * self.log_noise.pow(2) / 4.0
        return nll + pri

    def trainable(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
#  Prediction helpers
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_kinematic(model: TrackModel, T: SegTensors, use_run_offsets: bool = True) -> Dict[str, np.ndarray]:
    """
    Predict the balances from the *measured* kinematics, corrected only by the
    calibrated sensor offsets (global, and per run when the run was seen in
    training). No per-segment latent is used, so this is a genuine prediction.
    """
    off = model.s_g * model.z_g[0] + torch.zeros_like(T.Vx)
    if model.eiv and use_run_offsets:
        off = off + model.s_r * model.z_run[T.run_id]
    o = model.forward(T, beta=T.beta_meas + off)
    return {"ay_hat": (o["Fy_body"] / model.veh.m_tot).numpy(),
            "rdot_hat": (o["Mz"] / model.veh.Iz).numpy(),
            "alpha": np.stack([a.numpy() for a in o["alpha"]], 1)}


@torch.no_grad()
def saturation_fraction(model: TrackModel, out: Dict[str, object]) -> np.ndarray:
    """Per wheel: fraction of segments operating beyond the peak of the (track) curve."""
    from ..tyre import characteristics
    th = model.tyre.theta()
    fr = []
    for i in range(4):
        a = out["alpha"][i]
        Fz = out["Fz"][i]
        lmu = model.lmu[0 if i < 2 else 1]
        pk_pos = characteristics(th, Fz, model.tyre.Fz0, lmuy=lmu, lky=model.lky, side=1)["alpha_peak_deg"]
        pk_neg = characteristics(th, Fz, model.tyre.Fz0, lmuy=lmu, lky=model.lky, side=-1)["alpha_peak_deg"]
        a_deg = a / D2R
        beyond = torch.where(a_deg >= 0, a_deg > pk_pos, a_deg < pk_neg)
        fr.append(float(beyond.double().mean()))
    return np.array(fr)


# --------------------------------------------------------------------------- #
#  Fit
# --------------------------------------------------------------------------- #


@dataclass
class TrackFitResult:
    model: TrackModel
    segments: pd.DataFrame                 # inputs + latent beta, slip, Fz, Fy, residuals
    lambdas: Dict[str, float]
    lambdas_std: Dict[str, float]
    beta_offsets: pd.DataFrame             # per run: global + run offset [deg] (+/- sd)
    metrics: Dict[str, float]
    tyre_values: Dict[str, float]          # posterior tyre coefficients
    tyre_std: Dict[str, float]
    hess_eig: np.ndarray
    loro: Optional[pd.DataFrame] = None
    starts: List[Dict[str, float]] = field(default_factory=list)
    history: List[float] = field(default_factory=list)
    seconds: float = 0.0


def _init_lambda(tyre: MF62Lateral, seg: pd.DataFrame, veh: VehicleParams) -> float:
    mu_track = float(np.quantile(np.abs(seg["ay"]), 0.9)) / veh.g
    mu_rig = float(tyre.values()["pDy1"])
    return float(np.clip(1.05 * mu_track / mu_rig, 0.2, 1.5))


def _fit_once(model: TrackModel, T: SegTensors, cfg: RunConfig, lam0: float, beta0_deg: float,
              history: Optional[List[float]] = None) -> float:
    tc = cfg.track_fit
    with torch.no_grad():
        model.log_lmu.fill_(math.log(lam0))
        model.log_lky.zero_()
        model.z_g.fill_(beta0_deg * D2R / model.s_g)
        model.z_run.zero_()
        model.z_seg.zero_()
        model.zd_g.zero_()
        model.zd_run.zero_()
        model.log_noise.zero_()
        model.tyre.u.copy_(model.u_hat)
    K = float(T.Vx.numel())
    params = model.trainable()
    run_adam(params, lambda: model.neg_log_post(T) / K, iters=tc.adam_iters, lr=0.02, log=history)
    f = run_lbfgs(params, lambda: model.neg_log_post(T) / K, max_iter=tc.lbfgs_iters, log=history)
    return f * K


def fit_track(seg: pd.DataFrame, ttc_res: TTCFitResult, cfg: RunConfig, n_runs: int,
              progress: Progress = None, loro: bool = False) -> TrackFitResult:
    t0 = time.time()
    tc, veh = cfg.track_fit, cfg.vehicle
    torch.manual_seed(tc.seed)
    tyre = ttc_res.model
    T = SegTensors.from_df(seg, n_runs)
    model = TrackModel(tyre, ttc_res, veh, cfg, n_runs, len(seg))
    lam0 = _init_lambda(tyre, seg, veh)

    # multi-start over the sideslip offset (the plateau basin lives at large offsets)
    starts, history = [], []
    best_f, best_state = math.inf, None
    for k, b0 in enumerate((0.0, -3.0, 3.0, 6.0)):
        for lam in (lam0, 0.8 * lam0 + 0.2):
            hist: List[float] = []
            f = _fit_once(model, T, cfg, lam, b0, hist)
            with torch.no_grad():
                starts.append({"beta0_deg": b0, "lambda0": lam, "neg_log_post": f,
                               "lmu_f": float(model.lmu[0]), "lmu_r": float(model.lmu[1]),
                               "LKY": float(model.lky),
                               "b_global_deg": float(model.s_g * model.z_g.detach()[0] / D2R)})
            if f < best_f:
                best_f, best_state, history = f, {k_: v.detach().clone() for k_, v in model.state_dict().items()}, hist
        if progress:
            progress(f"track calibration start {k + 1}/4", 0.5 + 0.08 * (k + 1))
    model.load_state_dict(best_state)

    # ---------------- Laplace over every free quantity ---------------- #
    f_vec, x_hat, names = flat_neg_log_post(model, T)
    H = hessian(f_vec, x_hat)
    cov, hinfo = laplace_cov(H)
    sd = torch.sqrt(torch.clamp(torch.diag(cov), min=0)).numpy()
    sd_map = dict(zip(names, sd))
    lam_names = ["LMUY_front", "LMUY_rear"]
    lmu = model.lmu.detach().numpy()
    lam = {lam_names[0]: float(lmu[0]), lam_names[1]: float(lmu[1]), "LKY": float(model.lky.detach())}
    if tc.per_axle:
        lam_sd = {lam_names[i]: float(lmu[i] * sd_map[f"log_lmu[{i}]"]) for i in range(2)}
    else:
        lam_sd = {n: float(lmu[0] * sd_map["log_lmu[0]"]) for n in lam_names}
    lam_sd["LKY"] = float(model.lky.detach() * sd_map.get("log_lky[0]", 0.0))

    # tyre posterior
    th = model.tyre.theta().detach()
    tyre_vals = {n: float(th[i]) for i, n in enumerate(PARAM_NAMES)}
    tyre_sd = {}
    if tc.shape_coupling != "fixed":
        for j, i in enumerate(model.act_idx):
            s = torch.sigmoid(model.tyre.u.detach()[i])
            jac = float((model.tyre.hi - model.tyre.lo)[i] * s * (1 - s))
            tyre_sd[PARAM_NAMES[i]] = jac * float(sd_map.get(f"tyre.u[{i}]", 0.0))
    else:
        tyre_sd = dict(ttc_res.std)

    # ---------------- per-segment outputs ----------------------------- #
    with torch.no_grad():
        out = model(T)
        kin = predict_kinematic(model, T)
        sat = saturation_fraction(model, out)
        s_lat, s_yaw = model.sigmas()
    S = seg.copy()
    S["beta_latent_deg"] = out["beta"].numpy() / D2R
    S["beta_meas_deg"] = seg["beta_meas"].to_numpy() / D2R
    for i, w in enumerate(WHEELS):
        S[f"alpha_{w}_deg"] = out["alpha"][i].numpy() / D2R
        S[f"Fz_{w}"] = out["Fz"][i].numpy()
        S[f"Fy_{w}"] = out["Fy"][i].numpy()
    S["ay_fit"] = out["Fy_body"].numpy() / veh.m_tot
    S["ay_kin"] = kin["ay_hat"]
    S["rdot_kin"] = kin["rdot_hat"]
    S["rdot_fit"] = out["Mz"].numpy() / veh.Iz
    S["R_lat_N"] = out["R_lat"].numpy()
    S["R_yaw_Nm"] = out["R_yaw"].numpy()

    # per-run offsets
    rows = []
    run_names = seg.groupby("run_id")["run"].first()
    for j in range(n_runs):
        g = float(model.s_g * model.z_g.detach()[0])
        rr = float(model.s_r * model.z_run.detach()[j]) if model.eiv else 0.0
        var = (model.s_g * sd_map.get("z_g[0]", 0.0)) ** 2
        if model.eiv:
            var += (model.s_r * sd_map.get(f"z_run[{j}]", 0.0)) ** 2
        vx = float(seg.loc[seg["run_id"] == j, "Vx"].mean()) if (seg["run_id"] == j).any() else float("nan")
        rows.append({"run": run_names.get(j, f"run{j}"), "offset_deg": (g + rr) / D2R,
                     "offset_sd_deg": math.sqrt(var) / D2R,
                     "global_deg": g / D2R, "run_deg": rr / D2R,
                     "equiv_vy_offset_ms": vx * math.tan(g + rr)})
    offsets = pd.DataFrame(rows)

    ay = seg["ay"].to_numpy()
    run_mean = S.groupby("run")[["ay", "ay_kin"]].mean()
    metrics = {
        "segments": int(len(seg)), "runs": int(n_runs),
        "ay_r2_fit": r2(ay, S["ay_fit"]), "ay_rmse_fit": rmse(ay, S["ay_fit"]),
        "ay_r2_kin": r2(ay, S["ay_kin"]), "ay_rmse_kin": rmse(ay, S["ay_kin"]),
        "ay_bias_kin": float(np.mean(S["ay_kin"] - ay)),
        "ay_runmean_rmse_kin": rmse(run_mean["ay"], run_mean["ay_kin"]),
        "rdot_rmse_kin": rmse(seg["r_dot"], S["rdot_kin"]),
        "yaw_rms_Nm": float(np.sqrt(np.mean(S["R_yaw_Nm"] ** 2))),
        "lat_rms_N": float(np.sqrt(np.mean(S["R_lat_N"] ** 2))),
        "sigma_lat_g": float(s_lat), "sigma_yaw_Nm": float(s_yaw),
        "beta_global_offset_deg": float(model.s_g * model.z_g.detach()[0] / D2R),
        "beta_latent_minus_meas_rms_deg": float(np.sqrt(np.mean((S["beta_latent_deg"] - S["beta_meas_deg"]) ** 2))),
        "saturated_fraction": float(sat.mean()),
        "neg_log_post": float(best_f), "hessian_condition": float(hinfo["condition"]),
        "hessian_nonpositive": int(hinfo["n_nonpositive"]),
    }
    for i, w in enumerate(WHEELS):
        metrics[f"alpha_{w}_mean_deg"] = float(S[f"alpha_{w}_deg"].mean())
        metrics[f"saturated_{w}"] = float(sat[i])

    res = TrackFitResult(model=model, segments=S, lambdas=lam, lambdas_std=lam_sd, beta_offsets=offsets,
                         metrics=metrics, tyre_values=tyre_vals, tyre_std=tyre_sd, hess_eig=hinfo["eigvals"],
                         starts=starts, history=history, seconds=time.time() - t0)
    if loro and n_runs > 2:
        res.loro = leave_one_run_out(seg, ttc_res, cfg, n_runs, best_state, progress)
        lo = res.loro
        res.metrics["ay_r2_loro"] = r2(lo["ay"], lo["ay_pred"])
        res.metrics["ay_rmse_loro"] = rmse(lo["ay"], lo["ay_pred"])
        rm = lo.groupby("run")[["ay", "ay_pred"]].mean()
        res.metrics["ay_runmean_rmse_loro"] = rmse(rm["ay"], rm["ay_pred"])
    res.seconds = time.time() - t0
    return res


class _NLP(nn.Module):
    def __init__(self, m: "TrackModel"):
        super().__init__()
        self.m = m

    def forward(self, T_: SegTensors) -> torch.Tensor:
        return self.m.neg_log_post(T_)


def flat_neg_log_post(model: "TrackModel", T: SegTensors):
    """Return ``(f, x0, names)`` with ``f(x)`` the negative log-posterior as a
    function of the flattened free parameters (for exact Hessians)."""
    from torch.func import functional_call
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    sizes = [p.numel() for _, p in named]
    wrapper = _NLP(model)

    def f(x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, sizes)
        pdict = {f"m.{n}": c.reshape(p.shape) for (n, p), c in zip(named, chunks)}
        return functional_call(wrapper, pdict, (T,))

    x0 = torch.cat([p.detach().reshape(-1) for _, p in named])
    names = [f"{n}[{i}]" for n, p in named for i in range(p.numel())]
    return f, x0, names


def leave_one_run_out(seg: pd.DataFrame, ttc_res: TTCFitResult, cfg: RunConfig, n_runs: int,
                      init_state: Dict[str, torch.Tensor], progress: Progress = None) -> pd.DataFrame:
    """Refit without each run in turn and predict it Vy-free (no Vy channel used)."""
    rows = []
    T_all = SegTensors.from_df(seg, n_runs)
    for j in range(n_runs):
        train = (seg["run_id"] != j).to_numpy()
        if train.sum() < 10 or (~train).sum() == 0:
            continue
        seg_tr = seg[train].reset_index(drop=True)
        T_tr = T_all.subset(train)
        m = TrackModel(ttc_res.model, ttc_res, cfg.vehicle, cfg, n_runs, int(train.sum()))
        # warm start from the full fit (global quantities only)
        with torch.no_grad():
            m.log_lmu.copy_(init_state["log_lmu"])
            m.log_lky.copy_(init_state["log_lky"])
            m.z_g.copy_(init_state["z_g"])
            m.tyre.u.copy_(init_state["tyre.u"])
            m.log_noise.copy_(init_state["log_noise"])
            m.zd_g.copy_(init_state["zd_g"])
        K = float(len(seg_tr))
        run_lbfgs(m.trainable(), lambda: m.neg_log_post(T_tr) / K, max_iter=cfg.track_fit.lbfgs_iters)
        T_te = T_all.subset(~train)
        kin = predict_kinematic(m, T_te, use_run_offsets=False)
        te = seg[~train]
        with torch.no_grad():
            lf_, lr_ = float(m.lmu[0]), float(m.lmu[1])
        for k in range(len(te)):
            rows.append({"run": te["run"].iloc[k], "ay": float(te["ay"].iloc[k]), "ay_pred": float(kin["ay_hat"][k]),
                         "r_dot": float(te["r_dot"].iloc[k]), "rdot_pred": float(kin["rdot_hat"][k]),
                         "lmu_f": lf_, "lmu_r": lr_})
        if progress:
            progress(f"leave-one-run-out {j + 1}/{n_runs}", 0.85 + 0.1 * (j + 1) / n_runs)
    return pd.DataFrame(rows)
