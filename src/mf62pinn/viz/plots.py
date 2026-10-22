"""Publication figures for a pipeline result (PNG, light theme)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..analysis.characteristics import curve_band
from ..analysis.handling import measured_handling_points
from ..fit.common import DTYPE
from ..tyre import PARAM_NAMES, mf62_fy
from ..vehicle import WHEELS
from . import style as S

D2R = math.pi / 180.0


def _save(fig, path: Path, dpi: int) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _fy(model, alpha_deg, Fz, lmuy=1.0, lky=1.0, theta=None):
    a = torch.tensor(np.radians(alpha_deg), dtype=DTYPE)
    with torch.no_grad():
        th = model.theta() if theta is None else torch.tensor(theta, dtype=DTYPE)
        return mf62_fy(th, a, torch.full_like(a, float(Fz)), model.Fz0, lmuy=lmuy, lky=lky,
                       sign_eps=model.sign_eps, ey_beta=model.ey_beta).numpy()


# --------------------------------------------------------------------------- #
def fig_ttc_fit(res, path, dpi):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    tf, df = res.ttc_fit, res.ttc.df
    srcs = df.groupby("source")["Fz"].mean().sort_values()
    n = len(srcs)
    ncol = min(n, 3)
    nrow = int(math.ceil(n / ncol))
    fig = plt.figure(figsize=(4.7 * ncol, 4.1 * nrow))
    outer = GridSpec(nrow, ncol, figure=fig, hspace=0.38, wspace=0.28)
    cols = S.ordinal(n)
    for k, (src, Fz_m) in enumerate(srcs.items()):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // ncol, k % ncol], height_ratios=[3, 1],
                                        hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        g = df[df["source"] == src]
        a = np.degrees(g["alpha"].to_numpy())
        ax.scatter(a, g["Fy"], s=3, color=S.MUTED, alpha=0.35, linewidths=0, rasterized=True, label="TTC data")
        grid = np.linspace(a.min(), a.max(), 241)
        m, sd = curve_band(tf.model, grid, Fz_m, tf.cov_u)
        ax.fill_between(grid, m - 2 * sd, m + 2 * sd, color=cols[k], alpha=0.25, linewidth=0, label="±2σ band")
        ax.plot(grid, m, color=cols[k], lw=2, label="MF6.2")
        row = tf.per_source[tf.per_source["source"] == src].iloc[0]
        ax.set_title(f"Fz ≈ {Fz_m:.0f} N   R² {row['r2']:.4f} · RMSE {row['rmse_N']:.0f} N", fontsize=9.5)
        ax.set_ylabel("Fy [N]")
        ax.tick_params(labelbottom=False)
        with torch.no_grad():
            fit = tf.model(torch.tensor(g["alpha"].to_numpy()), torch.tensor(g["Fz"].to_numpy())).numpy()
        axr.scatter(a, g["Fy"] - fit, s=2, color=cols[k], alpha=0.4, linewidths=0, rasterized=True)
        axr.axhline(0, color=S.AXIS, lw=1)
        axr.set_xlabel("slip angle α [deg]")
        axr.set_ylabel("resid. [N]")
        if k == 0:
            ax.legend(loc="upper left", markerscale=3)
    fig.suptitle("Stage A · MF6.2 fit to the TTC cornering sweeps (±2σ Laplace band, residuals below)",
                 x=0.06, y=0.99, ha="left", fontweight="semibold", color=S.INK)
    return _save(fig, path, dpi)


def fig_tyre_curves(res, path, dpi):
    tf = res.ttc_fit
    loads = [200, 400, 600, 800, 1000, 1200]
    cols = S.ordinal(len(loads))
    a = np.linspace(-15, 15, 401)
    has_track = res.track is not None
    fig, axes = plt.subplots(1, 2 if has_track else 1, figsize=(12 if has_track else 6.5, 4.8), squeeze=False)
    ax = axes[0, 0]
    for Fz, c in zip(loads, cols):
        y = _fy(tf.model, a, Fz)
        ax.plot(a, y, color=c, lw=2)
        i = np.argmax(y)
        ax.plot(a[i], y[i], "o", ms=5, color=c, mec=S.SURFACE, mew=1.5)
        S.label_end(ax, a[-1], y[-1], f"{Fz} N")
    ax.set_title("Rig tyre (TTC belt)")
    ax.set_xlabel("slip angle α [deg]")
    ax.set_ylabel("Fy [N]")
    ax.axhline(0, color=S.AXIS, lw=0.8)
    ax.set_xlim(-15, 18.5)
    if has_track:
        ax2 = axes[0, 1]
        lam = res.track.lambdas
        for Fz, c in zip(loads, cols):
            y = _fy(tf.model, a, Fz, lmuy=lam["LMUY_front"], lky=lam["LKY"])
            yr = _fy(tf.model, a, Fz, lmuy=lam["LMUY_rear"], lky=lam["LKY"])
            ax2.plot(a, y, color=c, lw=2)
            ax2.plot(a, yr, color=c, lw=1.2, ls="--")
            i = np.argmax(y)
            ax2.plot(a[i], y[i], "o", ms=5, color=c, mec=S.SURFACE, mew=1.5)
            S.label_end(ax2, a[-1], y[-1], f"{Fz} N")
        ax2.plot([], [], color=S.INK2, lw=2, label=f"front  LMUY={lam['LMUY_front']:.3f}")
        ax2.plot([], [], color=S.INK2, lw=1.2, ls="--", label=f"rear   LMUY={lam['LMUY_rear']:.3f}")
        ax2.legend(loc="upper left")
        ax2.set_title("Track-calibrated tyre (same shape, friction scaled)")
        ax2.set_xlabel("slip angle α [deg]")
        ax2.axhline(0, color=S.AXIS, lw=0.8)
        ax2.set_xlim(-15, 18.5)
        ax2.set_ylim(ax.get_ylim())
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_characteristics(res, path, dpi):
    cr = res.char_rig[res.char_rig["side"] == "+"]
    ct = res.char_track[res.char_track["side"] == "+"] if res.char_track is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    specs = [("mu_peak", "peak friction μ_y [-]"), ("alpha_peak_deg", "peak slip angle [deg]"),
             ("Kya_N_per_deg", "cornering stiffness K_yα [N/deg]")]
    df = res.ttc.df
    for ax, (k, lab) in zip(axes, specs):
        ax.errorbar(cr["Fz_N"], cr[k], yerr=2 * cr[k + "_sd"], color=S.RIG, marker="o", ms=5, capsize=3,
                    lw=2, label="rig (TTC)")
        if ct is not None:
            ax.errorbar(ct["Fz_N"], ct[k], yerr=2 * ct[k + "_sd"], color=S.TRACK, marker="s", ms=5, capsize=3,
                        lw=2, label="track")
        if k == "mu_peak":
            for src, g in df.groupby("source"):
                mu = np.quantile(np.abs(g["Fy"]), 0.995) / g["Fz"].mean()
                ax.plot(g["Fz"].mean(), mu, "D", color=S.MUTED, ms=5, zorder=5)
            ax.plot([], [], "D", color=S.MUTED, label="TTC raw peak")
        ax.set_xlabel("Fz [N]")
        ax.set_title(lab, fontsize=10)
    axes[0].legend()
    fig.suptitle("Characteristic values vs. load (error bars ±2σ)", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_identifiability(res, path, dpi):
    tf = res.ttc_fit
    C = tf.corr.to_numpy()
    names = list(tf.corr.index)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(C, cmap=S.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and abs(C[i, j]) >= 0.7:
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(C[i, j]) > 0.85 else S.INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="posterior correlation")
    ax.set_title("Rig posterior correlation (|ρ| ≥ 0.7 annotated)")
    ax = axes[1]
    ev = np.sort(np.abs(tf.hess_eig))
    ax.semilogy(range(1, len(ev) + 1), ev, "o-", color=S.RIG, label="rig (stage A)")
    if res.track is not None:
        ev2 = np.sort(np.abs(res.track.hess_eig))
        ax.semilogy(np.linspace(1, len(ev), len(ev2)), ev2, ".", color=S.TRACK, ms=4, alpha=0.7,
                    label=f"track (stage B, {len(ev2)} dims, rescaled x)")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("|Hessian eigenvalue|")
    ax.set_title(f"Hessian spectrum · condition {tf.metrics['hessian_condition']:.2g}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_multistart(res, path, dpi):
    tf = res.ttc_fit
    a = np.linspace(-15, 15, 301)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    th_all = tf.starts[PARAM_NAMES].to_numpy()
    obj = tf.starts["objective"].to_numpy()
    best = obj.min()
    for ax, Fz in zip(axes, (300.0, 1100.0)):
        for th, f in zip(th_all, obj):
            near = f <= best + 3.0
            ax.plot(a, _fy(tf.model, a, Fz, theta=th), color=S.RIG if near else S.MUTED,
                    lw=0.8 if near else 0.6, alpha=0.35 if near else 0.25)
        ax.plot(a, _fy(tf.model, a, Fz), color=S.INK, lw=1.6, label="selected")
        g = res.ttc.df[np.abs(res.ttc.df["Fz"] - Fz) < 120]
        if len(g):
            ax.scatter(np.degrees(g["alpha"]), g["Fy"], s=2, color=S.ORANGE, alpha=0.25, rasterized=True)
        ax.set_title(f"Fz = {Fz:.0f} N")
        ax.set_xlabel("slip angle α [deg]")
        ax.set_ylabel("Fy [N]")
    axes[0].plot([], [], color=S.RIG, label="near-optimal starts")
    axes[0].plot([], [], color=S.MUTED, label="other starts")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Multi-start landscape: {tf.metrics['n_near_optimal']}/{tf.metrics['n_starts']} Sobol starts "
                 "reach the optimum", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_balance(res, path, dpi):
    t = res.track
    seg = t.segments
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    lim = [seg["ay"].min() - 0.5, seg["ay"].max() + 0.5]
    runs = list(seg["run"].unique())
    cols = {r: S.CATEGORICAL[i % len(S.CATEGORICAL)] for i, r in enumerate(runs)}
    for ax, col, title in ((axes[0], "ay_fit", "posterior fit (latent sideslip)"),
                           (axes[1], "ay_kin", "measured kinematics + calibrated offsets")):
        for r in runs:
            g = seg[seg["run"] == r]
            ax.scatter(g["ay"], g[col], s=14, color=cols[r], alpha=0.8, edgecolors=S.SURFACE, linewidths=0.6,
                       label=r.replace(".csv", ""))
        ax.plot(lim, lim, color=S.AXIS, lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        n_out = int(((g_all := seg[col]) < lim[0]).sum() + (g_all > lim[1]).sum())
        if n_out:
            ax.annotate(f"{n_out} point(s) outside the axes", (0.98, 0.03), xycoords="axes fraction",
                        ha="right", fontsize=7.5, color=S.INK2)
        ax.set_xlabel("measured ay [m/s²]")
        ax.set_ylabel("model ay [m/s²]")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7)
    ax = axes[2]
    if t.loro is not None and len(t.loro):
        rm = t.loro.groupby("run")[["ay", "ay_pred"]].agg(["mean", "std"])
        for i, r in enumerate(rm.index):
            ax.errorbar(rm.loc[r, ("ay", "mean")], rm.loc[r, ("ay_pred", "mean")],
                        xerr=rm.loc[r, ("ay", "std")], yerr=rm.loc[r, ("ay_pred", "std")],
                        fmt="o", color=cols.get(r, S.BLUE), ms=7, capsize=3)
        ax.plot(lim, lim, color=S.AXIS, lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"leave-one-run-out (held-out run means)\nrun-mean RMSE "
                     f"{t.metrics.get('ay_runmean_rmse_loro', float('nan')):.2f} m/s²", fontsize=10)
        ax.set_xlabel("measured ay [m/s²]")
        ax.set_ylabel("predicted ay [m/s²]")
    fig.suptitle("Stage B · lateral balance per quasi-steady segment", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_operating_points(res, path, dpi):
    t, tf = res.track, res.ttc_fit
    seg = t.segments
    lam = t.lambdas
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    a = np.linspace(-16, 4, 300)
    for ax, axle, wheels in ((axes[0], "front", ("FL", "FR")), (axes[1], "rear", ("RL", "RR"))):
        lm = lam[f"LMUY_{axle}"]
        for Fz, c in zip((300, 700, 1100), S.ordinal(3)):
            ax.plot(a, _fy(tf.model, a, Fz, lmuy=lm, lky=lam["LKY"]) / Fz, color=c, lw=1.6,
                    label=f"curve @ {Fz} N")
        for w in wheels:
            ax.scatter(seg[f"alpha_{w}_deg"], seg[f"Fy_{w}"] / seg[f"Fz_{w}"], s=12, color=S.WHEEL_COLORS[w],
                       edgecolors=S.SURFACE, linewidths=0.5, label=w, zorder=4)
        ax.axvline(0, color=S.AXIS, lw=0.8)
        ax.set_xlabel("slip angle α [deg]")
        ax.set_title(f"{axle} axle · LMUY = {lm:.3f}", fontsize=10)
        ax.legend(loc="upper left", ncol=2, fontsize=7.5)
    axes[0].set_ylabel("Fy / Fz  (track curve, normalised)")
    sat = 100 * t.metrics["saturated_fraction"]
    fig.suptitle(f"Operating points on the track curve · {sat:.0f}% of wheel-segments beyond the peak",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_sideslip(res, path, dpi):
    t = res.track
    seg = t.segments
    off = t.beta_offsets
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    runs = list(seg["run"].unique())
    for i, r in enumerate(runs):
        g = seg[seg["run"] == r]
        ax.scatter(g["beta_meas_deg"], g["beta_latent_deg"], s=14, color=S.CATEGORICAL[i % 8],
                   edgecolors=S.SURFACE, linewidths=0.5, label=r.replace(".csv", ""))
    lo = min(seg["beta_meas_deg"].min(), seg["beta_latent_deg"].min()) - 1
    hi = max(seg["beta_meas_deg"].max(), seg["beta_latent_deg"].max()) + 1
    ax.plot([lo, hi], [lo, hi], color=S.AXIS, lw=1)
    ax.set_xlabel("measured CoG sideslip β [deg]")
    ax.set_ylabel("latent (posterior) β [deg]")
    ax.set_title("Errors-in-variables: latent vs. measured sideslip", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1]
    y = np.arange(len(off))
    ax.barh(y, off["offset_deg"], xerr=2 * off["offset_sd_deg"], color=S.BLUE, height=0.55,
            error_kw={"ecolor": S.INK2, "capsize": 3})
    ax.axvline(t.metrics["beta_global_offset_deg"], color=S.ORANGE, lw=1.5, ls="--", label="global part")
    ax.set_yticks(y, [r.replace(".csv", "") for r in off["run"]], fontsize=8)
    ax.set_xlabel("sideslip correction [deg]  (±2σ)")
    for yi, (_, r) in zip(y, off.iterrows()):
        ax.annotate(f"{r['equiv_vy_offset_ms']:+.2f} m/s", (r["offset_deg"], yi), xytext=(6, -10),
                    textcoords="offset points", fontsize=7.5, color=S.INK2)
    ax.legend(loc="upper right")
    ax.set_title("Per-run sensor correction", fontsize=10)
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_handling(res, path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axes[0]
    for h, c, lab in ((res.handling_rig, S.RIG, "rig tyre"), (res.handling_track, S.TRACK, "track tyre")):
        if h is None or not len(h["curve"]):
            continue
        cv = h["curve"]
        ax.plot(cv["ay_g"], cv["understeer_deg"], color=c, lw=2,
                label=f"{lab}: limit {h['ay_limit_g']:.2f} g, K_us {h['Kus_linear_deg_per_g']:+.2f}°/g")
        ax.plot(cv["ay_g"].iloc[-1], cv["understeer_deg"].iloc[-1], "o", color=c, ms=7, mec=S.SURFACE, mew=1.5)
    if res.tel is not None and len(res.tel.segments):
        mp = measured_handling_points(res.tel.segments, res.cfg.vehicle.WB, res.cfg.vehicle.g)
        ax.scatter(mp["ay_g"], mp["understeer_deg"], s=10, color=S.MUTED, alpha=0.6, label="measured segments")
    ax.axhline(0, color=S.AXIS, lw=0.8)
    ax.set_xlabel("lateral acceleration [g]")
    ax.set_ylabel("δ − L/R  [deg]   (+ understeer)")
    R = res.handling_rig["radius_m"] if res.handling_rig else float("nan")
    ax.set_title(f"Handling diagram, R = {R:.2f} m", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    h = res.handling_track or res.handling_rig
    cv = h["curve"]
    for w in WHEELS:
        ax.plot(cv["ay_g"], cv[f"alpha_{w}_deg"], color=S.WHEEL_COLORS[w], ls=S.WHEEL_LS[w], lw=2, label=w)
    ax.set_xlabel("lateral acceleration [g]")
    ax.set_ylabel("slip angle [deg]")
    ax.set_title("Steady-state slip angles ({} tyre)".format("track" if res.handling_track else "rig"), fontsize=10)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_timeseries(res, path, dpi):
    ts = res.timeseries
    runs = list(ts["run"].unique())
    # concatenate runs with gaps on a running time axis
    t_axis, offs = [], 0.0
    for r in runs:
        g = ts[ts["run"] == r]
        t_axis.append(g["time"].to_numpy() - g["time"].iloc[0] + offs)
        offs = t_axis[-1][-1] + 3.0
    t_cat = np.concatenate(t_axis)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, q, lab in ((axes[0], "Fz", "Fz [N]"), (axes[1], "alpha", "α [deg]"), (axes[2], "Fy", "Fy [N]")):
        for w in WHEELS:
            col = f"{q}_{w}_deg" if q == "alpha" else f"{q}_{w}"
            y = ts[col].to_numpy()
            ax.plot(t_cat, pd.Series(y).rolling(25, center=True, min_periods=1).mean(),
                    color=S.WHEEL_COLORS[w], ls=S.WHEEL_LS[w], lw=1.2, label=w)
        ax.set_ylabel(lab)
    axes[0].legend(ncol=4, loc="upper right")
    ax = axes[3]
    ax.plot(t_cat, pd.Series(ts["ay"]).rolling(25, center=True, min_periods=1).mean(), color=S.MUTED, lw=1.2,
            label="measured")
    ax.plot(t_cat, pd.Series(ts["ay_pred"]).rolling(25, center=True, min_periods=1).mean(), color=S.TRACK, lw=1.2,
            label="model (calibrated kinematics)")
    ax.set_ylabel("ay [m/s²]")
    ax.set_xlabel("time, runs concatenated [s]")
    ax.legend(loc="upper right")
    for (r, ta) in zip(runs, t_axis):
        axes[0].annotate(r.replace(".csv", "").replace("processed_", ""), (ta[0], 1.0),
                         xycoords=("data", "axes fraction"), fontsize=7, color=S.INK2, va="bottom")
        for ax in axes:
            ax.axvline(ta[0], color=S.GRID, lw=1)
    fig.suptitle("Per-wheel reconstruction (0.25 s moving mean)", x=0.01, ha="left", fontweight="semibold", y=1.01)
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_audit(res, path, dpi):
    runs = res.tel.runs
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for i, (_, r) in enumerate(runs.iterrows()):
        c = S.BLUE if r["used"] else S.CRITICAL
        ax.scatter(r["ackermann_med_deg"], r["steer_rw_med_deg"], s=60, color=c, marker="o" if r["used"] else "X",
                   edgecolors=S.SURFACE, linewidths=1, zorder=4)
        ax.annotate(r["run"].replace("processed_", "").replace(".csv", ""), (r["ackermann_med_deg"], r["steer_rw_med_deg"]),
                    xytext=(6, 8 if i % 2 else -10), textcoords="offset points", fontsize=7, color=S.INK2)
    lim = np.array([runs["ackermann_med_deg"].min() - 2, 2])
    ax.plot(lim, lim, color=S.AXIS, lw=1, label="neutral steer (δ = L/R)")
    ax.axhline(0, color=S.AXIS, lw=0.8)
    ax.scatter([], [], color=S.BLUE, label="used")
    ax.scatter([], [], color=S.CRITICAL, marker="X", label="quarantined")
    ax.set_xlabel("Ackermann angle L·r/Vx [deg]")
    ax.set_ylabel("road-wheel steer [deg]")
    ax.set_title("Steering-sign audit (run medians)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    y = np.arange(len(runs))
    ax.barh(y, runs["kin_resid_rms"], color=[S.BLUE if u else S.MUTED for u in runs["used"]], height=0.6)
    ax.set_yticks(y, [r.replace("processed_", "").replace(".csv", "") for r in runs["run"]], fontsize=8)
    ax.set_xlabel("RMS of ay − (dVy/dt + Vx r) [m/s²]")
    ax.set_title("Kinematic-identity residual", fontsize=10)
    ax = axes[2]
    ax.barh(y, runs["beta_meas_std_deg"], color=[S.BLUE if u else S.MUTED for u in runs["used"]], height=0.6)
    for yi, s in zip(y, runs["segments"]):
        ax.annotate(f"{s} seg.", (0, yi), xytext=(3, 0), textcoords="offset points", fontsize=7, color=S.SURFACE,
                    va="center", fontweight="bold")
    ax.set_yticks(y, [""] * len(runs))
    ax.set_xlabel("within-run std of measured sideslip [deg]")
    ax.set_title("Sideslip channel noise", fontsize=10)
    fig.tight_layout()
    return _save(fig, path, dpi)


def fig_training(res, path, dpi):
    fig, axes = plt.subplots(1, 2 if res.track else 1, figsize=(12 if res.track else 6, 3.8), squeeze=False)
    h = np.array(res.ttc_fit.history)
    ax = axes[0, 0]
    ax.plot(h - h.min() + 1e-6, color=S.RIG, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("iteration (Adam, logged every 10, then L-BFGS)")
    ax.set_ylabel("objective − min")
    ax.set_title("Stage A convergence", fontsize=10)
    if res.track:
        h2 = np.array(res.track.history)
        ax = axes[0, 1]
        ax.plot(h2 - h2.min() + 1e-6, color=S.TRACK, lw=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title("Stage B convergence (best start)", fontsize=10)
    fig.tight_layout()
    return _save(fig, path, dpi)


def make_all(res) -> Dict[str, str]:
    fig_dir = Path(res.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = res.cfg.output.dpi
    jobs = [("01_ttc_fit", fig_ttc_fit), ("02_tyre_curves", fig_tyre_curves),
            ("03_characteristics", fig_characteristics), ("04_identifiability", fig_identifiability),
            ("05_multistart", fig_multistart), ("09_handling", fig_handling), ("12_convergence", fig_training)]
    if res.tel is not None:
        jobs.append(("11_data_audit", fig_audit))
    if res.track is not None:
        jobs += [("06_balance", fig_balance), ("07_operating_points", fig_operating_points),
                 ("08_sideslip", fig_sideslip)]
        if res.timeseries is not None and len(res.timeseries):
            jobs.append(("10_timeseries", fig_timeseries))
    out = {}
    for name, fn in sorted(jobs):
        try:
            out[f"fig_{name}"] = fn(res, fig_dir / f"{name}.png", dpi)
        except Exception as exc:  # a broken figure must never kill a run
            print(f"  [plot] {name} failed: {exc}")
    return out
