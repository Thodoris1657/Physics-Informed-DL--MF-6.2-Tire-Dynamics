"""
Skidpad telemetry: loading, channel standardisation, per-run audit and
quasi-steady segmentation.

Why segments?
-------------
At 100 Hz the samples of one run are strongly autocorrelated, and the v1
audit showed the kinematic slip angle carries 1-4 deg of within-run noise
plus a run-to-run bias. Fitting the balances sample-by-sample (v1) therefore
gives the optimiser a strong incentive to hide every wheel on the flat,
post-peak part of the tyre curve (errors-in-variables attenuation).

v2 instead cuts each run into short windows (default 1 s), keeps the ones that
are quasi-steady and close to pure-lateral, and fits the window *means*, each
with its own latent sideslip angle (see :mod:`mf62pinn.fit.track`). The
within-window spread of the measured sideslip becomes that window's
measurement-noise estimate.
"""

from __future__ import annotations

import fnmatch
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..config import TelemetryConfig, VehicleParams
from .io import read_table

CANON = ("time", "steer", "ax", "ay", "vx", "vy", "yaw_rate")


@dataclass
class TelemetryData:
    samples: pd.DataFrame        # all retained samples (standardised SI channels)
    segments: pd.DataFrame       # quasi-steady window means
    runs: pd.DataFrame           # per-run audit table
    notes: List[Dict] = field(default_factory=list)

    @property
    def run_names(self) -> List[str]:
        return list(self.runs.loc[self.runs["used"], "run"])


def _smooth(x: np.ndarray, wlen: int = 15, poly: int = 3) -> np.ndarray:
    if len(x) < wlen + 2:
        return x
    return savgol_filter(x, wlen | 1, poly)


def _standardise(raw: pd.DataFrame, cfg: TelemetryConfig, veh: VehicleParams, run: str) -> pd.DataFrame:
    cols = cfg.columns
    missing = [c for c in CANON if cols.get(c) not in raw.columns]
    if missing:
        raise KeyError(f"{run}: missing channels {[(m, cols.get(m)) for m in missing]}")
    d = pd.DataFrame({c: raw[cols[c]].to_numpy(dtype=float) for c in CANON})
    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    is_processed = all(f"{cols[c]}_raw" in raw.columns for c in ("steer", "ax", "ay", "vx", "vy", "yaw_rate"))
    if cfg.smooth_raw and not is_processed:
        for c in ("steer", "ax", "ay", "vx", "vy", "yaw_rate"):
            d[c] = _smooth(d[c].to_numpy())

    if cfg.accel_unit == "g":
        d["ax"] *= veh.g
        d["ay"] *= veh.g
    if cfg.speed_unit == "km/h":
        d["vx"] /= 3.6
        d["vy"] /= 3.6
    if cfg.yaw_rate_unit == "deg/s":
        d["yaw_rate"] = np.radians(d["yaw_rate"])
    steer = d["steer"].to_numpy()
    if cfg.steer_unit == "deg":
        steer = np.radians(steer)
    if not cfg.steer_is_roadwheel:
        steer = steer / veh.steer_ratio
    d["delta"] = steer

    t = d["time"].to_numpy()
    dt = np.gradient(t)
    dt = np.where(dt < 1e-6, np.nanmedian(np.diff(t)) if len(t) > 2 else 0.01, dt)
    d["r_dot"] = _smooth(np.gradient(d["yaw_rate"].to_numpy()) / dt, wlen=21, poly=3)
    vy_cog = d["vy"] - veh.vy_sensor_x * d["yaw_rate"]
    d["beta_meas"] = np.arctan2(vy_cog, d["vx"])
    # kinematic identity residual  ay - (dVy/dt + Vx r)
    d["kin_resid"] = d["ay"] - (np.gradient(vy_cog.to_numpy()) / dt + d["vx"] * d["yaw_rate"])
    d["run"] = run
    return d


def _audit_run(d: pd.DataFrame, veh: VehicleParams) -> Dict:
    core = d[np.abs(d["ay"]) > 0.5 * veh.g]
    if len(core) < 20:
        core = d
    delta_ack = veh.WB * core["yaw_rate"] / core["vx"]
    turn = float(np.sign(np.median(core["yaw_rate"])))
    steer_med = float(np.median(core["delta"]))
    ack_med = float(np.median(delta_ack))
    ay_g = float(np.median(np.abs(core["ay"]))) / veh.g
    # understeer angle (positive = understeer) and implied gradient
    us_deg = math.degrees((steer_med - ack_med) * turn)
    return {
        "samples": len(d),
        "turn": "left" if turn < 0 else "right",
        "vx_med": float(np.median(core["vx"])),
        "radius_m": float(np.median(core["vx"] / np.abs(core["yaw_rate"]))),
        "ay_med_g": ay_g,
        "steer_rw_med_deg": math.degrees(steer_med),
        "ackermann_med_deg": math.degrees(ack_med),
        "steer_sign_ok": bool(np.sign(steer_med) == np.sign(ack_med) or abs(math.degrees(steer_med)) < 0.5),
        "Kus_deg_per_g": us_deg / max(ay_g, 1e-3),
        "beta_meas_med_deg": math.degrees(float(np.median(core["beta_meas"]))),
        "beta_meas_std_deg": math.degrees(float(np.std(core["beta_meas"]))),
        "kin_resid_rms": float(np.sqrt(np.mean(core["kin_resid"] ** 2))),
        "ax_std": float(np.std(core["ax"])),
    }


def segment_run(d: pd.DataFrame, cfg: TelemetryConfig, veh: VehicleParams) -> pd.DataFrame:
    """Cut one run into windows and keep the quasi-steady, near-pure-lateral ones."""
    t = d["time"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.01
    n = max(int(round(cfg.window_s / dt)), 5)
    speed_ok = ((d["vx"] >= cfg.vx_min) & (d["vx"] <= cfg.vx_max)).to_numpy()
    rows = []
    for k in range(len(d) // n):
        sl = slice(k * n, (k + 1) * n)
        w = d.iloc[sl]
        fill = float(speed_ok[sl].mean())
        if fill < cfg.min_fill:
            continue
        w = w[speed_ok[sl]]
        m = w[["vx", "vy", "yaw_rate", "r_dot", "delta", "ax", "ay", "beta_meas"]].mean()
        if abs(m["r_dot"]) > cfg.rdot_max or w["yaw_rate"].std() > cfg.r_std_max:
            continue
        if abs(m["ax"]) > cfg.ax_max or abs(m["ay"]) < cfg.ay_min_g * veh.g:
            continue
        # effective sample size of the window mean (lag-1 autocorrelation of beta)
        b = w["beta_meas"].to_numpy()
        b = b - b.mean()
        rho = float(np.clip(np.dot(b[1:], b[:-1]) / max(np.dot(b, b), 1e-12), 0.0, 0.999))
        n_eff = max(len(b) * (1 - rho) / (1 + rho), 1.0)
        rows.append({
            "run": d["run"].iloc[0], "t0": float(w["time"].iloc[0]), "n": len(w),
            "Vx": m["vx"], "Vy": m["vy"], "r": m["yaw_rate"], "r_dot": m["r_dot"],
            "delta": m["delta"], "ax": m["ax"], "ay": m["ay"], "beta_meas": m["beta_meas"],
            "beta_std": float(w["beta_meas"].std()), "beta_sem": float(w["beta_meas"].std() / math.sqrt(n_eff)),
            "ay_std": float(w["ay"].std()),
        })
    return pd.DataFrame(rows)


def load_telemetry(paths: List[str], cfg: TelemetryConfig, veh: VehicleParams) -> TelemetryData:
    samples, segs, audits, notes = [], [], [], []
    for path in paths:
        run = os.path.basename(path)
        try:
            d = _standardise(read_table(path), cfg, veh, run)
        except Exception as exc:  # noqa: BLE001 - report and continue
            notes.append({"file": run, "level": "error", "msg": str(exc)})
            continue
        audit = {"run": run, **_audit_run(d, veh)}
        used, reason = True, ""
        if any(fnmatch.fnmatch(run, p) for p in cfg.exclude_runs):
            used, reason = False, "excluded by config"
        elif not audit["steer_sign_ok"]:
            msg = (f"steering sign contradicts the turn direction "
                   f"(road-wheel {audit['steer_rw_med_deg']:+.1f} deg vs Ackermann "
                   f"{audit['ackermann_med_deg']:+.1f} deg)")
            if cfg.steer_sign_policy == "flip":
                d["delta"] = -d["delta"]
                audit = {"run": run, **_audit_run(d, veh)}
                notes.append({"file": run, "level": "warn", "msg": msg + " -> sign flipped"})
                reason = "steer sign flipped"
            elif cfg.steer_sign_policy == "exclude":
                used, reason = False, "steer sign inconsistent"
                notes.append({"file": run, "level": "warn", "msg": msg + " -> run quarantined"})
            else:
                notes.append({"file": run, "level": "warn", "msg": msg + " (kept as is)"})
        audit["used"], audit["reason"] = used, reason
        seg = segment_run(d, cfg, veh) if used else pd.DataFrame()
        audit["segments"] = len(seg)
        audit["samples_in_window"] = int(((d["vx"] >= cfg.vx_min) & (d["vx"] <= cfg.vx_max)).sum())
        audits.append(audit)
        d["used"] = used
        samples.append(d)
        if used and len(seg):
            segs.append(seg)
    if not samples:
        raise RuntimeError("No telemetry could be loaded")
    S = pd.concat(samples, ignore_index=True)
    G = pd.concat(segs, ignore_index=True) if segs else pd.DataFrame()
    runs = pd.DataFrame(audits)
    if len(G):
        G["run_id"] = G["run"].map({r: i for i, r in enumerate(runs.loc[runs["used"], "run"])}).astype(int)
    return TelemetryData(samples=S, segments=G, runs=runs, notes=notes)
