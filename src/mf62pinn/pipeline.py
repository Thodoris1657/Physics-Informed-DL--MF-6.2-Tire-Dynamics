"""
End-to-end pipeline: data -> audit -> rig fit -> track calibration ->
validation -> exports (.tir, JSON, CSV) -> figures -> HTML report.

    from mf62pinn import RunConfig, run_pipeline
    res = run_pipeline(RunConfig.from_yaml("configs/p26.yaml"))
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import __version__
from .analysis import characteristics as ch
from .analysis.handling import grip_limit_by_run, handling_diagram
from .config import RunConfig
from .data import TelemetryData, TTCData, expand_files, load_telemetry, load_ttc
from .export.tir import write_tir
from .fit.common import DTYPE
from .fit.track import SegTensors, TrackFitResult, TrackModel, fit_track
from .fit.ttc import TTCFitResult, fit_ttc
from .vehicle import WHEELS

Progress = Optional[Callable[[str, float], None]]
D2R = math.pi / 180.0


@dataclass
class PipelineResult:
    cfg: RunConfig
    out_dir: Path
    ttc: TTCData
    ttc_fit: TTCFitResult
    tel: Optional[TelemetryData] = None
    track: Optional[TrackFitResult] = None
    rig_model: Optional[TrackModel] = None          # vehicle model with LMUY = LKY = 1
    handling_rig: Optional[Dict] = None
    handling_track: Optional[Dict] = None
    grip_by_run: Optional[pd.DataFrame] = None
    char_rig: Optional[pd.DataFrame] = None
    char_track: Optional[pd.DataFrame] = None
    coef_table: Optional[pd.DataFrame] = None
    timeseries: Optional[pd.DataFrame] = None
    findings: List[Dict] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def headline(self) -> Dict[str, float]:
        h = {"ttc_r2": self.ttc_fit.metrics["r2"], "ttc_rmse_N": self.ttc_fit.metrics["rmse_N"]}
        rig = self.char_rig[(self.char_rig["side"] == "+") & np.isclose(self.char_rig["Fz_N"], round(self.ttc_fit.Fz0, 1), atol=0.06)]
        if len(rig):
            r = rig.iloc[0]
            h.update({"rig_mu_peak_Fz0": r["mu_peak"], "rig_alpha_peak_Fz0_deg": r["alpha_peak_deg"],
                      "rig_Kya_Fz0_N_per_deg": r["Kya_N_per_deg"]})
        if self.track is not None:
            h.update({k: v for k, v in self.track.lambdas.items()})
            h["track_mu_peak_Fz0_front"] = self.track.lambdas["LMUY_front"] * h.get("rig_mu_peak_Fz0", float("nan"))
            h["track_mu_peak_Fz0_rear"] = self.track.lambdas["LMUY_rear"] * h.get("rig_mu_peak_Fz0", float("nan"))
            h["saturated_fraction"] = self.track.metrics["saturated_fraction"]
        if self.handling_track is not None:
            h["ay_limit_g"] = self.handling_track["ay_limit_g"]
            h["Kus_linear_deg_per_g"] = self.handling_track["Kus_linear_deg_per_g"]
        return h


def _say(progress: Progress, msg: str, frac: float, verbose: bool) -> None:
    if verbose:
        print(f"[{frac * 100:5.1f}%] {msg}", flush=True)
    if progress:
        progress(msg, frac)


def _rig_model(ttc_fit: TTCFitResult, cfg: RunConfig, n_runs: int, n_seg: int) -> TrackModel:
    m = TrackModel(ttc_fit.model, ttc_fit, cfg.vehicle, cfg, max(n_runs, 1), max(n_seg, 1))
    with torch.no_grad():
        m.log_lmu.zero_()
        m.log_lky.zero_()
    return m


def _findings(res: PipelineResult) -> List[Dict]:
    """Plain-language conclusions with a severity for the report header."""
    out: List[Dict] = []
    for n in res.ttc.notes:
        out.append({"level": n["level"], "area": "TTC data", "msg": f"{n['file']}: {n['msg']}"})
    if res.tel is not None:
        for n in res.tel.notes:
            out.append({"level": n["level"], "area": "Telemetry", "msg": f"{n['file']}: {n['msg']}"})
        bad = res.tel.runs[res.tel.runs["kin_resid_rms"] > 1.5]
        if len(bad):
            out.append({"level": "warn", "area": "Telemetry",
                        "msg": f"kinematic identity ay = dVy/dt + Vx r violated by {bad['kin_resid_rms'].median():.1f} m/s^2 "
                               f"RMS in {len(bad)} run(s): the lateral-velocity channel is noisy/biased"})
    m = res.ttc_fit.metrics
    out.append({"level": "good", "area": "Rig fit",
                "msg": f"MF6.2 reproduces the TTC sweeps with R^2 = {m['r2']:.4f}, RMSE = {m['rmse_N']:.1f} N "
                       f"({m['n_rows']} rows, effective sample size {m['n_eff']:.0f})"})
    a_max = float(np.degrees(res.ttc.df["alpha"].abs().quantile(0.99)))
    if res.char_rig is not None:
        beyond = res.char_rig[(res.char_rig["side"] == "+") & (res.char_rig["alpha_peak_deg"] > a_max)
                              & (res.char_rig["Fz_N"] <= res.ttc.df["Fz"].max() * 1.1)]
        if len(beyond):
            out.append({"level": "info", "area": "Rig fit",
                        "msg": f"peak slip above {beyond['Fz_N'].min():.0f} N lies beyond the tested ±{a_max:.0f} deg "
                               "(post-peak shape extrapolated there)"})
    if res.ttc_fit.frozen:
        out.append({"level": "info", "area": "Rig fit",
                    "msg": "structurally non-identifiable (no camber in the data), frozen at 0: " + ", ".join(res.ttc_fit.frozen)})
    if m["n_near_optimal"] < max(2, m["n_starts"] // 4):
        out.append({"level": "warn", "area": "Rig fit",
                    "msg": f"only {m['n_near_optimal']}/{m['n_starts']} multi-starts reached the optimum: check the data coverage"})
    if res.track is not None:
        tm = res.track.metrics
        lam = res.track.lambdas
        sd = res.track.lambdas_std
        out.append({"level": "good", "area": "Track grip",
                    "msg": f"friction scale LMUY = {lam['LMUY_front']:.3f} +/- {sd['LMUY_front']:.3f} (front), "
                           f"{lam['LMUY_rear']:.3f} +/- {sd['LMUY_rear']:.3f} (rear)"})
        if res.track.loro is not None and len(res.track.loro):
            lo = res.track.loro
            out.append({"level": "info", "area": "Track grip",
                        "msg": f"leave-one-run-out LMUY range: front {lo['lmu_f'].min():.3f}-{lo['lmu_f'].max():.3f}, "
                               f"rear {lo['lmu_r'].min():.3f}-{lo['lmu_r'].max():.3f}"})
        if res.grip_by_run is not None and len(res.grip_by_run):
            g = res.grip_by_run
            best = g.loc[g["ay_driven_mean_g"].idxmax()]
            out.append({"level": "good", "area": "Validation",
                        "msg": f"predicted steady-state grip limit {best['ay_limit_pred_g']:.2f} g at R = "
                               f"{best['radius_m']:.1f} m vs {best['ay_driven_mean_g']:.2f} g mean / "
                               f"{best['ay_driven_p90_g']:.2f} g p90 driven in the best run"})
        if tm["saturated_fraction"] > 0.6:
            out.append({"level": "warn", "area": "Operating point",
                        "msg": f"{100 * tm['saturated_fraction']:.0f}% of wheel-segments sit beyond the tyre peak. The car was driven "
                               "at its grip limit, so the telemetry fixes the friction level but not the on-track "
                               "slip angles / cornering stiffness (they are prior-dominated)."})
        starts = pd.DataFrame(res.track.starts)
        if len(starts):
            best = starts["neg_log_post"].min()
            alt = starts[(starts["neg_log_post"] < best + 5.0) &
                         (np.abs(starts["b_global_deg"] - starts.loc[starts["neg_log_post"].idxmin(), "b_global_deg"]) > 1.0)]
            if len(alt):
                out.append({"level": "warn", "area": "Operating point",
                            "msg": "competing sideslip explanations within 5 nats of the optimum (multi-modal posterior)"})
        if abs(tm["beta_global_offset_deg"]) > 2.5:
            out.append({"level": "warn", "area": "Sensors",
                        "msg": f"global sideslip correction of {tm['beta_global_offset_deg']:+.1f} deg: treat as a nuisance "
                               "parameter, not a calibrated sensor bias"})
    return out


def run_pipeline(cfg: RunConfig, base_dir: str | os.PathLike = ".", progress: Progress = None,
                 verbose: bool = True, make_plots: Optional[bool] = None,
                 make_report: Optional[bool] = None) -> PipelineResult:
    t_all = time.time()
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    base_dir = Path(base_dir)
    out = Path(cfg.output.out_dir)
    if not out.is_absolute():
        out = base_dir / out
    out.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    # ---------------- data ---------------- #
    _say(progress, "loading TTC data", 0.01, verbose)
    ttc_files = expand_files(cfg.ttc.files, base_dir)
    if not ttc_files:
        raise FileNotFoundError(f"No TTC files match {cfg.ttc.files} (base {base_dir})")
    ttc = load_ttc(ttc_files, cfg.ttc)
    tel = None
    tel_files = expand_files(cfg.telemetry.files, base_dir) if cfg.track_fit.enabled else []
    if tel_files:
        _say(progress, f"loading {len(tel_files)} telemetry file(s)", 0.03, verbose)
        tel = load_telemetry(tel_files, cfg.telemetry, cfg.vehicle)

    # ---------------- stage A: rig ---------------- #
    t0 = time.time()
    _say(progress, "stage A - rig identification (multi-start)", 0.05, verbose)
    ttc_fit = fit_ttc(ttc, cfg, progress=progress)
    timings["ttc_fit_s"] = time.time() - t0
    if verbose:
        print(f"         R^2 = {ttc_fit.metrics['r2']:.5f}   RMSE = {ttc_fit.metrics['rmse_N']:.1f} N")

    res = PipelineResult(cfg=cfg, out_dir=out, ttc=ttc, ttc_fit=ttc_fit, tel=tel, timings=timings)

    # ---------------- stage B: track ---------------- #
    n_runs = 0
    if tel is not None and len(tel.segments) >= 10:
        n_runs = int(tel.runs["used"].sum())
        t0 = time.time()
        _say(progress, f"stage B - errors-in-variables track calibration ({len(tel.segments)} segments, {n_runs} runs)", 0.5, verbose)
        res.track = fit_track(tel.segments, ttc_fit, cfg, n_runs, progress=progress,
                              loro=cfg.validation.leave_one_run_out)
        timings["track_fit_s"] = time.time() - t0
        if verbose:
            lam = res.track.lambdas
            print(f"         LMUY front = {lam['LMUY_front']:.3f}   rear = {lam['LMUY_rear']:.3f}   LKY = {lam['LKY']:.3f}")
    elif tel is not None:
        res.findings.append({"level": "warn", "area": "Telemetry", "msg": "fewer than 10 quasi-steady segments: track stage skipped"})

    # ---------------- analysis ---------------- #
    _say(progress, "analysis - characteristic values, handling diagram", 0.9, verbose)
    loads = sorted(set([round(float(f), 1) for f in ttc.df.groupby("source")["Fz"].mean()] + [round(ttc_fit.Fz0, 1)]))
    loads = sorted(set(loads + [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0]))
    res.char_rig = ch.table(ttc_fit.model, loads, ttc_fit.cov_u)
    res.coef_table = ch.coefficient_table(ttc_fit.values, ttc_fit.std, ttc_fit.frozen, ttc_fit.model.specs)
    vc = cfg.validation
    n_seg = len(tel.segments) if (tel is not None and len(tel.segments)) else 1
    res.rig_model = _rig_model(ttc_fit, cfg, n_runs, n_seg)
    R_h = vc.handling_radius_m
    if R_h is None:
        R_h = (float(np.median(tel.segments["Vx"] / np.abs(tel.segments["r"])))
               if (tel is not None and len(tel.segments)) else 9.125)
    res.handling_rig = handling_diagram(res.rig_model, R_h, vc.handling_vmin, 30.0, vc.handling_points)
    if res.track is not None:
        lam = res.track.lambdas
        lm = 0.5 * (lam["LMUY_front"] + lam["LMUY_rear"])
        res.char_track = ch.table(ttc_fit.model, loads, ttc_fit.cov_u, lmuy=lm, lky=lam["LKY"],
                                  lmuy_sd=0.5 * (res.track.lambdas_std["LMUY_front"] + res.track.lambdas_std["LMUY_rear"]))
        res.handling_track = handling_diagram(res.track.model, R_h, vc.handling_vmin,
                                              vc.handling_vmax, vc.handling_points)
        res.grip_by_run = grip_limit_by_run(res.track.model, tel.segments, cfg.vehicle.g)
        res.timeseries = per_sample_timeseries(res.track.model, tel, cfg)

    order = {"error": 0, "good": 1, "warn": 2, "info": 3}
    res.findings = sorted(_findings(res) + res.findings, key=lambda f: order.get(f["level"], 9))

    # ---------------- exports ---------------- #
    _say(progress, "writing results", 0.93, verbose)
    export_all(res)
    if make_plots if make_plots is not None else cfg.output.plots:
        _say(progress, "rendering figures", 0.95, verbose)
        from .viz.plots import make_all
        res.files.update(make_all(res))
    timings["total_s"] = time.time() - t_all
    if make_report if make_report is not None else cfg.output.report:
        from .report import write_report
        res.files["report"] = str(write_report(res))
    timings["total_s"] = time.time() - t_all
    (out / "metrics.json").write_text(json.dumps(_metrics_payload(res), indent=2, default=_json_default))
    _say(progress, f"done in {timings['total_s']:.0f} s -> {out}", 1.0, verbose)
    return res


# --------------------------------------------------------------------------- #
#  Per-sample reconstruction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def per_sample_timeseries(model: TrackModel, tel: TelemetryData, cfg: RunConfig) -> pd.DataFrame:
    """Wheel loads, slip angles and forces for every telemetry sample of the used runs,
    with the calibrated (global + run) sideslip correction."""
    S = tel.samples[tel.samples["used"]].copy()
    run_ids = {r: i for i, r in enumerate(tel.runs.loc[tel.runs["used"], "run"])}
    S = S[S["run"].isin(run_ids)].reset_index(drop=True)
    if not len(S):
        return pd.DataFrame()
    rid = torch.tensor(S["run"].map(run_ids).to_numpy(), dtype=torch.long)
    t = lambda c: torch.tensor(S[c].to_numpy(), dtype=DTYPE)  # noqa: E731
    T = SegTensors(t("vx"), t("yaw_rate"), t("r_dot"), t("delta"), t("ax"), t("ay"), t("beta_meas"),
                   torch.zeros(len(S), dtype=DTYPE), rid, model.z_run.numel())
    off = model.s_g * model.z_g[0] + (model.s_r * model.z_run[rid] if model.eiv else 0.0)
    o = model.forward(T, beta=T.beta_meas + off)
    out = pd.DataFrame({"run": S["run"], "time": S["time"], "vx": S["vx"], "vy": S["vy"],
                        "yaw_rate": S["yaw_rate"], "r_dot": S["r_dot"], "ax": S["ax"], "ay": S["ay"],
                        "delta_deg": S["delta"] / D2R, "beta_meas_deg": S["beta_meas"] / D2R,
                        "beta_corr_deg": o["beta"].numpy() / D2R})
    for i, w in enumerate(WHEELS):
        out[f"Fz_{w}"] = o["Fz"][i].numpy()
        out[f"alpha_{w}_deg"] = o["alpha"][i].numpy() / D2R
        out[f"Fy_{w}"] = o["Fy"][i].numpy()
    out["ay_pred"] = o["Fy_body"].numpy() / cfg.vehicle.m_tot
    out["rdot_pred"] = o["Mz"].numpy() / cfg.vehicle.Iz
    in_window = (S["vx"] >= cfg.telemetry.vx_min) & (S["vx"] <= cfg.telemetry.vx_max)
    out["in_speed_window"] = in_window.to_numpy()
    return out


# --------------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------------- #


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _metrics_payload(res: PipelineResult) -> Dict:
    p = {"version": __version__, "headline": res.headline, "ttc": res.ttc_fit.metrics,
         "timings": res.timings, "findings": res.findings,
         "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "platform": platform.platform()}}
    if res.track is not None:
        p["track"] = res.track.metrics
        p["lambdas"] = res.track.lambdas
        p["lambdas_sd"] = res.track.lambdas_std
    if res.handling_track is not None:
        p["handling"] = {k: v for k, v in res.handling_track.items() if k != "curve"}
    return p


def export_all(res: PipelineResult) -> None:
    cfg, out, tf = res.cfg, res.out_dir, res.ttc_fit
    files = res.files
    cfg.to_yaml(out / "config_used.yaml")
    files["config"] = str(out / "config_used.yaml")

    # coefficients
    coeffs = {"model": "MF6.2 pure lateral", "convention": "internal (alpha>0 -> Fy>0); see .tir for ISO-W",
              "Fz0_N": tf.Fz0, "rig": tf.values, "rig_sd": tf.std, "frozen": tf.frozen}
    if res.track is not None:
        coeffs["track_scaling"] = res.track.lambdas
        coeffs["track_scaling_sd"] = res.track.lambdas_std
        coeffs["track_tyre"] = res.track.tyre_values
    (out / "mf62_coefficients.json").write_text(json.dumps(coeffs, indent=2, default=_json_default))
    res.coef_table.to_csv(out / "mf62_coefficients.csv", index=False)
    files["coefficients_json"] = str(out / "mf62_coefficients.json")
    files["coefficients_csv"] = str(out / "mf62_coefficients.csv")
    res.char_rig.to_csv(out / "characteristics_rig.csv", index=False)
    if res.char_track is not None:
        res.char_track.to_csv(out / "characteristics_track.csv", index=False)
    tf.per_source.to_csv(out / "ttc_fit_per_load.csv", index=False)
    tf.starts.to_csv(out / "ttc_multistart.csv", index=False)
    tf.corr.to_csv(out / "ttc_posterior_correlation.csv")

    # .tir files
    if cfg.output.tir:
        P = res.ttc.df["P_kPa"].mean()
        p_pa = float(P) * 1000.0 if np.isfinite(P) else 68947.6
        Vk = res.ttc.df["V_kph"].mean()
        longvl = float(Vk) / 3.6 if np.isfinite(Vk) else 11.176
        common = dict(tyre_name=cfg.tyre_name, unloaded_radius=cfg.vehicle.Rw, pressure_pa=p_pa, longvl=longvl,
                      fz_range=(float(res.ttc.df["Fz"].min()), float(res.ttc.df["Fz"].max())),
                      alpha_range_deg=(float(np.degrees(res.ttc.df["alpha"].min())),
                                       float(np.degrees(res.ttc.df["alpha"].max()))),
                      stds=tf.std)
        files["tir_rig"] = str(write_tir(out / "tyre_rig_TTC.tir", tf.values, tf.Fz0,
                                         comment="Rig tyre (Calspan TTC belt), LMUY = 1.", **common))
        if res.track is not None:
            lam = res.track.lambdas
            for axle in ("front", "rear"):
                files[f"tir_track_{axle}"] = str(write_tir(
                    out / f"tyre_track_{axle}.tir", res.track.tyre_values, tf.Fz0,
                    lmuy=lam[f"LMUY_{axle}"], lky=lam["LKY"],
                    comment=f"Track-calibrated {axle} axle: rig shape with LMUY = {lam[f'LMUY_{axle}']:.4f} "
                            f"+/- {res.track.lambdas_std[f'LMUY_{axle}']:.4f}", **common))

    # telemetry products
    if res.tel is not None:
        res.tel.runs.to_csv(out / "telemetry_audit.csv", index=False)
    if res.track is not None:
        res.track.segments.to_csv(out / "segments.csv", index=False)
        res.track.beta_offsets.to_csv(out / "sideslip_offsets.csv", index=False)
        pd.DataFrame(res.track.starts).to_csv(out / "track_multistart.csv", index=False)
        if res.track.loro is not None:
            res.track.loro.to_csv(out / "validation_leave_one_run_out.csv", index=False)
        if res.grip_by_run is not None:
            res.grip_by_run.to_csv(out / "validation_grip_limit.csv", index=False)
        if res.handling_track is not None:
            res.handling_track["curve"].to_csv(out / "handling_diagram.csv", index=False)
        if res.timeseries is not None and cfg.output.timeseries and len(res.timeseries):
            res.timeseries.to_csv(out / "per_wheel_timeseries.csv", index=False)
            files["timeseries"] = str(out / "per_wheel_timeseries.csv")

    # model state
    torch.save({"version": __version__, "Fz0": tf.Fz0, "rig_u": tf.model.u.detach(),
                "frozen": tf.frozen, "rig_values": tf.values,
                "track_state": res.track.model.state_dict() if res.track is not None else None,
                "config": cfg.to_dict()}, out / "model.pt")
    files["model"] = str(out / "model.pt")
    (out / "SUMMARY.txt").write_text(summary_text(res), encoding="utf-8")
    files["summary"] = str(out / "SUMMARY.txt")


def summary_text(res: PipelineResult) -> str:
    tf, cfg = res.ttc_fit, res.cfg
    v = cfg.vehicle
    L = ["=" * 72, f" MF6.2 PINN v{__version__} - {cfg.tyre_name} on {v.name}",
         f" {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 72, "",
         "FINDINGS"]
    for f in res.findings:
        L.append(f"  [{f['level'].upper():5s}] {f['area']}: {f['msg']}")
    L += ["", "RIG FIT (TTC)",
          f"  rows {tf.metrics['n_rows']}  R^2 {tf.metrics['r2']:.5f}  RMSE {tf.metrics['rmse_N']:.1f} N  "
          f"(sigma_mu {tf.sigma_mu:.4f}, AR(1) inflation {tf.inflation:.1f})",
          f"  Fz0 = {tf.Fz0:.1f} N", "", "  coefficient        value        1-sd     status"]
    for _, r in res.coef_table.iterrows():
        sd = f"{r['sd']:.4g}" if np.isfinite(r["sd"]) else "-"
        L.append(f"  {r['param']:<10s} {r['value']:>14.6g} {sd:>11s}     {r['status']}")
    rig = res.char_rig[res.char_rig["side"] == "+"]
    L += ["", "  Fz [N]   mu_peak         alpha_peak [deg]   Kya [N/deg]"]
    for _, r in rig.iterrows():
        L.append(f"  {r['Fz_N']:7.0f}   {r['mu_peak']:.3f} +/- {r['mu_peak_sd']:.3f}   "
                 f"{r['alpha_peak_deg']:6.2f} +/- {r['alpha_peak_deg_sd']:.2f}    {r['Kya_N_per_deg']:7.1f} +/- {r['Kya_N_per_deg_sd']:.1f}")
    if res.track is not None:
        t = res.track
        L += ["", "TRACK CALIBRATION (errors-in-variables)"]
        for k, val in t.lambdas.items():
            L.append(f"  {k:<11s} = {val:.4f} +/- {t.lambdas_std.get(k, 0):.4f}")
        for k in ("segments", "runs", "ay_r2_fit", "ay_rmse_fit", "ay_rmse_kin", "ay_runmean_rmse_kin",
                  "ay_rmse_loro", "ay_runmean_rmse_loro", "beta_global_offset_deg", "saturated_fraction"):
            if k in t.metrics:
                val = t.metrics[k]
                L.append(f"  {k:<26s} {val:.4f}" if isinstance(val, float) else f"  {k:<26s} {val}")
        L += ["", "  per-run sideslip correction [deg]:"]
        for _, r in t.beta_offsets.iterrows():
            L.append(f"    {r['run']:<30s} {r['offset_deg']:+6.2f} +/- {r['offset_sd_deg']:.2f}   (= {r['equiv_vy_offset_ms']:+.2f} m/s)")
    if res.handling_track is not None:
        h = res.handling_track
        L += ["", f"STEADY STATE (R = {h['radius_m']:.2f} m, track tyre)",
              f"  lateral grip limit  {h['ay_limit_g']:.3f} g at {h['v_limit_ms']:.2f} m/s",
              f"  understeer gradient {h['Kus_linear_deg_per_g']:+.2f} deg/g (linear range)"]
    L += ["", "=" * 72]
    return "\n".join(L)
