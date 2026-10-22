"""Self-contained HTML report (figures embedded, no external assets)."""

from __future__ import annotations

import base64
import html
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

CSS = """
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--good:#006300;--warn:#a86f00;--bad:#d03b3b;--info:#2a78d6;--chip:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--good:#0ca30c;--warn:#fab219;--bad:#e66767;
--info:#3987e5;--chip:#262624}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
header h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}header p{margin:0;color:var(--ink2)}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0 8px}nav a{font-size:13px;color:var(--ink2);
text-decoration:none;padding:4px 10px;border:1px solid var(--line);border-radius:999px}nav a:hover{color:var(--accent)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px 24px;margin:18px 0}
h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:18px 0 8px;color:var(--ink2)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:600}.kpi .l{font-size:12.5px;color:var(--ink2)}.kpi .s{font-size:12px;color:var(--muted)}
.finding{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px}
.finding:last-child{border-bottom:0}.tag{flex:0 0 auto;font-size:11px;font-weight:700;padding:2px 8px;
border-radius:999px;background:var(--chip);height:fit-content;text-transform:uppercase;letter-spacing:.03em}
.tag.good{color:var(--good)}.tag.warn{color:var(--warn)}.tag.error{color:var(--bad)}.tag.info{color:var(--info)}
.area{color:var(--ink2);font-weight:600;margin-right:4px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600}td:first-child,th:first-child{text-align:left}
.tw{overflow-x:auto}figure{margin:14px 0}figure img{width:100%;height:auto;border-radius:8px;
border:1px solid var(--line);background:#fcfcfb}figcaption{font-size:13px;color:var(--ink2);margin-top:6px}
.note{font-size:13.5px;color:var(--ink2)}code{font-size:12.5px;background:var(--chip);padding:1px 5px;border-radius:4px}
.files{columns:2;font-size:13px}@media(max-width:700px){.files{columns:1}.kpi .v{font-size:20px}}
footer{color:var(--muted);font-size:12.5px;margin-top:30px}
"""


def _img(path: str | None, caption: str) -> str:
    if not path or not Path(path).exists():
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}">'
            f"<figcaption>{caption}</figcaption></figure>")


def _table(df: pd.DataFrame, fmt: dict | None = None, max_rows: int = 200) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    h = ["<div class='tw'><table><thead><tr>"] + [f"<th>{html.escape(str(c))}</th>" for c in cols] + ["</tr></thead><tbody>"]
    for _, r in df.head(max_rows).iterrows():
        h.append("<tr>")
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                s = "–" if not np.isfinite(v) else (fmt.get(c, "{:.4g}").format(v))
            else:
                s = html.escape(str(v))
            h.append(f"<td>{s}</td>")
        h.append("</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def _kpi(value: str, label: str, sub: str = "") -> str:
    return f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{label}</div><div class='s'>{sub}</div></div>"


def write_report(res) -> Path:
    cfg, tf = res.cfg, res.ttc_fit
    F = res.files
    hl = res.headline
    kpis = [_kpi(f"{tf.metrics['r2']:.4f}", "TTC fit R²", f"RMSE {tf.metrics['rmse_N']:.1f} N"),
            _kpi(f"{hl.get('rig_mu_peak_Fz0', float('nan')):.2f}", "rig peak μ at Fz0",
                 f"Fz0 = {tf.Fz0:.0f} N"),
            _kpi(f"{hl.get('rig_alpha_peak_Fz0_deg', float('nan')):.1f}°", "rig peak slip at Fz0", ""),
            _kpi(f"{hl.get('rig_Kya_Fz0_N_per_deg', float('nan')):.0f}", "K_yα at Fz0 [N/deg]", "")]
    if res.track is not None:
        lam, sd = res.track.lambdas, res.track.lambdas_std
        kpis += [_kpi(f"{lam['LMUY_front']:.3f}", "LMUY front (track/rig)", f"± {sd['LMUY_front']:.3f}"),
                 _kpi(f"{lam['LMUY_rear']:.3f}", "LMUY rear (track/rig)", f"± {sd['LMUY_rear']:.3f}"),
                 _kpi(f"{hl.get('track_mu_peak_Fz0_front', float('nan')):.2f}", "track peak μ at Fz0", "front axle")]
    if res.handling_track is not None:
        kpis.append(_kpi(f"{res.handling_track['ay_limit_g']:.2f} g", "steady-state grip limit",
                         f"R = {res.handling_track['radius_m']:.2f} m"))

    findings = "".join(
        f"<div class='finding'><span class='tag {f['level']}'>{f['level']}</span><div><span class='area'>{html.escape(f['area'])}</span>"
        f"{html.escape(f['msg'])}</div></div>" for f in res.findings)

    coef = res.coef_table.copy()
    coef["tir (ISO-W)"] = [(-v if p in ("pKy1", "pVy1", "pVy2", "pVy3", "pVy4") else v)
                           for p, v in zip(coef["param"], coef["value"])]
    coef = coef[["param", "value", "sd", "rel_sd_%", "tir (ISO-W)", "status"]]
    rig = res.char_rig[res.char_rig["side"] == "+"][["Fz_N", "mu_peak", "mu_peak_sd", "alpha_peak_deg",
                                                    "alpha_peak_deg_sd", "Kya_N_per_deg", "Kya_N_per_deg_sd",
                                                    "B", "C", "D", "E"]]

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>MF6.2 identification report</title>"
             f"<style>{CSS}</style></head><body><main>",
             f"<header><h1>MF6.2 lateral identification · {html.escape(cfg.tyre_name)}</h1>"
             f"<p>{html.escape(cfg.vehicle.name)} · generated {time.strftime('%Y-%m-%d %H:%M')} · mf62pinn v{__version__}</p></header>",
             "<nav><a href='#summary'>Summary</a><a href='#rig'>Rig fit</a><a href='#uq'>Uncertainty</a>"
             + ("<a href='#track'>Track calibration</a><a href='#vehicle'>Vehicle</a><a href='#data'>Data audit</a>" if res.track is not None else "")
             + "<a href='#files'>Files</a></nav>",
             f"<section id='summary'><h2>Summary</h2><div class='kpis'>{''.join(kpis)}</div>"
             f"<h3>Findings</h3>{findings}</section>",
             f"<section id='rig'><h2>Stage A · rig identification</h2>"
             f"<p class='note'>Robust (Student-t) likelihood in friction space, equal weight per load, Sobol multi-start "
             f"({tf.metrics['n_starts']} starts) and full-batch L-BFGS. The data term is tempered by the AR(1) variance "
             f"inflation factor {tf.inflation:.1f} (effective sample size {tf.metrics['n_eff']:.0f} of {tf.metrics['n_rows']} rows), "
             f"so the uncertainties below are not over-confident.</p>",
             _img(F.get("fig_01_ttc_fit"), "MF6.2 fit per load with ±2σ band and residuals."),
             "<h3>Coefficients (internal convention: α > 0 → Fy > 0)</h3>",
             _table(coef, {"value": "{:.6g}", "sd": "{:.3g}", "rel_sd_%": "{:.1f}", "tir (ISO-W)": "{:.6g}"}),
             "<h3>Characteristic values (positive side, ±1σ)</h3>",
             _table(rig, {"Fz_N": "{:.0f}", "mu_peak": "{:.3f}", "mu_peak_sd": "{:.3f}", "alpha_peak_deg": "{:.2f}",
                          "alpha_peak_deg_sd": "{:.2f}", "Kya_N_per_deg": "{:.1f}", "Kya_N_per_deg_sd": "{:.1f}",
                          "B": "{:.3f}", "C": "{:.3f}", "D": "{:.0f}", "E": "{:.3f}"}),
             _img(F.get("fig_02_tyre_curves"), "Identified curves; dots mark the positive-side peaks."),
             _img(F.get("fig_03_characteristics"), "Peak friction, peak slip and cornering stiffness vs load."),
             "<h3>Per-load fit</h3>",
             _table(tf.per_source, {"Fz_mean_N": "{:.0f}", "rmse_N": "{:.1f}", "r2": "{:.5f}", "mu_rmse": "{:.4f}"}),
             "</section>",
             "<section id='uq'><h2>Uncertainty and identifiability</h2>",
             _img(F.get("fig_04_identifiability"), "Posterior correlation (Laplace) and Hessian eigen-spectrum."),
             _img(F.get("fig_05_multistart"), "Every multi-start end point: near-optimal starts in blue."),
             _img(F.get("fig_12_convergence"), "Optimiser convergence."),
             "</section>"]
    if res.track is not None:
        t = res.track
        tm = t.metrics
        mt = pd.DataFrame([{"metric": k, "value": v} for k, v in tm.items() if isinstance(v, (int, float))])
        parts += [f"<section id='track'><h2>Stage B · track calibration (errors-in-variables)</h2>"
                  f"<p class='note'>{tm['segments']} quasi-steady {cfg.telemetry.window_s:.1f}-s segments from {tm['runs']} runs. "
                  f"Each segment carries a latent sideslip (global + run + segment terms with measurement priors "
                  f"σ = {cfg.track_fit.sigma_beta_global_deg}°/{cfg.track_fit.sigma_beta_run_deg}°/{cfg.track_fit.sigma_beta_seg_deg}°); "
                  f"grip is scaled the MF6.2 way (LMUY on friction, LKY on stiffness{' – fixed at 1' if not cfg.track_fit.learn_stiffness_scale else ''}).</p>",
                  _img(F.get("fig_06_balance"), "Lateral balance per segment: posterior fit, prediction from measured "
                       "kinematics, and leave-one-run-out run means."),
                  _img(F.get("fig_07_operating_points"), "Where the tyres operated on the calibrated curves."),
                  _img(F.get("fig_08_sideslip"), "Latent vs measured sideslip and per-run sensor corrections."),
                  "<h3>Metrics</h3>", _table(mt, {"value": "{:.4g}"}),
                  "<h3>Per-run sideslip corrections</h3>",
                  _table(t.beta_offsets, {"offset_deg": "{:+.2f}", "offset_sd_deg": "{:.2f}", "global_deg": "{:+.2f}",
                                          "run_deg": "{:+.2f}", "equiv_vy_offset_ms": "{:+.3f}"}),
                  "<h3>Multi-start basins</h3>",
                  _table(pd.DataFrame(t.starts), {"neg_log_post": "{:.2f}", "lmu_f": "{:.3f}", "lmu_r": "{:.3f}",
                                                  "LKY": "{:.3f}", "b_global_deg": "{:+.2f}", "lambda0": "{:.3f}"}),
                  "</section>",
                  "<section id='vehicle'><h2>Vehicle level</h2>",
                  _img(F.get("fig_09_handling"), "Constant-radius handling diagram and steady-state slip angles."),
                  "<h3>Grip limit at each run's radius</h3>",
                  _table(res.grip_by_run, {"radius_m": "{:.2f}", "ay_limit_pred_g": "{:.3f}",
                                           "ay_driven_p90_g": "{:.3f}", "ay_driven_mean_g": "{:.3f}"}) if res.grip_by_run is not None else "",
                  _img(F.get("fig_10_timeseries"), "Per-wheel reconstruction for every used run."),
                  "</section>"]
    if res.tel is not None:
        audit = res.tel.runs[["run", "used", "reason", "segments", "radius_m", "vx_med", "ay_med_g",
                              "steer_rw_med_deg", "ackermann_med_deg", "Kus_deg_per_g", "beta_meas_std_deg",
                              "kin_resid_rms"]]
        parts += ["<section id='data'><h2>Data audit</h2>",
                  _img(F.get("fig_11_data_audit"), "Steering-sign consistency, kinematic identity and sideslip noise per run."),
                  _table(audit, {"radius_m": "{:.2f}", "vx_med": "{:.2f}", "ay_med_g": "{:.3f}", "steer_rw_med_deg": "{:+.2f}",
                                 "ackermann_med_deg": "{:+.2f}", "Kus_deg_per_g": "{:+.2f}", "beta_meas_std_deg": "{:.2f}",
                                 "kin_resid_rms": "{:.2f}"}),
                  "<h3>TTC files</h3>", _table(res.ttc.summary()), "</section>"]
    files = "".join(f"<li><code>{html.escape(Path(p).name)}</code> – {html.escape(k)}</li>"
                    for k, p in sorted(F.items()) if not k.startswith("fig_"))
    parts += [f"<section id='files'><h2>Output files</h2><ul class='files'>{files}</ul>"
              f"<p class='note'>The <code>.tir</code> files are ISO-W MF6.2 property files; only the pure-lateral "
              f"coefficients are identified. Run time {res.timings.get('total_s', float('nan')):.0f} s.</p></section>",
              f"<footer>mf62pinn v{__version__} · reproducible with <code>config_used.yaml</code></footer>",
              "</main></body></html>"]
    path = Path(res.out_dir) / "report.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path
