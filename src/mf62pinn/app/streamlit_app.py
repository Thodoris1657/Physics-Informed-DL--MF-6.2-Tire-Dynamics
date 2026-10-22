"""
MF6.2 PINN - graphical tool.

    mf62pinn app            # or:  streamlit run src/mf62pinn/app/streamlit_app.py

Drop in TTC sweeps (and optionally skidpad telemetry), check the vehicle
parameters, press *Run identification*, and download the MF6.2 ``.tir``
files, the HTML report and every table.
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
import zipfile
from pathlib import Path

try:
    import mf62pinn  # noqa: F401
except ImportError:  # running from a source checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from mf62pinn import __version__
from mf62pinn.config import RunConfig
from mf62pinn.data import expand_files, load_telemetry, load_ttc
from mf62pinn.pipeline import run_pipeline
from mf62pinn.tyre import mf62_fy

st.set_page_config(page_title="MF6.2 PINN", page_icon="🏁", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_TTC = REPO_ROOT / "data" / "TTC data"
SAMPLE_TEL = REPO_ROOT / "data" / "Testing_Data"

PRESETS = {
    "Quick (≈30 s)": dict(n_starts=6, adam=150, lbfgs=200, t_adam=150, t_lbfgs=250, loro=False),
    "Standard (≈2 min)": dict(n_starts=24, adam=300, lbfgs=400, t_adam=400, t_lbfgs=600, loro=True),
    "Thorough (≈5 min)": dict(n_starts=64, adam=500, lbfgs=800, t_adam=600, t_lbfgs=1000, loro=True),
}

VEHICLE_FIELDS = [
    ("Mass & inertia", [("NSM", "unsprung mass [kg]"), ("SM", "sprung mass incl. driver [kg]"),
                        ("Iz", "yaw inertia [kg m²]"), ("hcog_SM", "sprung CoG height [m]"),
                        ("hcog_NSM", "unsprung CoG height [m]")]),
    ("Geometry", [("WB", "wheelbase [m]"), ("WD_tot", "front weight distribution [-]"),
                  ("WD_NSM", "unsprung front distribution [-]"), ("Tf", "front track [m]"),
                  ("Tr", "rear track [m]"), ("Rw", "unloaded tyre radius [m]")]),
    ("Suspension", [("hRC_f", "front roll centre [m]"), ("hRC_r", "rear roll centre [m]"),
                    ("roll_rate_deg_per_g", "roll gradient [deg/g]"), ("MB", "roll-stiffness balance, front [-]")]),
    ("Aero & steering", [("clA_nom", "Cl·A [m²]"), ("CdA", "Cd·A [m²]"), ("AD_long", "aero balance, front [-]"),
                         ("roll_loss", "ClA loss per deg roll [m²/deg]"), ("steer_ratio", "steering ratio [-]"),
                         ("vy_sensor_x", "Vy sensor ahead of CoG [m]")]),
]


def _save_uploads(files, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files or []:
        p = folder / Path(f.name).name
        p.write_bytes(f.getbuffer())
        paths.append(str(p))
    return paths


def _zip_dir(d: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(d))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Sidebar - inputs
# --------------------------------------------------------------------------- #
st.sidebar.title("MF6.2 PINN")
st.sidebar.caption(f"v{__version__} · physics-informed tyre identification")

if "work" not in st.session_state:
    st.session_state.work = Path(tempfile.mkdtemp(prefix="mf62pinn_"))
work: Path = st.session_state.work

cfg_upload = st.sidebar.file_uploader("Load a run configuration (.yaml, optional)", type=["yaml", "yml"])
base_cfg = RunConfig()
if cfg_upload is not None:
    p = work / "uploaded_config.yaml"
    p.write_bytes(cfg_upload.getbuffer())
    try:
        base_cfg = RunConfig.from_yaml(p)
        st.sidebar.success("configuration loaded")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"could not read config: {exc}")

st.sidebar.header("1 · Data")
has_sample = SAMPLE_TTC.exists()
source = st.sidebar.radio("Source", ["Upload my files", "P26 sample data"] if has_sample else ["Upload my files"],
                          index=1 if has_sample else 0)
ttc_paths, tel_paths = [], []
if source == "P26 sample data":
    ttc_paths = expand_files([str(SAMPLE_TTC / "*.csv")])
    tel_paths = expand_files([str(SAMPLE_TEL / "*.csv")])
    st.sidebar.caption(f"{len(ttc_paths)} TTC sweeps · {len(tel_paths)} skidpad runs")
else:
    up_ttc = st.sidebar.file_uploader("TTC cornering sweeps (.csv / .dat / .mat)", accept_multiple_files=True,
                                      type=["csv", "dat", "mat", "txt"])
    up_tel = st.sidebar.file_uploader("Skidpad telemetry (optional)", accept_multiple_files=True,
                                      type=["csv", "txt", "xlsx", "parquet"])
    ttc_paths = _save_uploads(up_ttc, work / "ttc")
    tel_paths = _save_uploads(up_tel, work / "telemetry")

with st.sidebar.expander("Telemetry channels & units"):
    cols = dict(base_cfg.telemetry.columns)
    for k, label in (("time", "time [s]"), ("steer", "steering"), ("ax", "long. accel"), ("ay", "lat. accel"),
                     ("vx", "long. velocity"), ("vy", "lat. velocity"), ("yaw_rate", "yaw rate")):
        cols[k] = st.text_input(label, value=cols.get(k, k), key=f"col_{k}")
    steer_unit = st.selectbox("steering unit", ["deg", "rad"], index=["deg", "rad"].index(base_cfg.telemetry.steer_unit))
    steer_rw = st.checkbox("steering is already road-wheel angle", value=base_cfg.telemetry.steer_is_roadwheel)
    yaw_unit = st.selectbox("yaw-rate unit", ["rad/s", "deg/s"], index=["rad/s", "deg/s"].index(base_cfg.telemetry.yaw_rate_unit))
    acc_unit = st.selectbox("acceleration unit", ["m/s2", "g"], index=["m/s2", "g"].index(base_cfg.telemetry.accel_unit))
    spd_unit = st.selectbox("speed unit", ["m/s", "km/h"], index=["m/s", "km/h"].index(base_cfg.telemetry.speed_unit))
    steer_policy = st.selectbox("runs whose steering sign contradicts the turn", ["exclude", "flip", "keep"],
                                index=["exclude", "flip", "keep"].index(base_cfg.telemetry.steer_sign_policy))

st.sidebar.header("2 · Vehicle")
veh_vals = {}
for group, fields in VEHICLE_FIELDS:
    with st.sidebar.expander(group):
        for name, label in fields:
            veh_vals[name] = st.number_input(label, value=float(getattr(base_cfg.vehicle, name)), format="%.5g",
                                             key=f"veh_{name}")
veh_name = st.sidebar.text_input("vehicle name", base_cfg.vehicle.name)
tyre_name = st.sidebar.text_input("tyre name", base_cfg.tyre_name)

st.sidebar.header("3 · Settings")
preset = st.sidebar.selectbox("Effort", list(PRESETS), index=1)
with st.sidebar.expander("Track calibration (advanced)"):
    tc = base_cfg.track_fit
    do_track = st.checkbox("calibrate on telemetry", value=tc.enabled)
    per_axle = st.checkbox("separate front / rear friction scale", value=tc.per_axle)
    learn_k = st.checkbox("also learn stiffness scale LKY", value=tc.learn_stiffness_scale)
    coupling = st.selectbox("rig-shape coupling", ["fixed", "laplace", "joint"][:2],
                            index=0 if tc.shape_coupling == "fixed" else 1)
    s_g = st.slider("sideslip prior: global σ [deg]", 0.25, 6.0, float(tc.sigma_beta_global_deg), 0.25)
    s_r = st.slider("sideslip prior: per-run σ [deg]", 0.25, 6.0, float(tc.sigma_beta_run_deg), 0.25)
    s_s = st.slider("sideslip prior: per-segment σ [deg]", 0.25, 6.0, float(tc.sigma_beta_seg_deg), 0.25)
    win = st.slider("segment window [s]", 0.5, 3.0, float(base_cfg.telemetry.window_s), 0.25)


def build_config() -> RunConfig:
    cfg = base_cfg.copy()
    cfg.ttc.files = ttc_paths
    cfg.telemetry.files = tel_paths
    cfg.telemetry.columns = cols
    cfg.telemetry.steer_unit, cfg.telemetry.steer_is_roadwheel = steer_unit, steer_rw
    cfg.telemetry.yaw_rate_unit, cfg.telemetry.accel_unit, cfg.telemetry.speed_unit = yaw_unit, acc_unit, spd_unit
    cfg.telemetry.steer_sign_policy = steer_policy
    cfg.telemetry.window_s = win
    for k, v in veh_vals.items():
        setattr(cfg.vehicle, k, float(v))
    cfg.vehicle.name, cfg.tyre_name = veh_name, tyre_name
    pr = PRESETS[preset]
    cfg.ttc_fit.n_starts, cfg.ttc_fit.adam_iters, cfg.ttc_fit.lbfgs_iters = pr["n_starts"], pr["adam"], pr["lbfgs"]
    cfg.track_fit.adam_iters, cfg.track_fit.lbfgs_iters = pr["t_adam"], pr["t_lbfgs"]
    cfg.validation.leave_one_run_out = pr["loro"]
    cfg.track_fit.enabled = do_track
    cfg.track_fit.per_axle, cfg.track_fit.learn_stiffness_scale = per_axle, learn_k
    cfg.track_fit.shape_coupling = coupling
    cfg.track_fit.sigma_beta_global_deg, cfg.track_fit.sigma_beta_run_deg = s_g, s_r
    cfg.track_fit.sigma_beta_seg_deg = s_s
    cfg.output.out_dir = str(work / f"run_{time.strftime('%H%M%S')}")
    return cfg


# --------------------------------------------------------------------------- #
#  Main panel
# --------------------------------------------------------------------------- #
st.title("Magic Formula 6.2 · lateral tyre identification")
st.caption("Rig sweeps → robust multi-start MF6.2 fit with uncertainty → errors-in-variables track calibration "
           "→ validation → `.tir` export")

c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("TTC files", len(ttc_paths))
c2.metric("telemetry runs", len(tel_paths))
with c3:
    if st.button("Preview & audit data", disabled=not ttc_paths):
        cfg = build_config()
        try:
            ttc = load_ttc(ttc_paths, cfg.ttc)
            st.dataframe(ttc.summary(), hide_index=True, use_container_width=True)
            for n in ttc.notes:
                st.warning(f"{n['file']}: {n['msg']}") if n["level"] != "info" else st.info(f"{n['file']}: {n['msg']}")
            if tel_paths:
                tel = load_telemetry(tel_paths, cfg.telemetry, cfg.vehicle)
                st.dataframe(tel.runs[["run", "used", "reason", "segments", "radius_m", "ay_med_g",
                                       "steer_rw_med_deg", "ackermann_med_deg", "Kus_deg_per_g",
                                       "beta_meas_std_deg"]].round(3), hide_index=True, use_container_width=True)
                for n in tel.notes:
                    st.warning(f"{n['file']}: {n['msg']}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"data problem: {exc}")

run = st.button("▶  Run identification", type="primary", disabled=not ttc_paths, use_container_width=True)
if not ttc_paths:
    st.info("Add at least one TTC cornering sweep (Calspan .csv/.dat/.mat with SA, FY, FZ columns) in the sidebar.")

if run:
    cfg = build_config()
    bar = st.progress(0.0, text="starting…")
    log = st.empty()

    def progress(msg: str, frac: float) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    t0 = time.time()
    try:
        res = run_pipeline(cfg, progress=progress, verbose=False)
        st.session_state.res = res
        bar.progress(1.0, text=f"done in {time.time() - t0:.0f} s")
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.exception(exc)

res = st.session_state.get("res")
if res is not None:
    F = res.files
    hl = res.headline
    tabs = st.tabs(["Overview", "Rig fit", "Track calibration", "Vehicle", "Data audit", "Tyre explorer", "Downloads"])

    with tabs[0]:
        k = st.columns(4)
        k[0].metric("TTC fit R²", f"{res.ttc_fit.metrics['r2']:.4f}",
                    help=f"RMSE {res.ttc_fit.metrics['rmse_N']:.1f} N over {res.ttc_fit.metrics['n_rows']} rows")
        k[1].metric("rig peak μ @ Fz0", f"{hl.get('rig_mu_peak_Fz0', float('nan')):.2f}")
        k[2].metric("rig peak slip @ Fz0", f"{hl.get('rig_alpha_peak_Fz0_deg', float('nan')):.1f}°")
        k[3].metric("K_yα @ Fz0", f"{hl.get('rig_Kya_Fz0_N_per_deg', float('nan')):.0f} N/°")
        if res.track is not None:
            k = st.columns(4)
            lam, sd = res.track.lambdas, res.track.lambdas_std
            k[0].metric("LMUY front (track / rig)", f"{lam['LMUY_front']:.3f} ± {sd['LMUY_front']:.3f}")
            k[1].metric("LMUY rear (track / rig)", f"{lam['LMUY_rear']:.3f} ± {sd['LMUY_rear']:.3f}")
            k[2].metric("track peak μ @ Fz0", f"{hl.get('track_mu_peak_Fz0_front', float('nan')):.2f}")
            if res.handling_track:
                k[3].metric(f"grip limit @ R = {res.handling_track['radius_m']:.1f} m",
                            f"{res.handling_track['ay_limit_g']:.2f} g")
        st.subheader("Findings")
        icon = {"good": "✅", "info": "ℹ️", "warn": "⚠️", "error": "⛔"}
        for f in res.findings:
            st.markdown(f"{icon.get(f['level'], '•')} **{f['area']}** — {f['msg']}")

    def _imgs(keys):
        for key in keys:
            if key in F:
                st.image(F[key], use_container_width=True)

    with tabs[1]:
        _imgs(["fig_01_ttc_fit", "fig_02_tyre_curves", "fig_03_characteristics"])
        st.subheader("Coefficients")
        st.dataframe(res.coef_table, hide_index=True, use_container_width=True)
        _imgs(["fig_04_identifiability", "fig_05_multistart"])
    with tabs[2]:
        if res.track is None:
            st.info("No telemetry calibration in this run.")
        else:
            _imgs(["fig_06_balance", "fig_07_operating_points", "fig_08_sideslip"])
            st.dataframe(res.track.beta_offsets, hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame(res.track.starts), hide_index=True, use_container_width=True)
    with tabs[3]:
        _imgs(["fig_09_handling", "fig_10_timeseries"])
        if res.grip_by_run is not None:
            st.dataframe(res.grip_by_run, hide_index=True, use_container_width=True)
    with tabs[4]:
        _imgs(["fig_11_data_audit"])
        if res.tel is not None:
            st.dataframe(res.tel.runs, hide_index=True, use_container_width=True)
        st.dataframe(res.ttc.summary(), hide_index=True, use_container_width=True)
    with tabs[5]:
        st.caption("Evaluate the identified tyre interactively (internal convention: α > 0 → Fy > 0).")
        e1, e2, e3 = st.columns(3)
        loads = e1.multiselect("loads [N]", [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400],
                               default=[300, 700, 1100])
        default_mu = float(res.track.lambdas["LMUY_front"]) if res.track is not None else 1.0
        lmu = e2.slider("LMUY (friction scale)", 0.2, 1.5, default_mu, 0.01)
        lky = e3.slider("LKY (stiffness scale)", 0.3, 1.8, 1.0, 0.01)
        a = np.linspace(-15, 15, 241)
        tyre = res.ttc_fit.model
        data = {}
        with torch.no_grad():
            th = tyre.theta()
            for Fz in loads:
                at = torch.tensor(np.radians(a))
                data[f"{Fz} N"] = mf62_fy(th, at, torch.full_like(at, float(Fz)), tyre.Fz0, lmuy=lmu, lky=lky).numpy()
        if data:
            st.line_chart(pd.DataFrame(data, index=pd.Index(a, name="slip angle [deg]")), y_label="Fy [N]")
    with tabs[6]:
        out_dir = Path(res.out_dir)
        st.download_button("⬇ everything (.zip)", _zip_dir(out_dir), file_name=f"{out_dir.name}.zip",
                           mime="application/zip", type="primary")
        cols_dl = st.columns(3)
        i = 0
        for key in ("report", "tir_rig", "tir_track_front", "tir_track_rear", "coefficients_json",
                    "coefficients_csv", "timeseries", "summary", "config"):
            if key in F:
                pth = Path(F[key])
                cols_dl[i % 3].download_button(pth.name, pth.read_bytes(), file_name=pth.name, key=f"dl_{key}")
                i += 1
        st.caption(f"Files are also on disk in `{out_dir}`")
