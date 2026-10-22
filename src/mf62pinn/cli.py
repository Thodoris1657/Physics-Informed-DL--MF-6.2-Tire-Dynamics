"""
Command-line interface.

    mf62pinn fit     --config configs/p26.yaml            # full identification
    mf62pinn fit     --ttc "data/TTC data/*.csv" --tel "data/Testing_Data/*.csv" --out runs/today
    mf62pinn audit   --config configs/p26.yaml            # data checks only, no fitting
    mf62pinn predict --tir runs/latest/tyre_track_front.tir --input points.csv
    mf62pinn init    my_car.yaml                          # write an editable config template
    mf62pinn app                                          # graphical tool in the browser
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import RunConfig, set_by_path


def _load_cfg(args) -> RunConfig:
    cfg = RunConfig.from_yaml(args.config) if getattr(args, "config", None) else RunConfig()
    if getattr(args, "ttc", None):
        cfg.ttc.files = list(args.ttc)
    if getattr(args, "tel", None):
        cfg.telemetry.files = list(args.tel)
    if getattr(args, "no_track", False):
        cfg.track_fit.enabled = False
    if getattr(args, "out", None):
        cfg.output.out_dir = args.out
    for kv in getattr(args, "set", None) or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got '{kv}'")
        k, v = kv.split("=", 1)
        set_by_path(cfg, k.strip(), v.strip())
    if getattr(args, "quick", False):
        cfg.ttc_fit.n_starts = 6
        cfg.ttc_fit.adam_iters = 150
        cfg.ttc_fit.lbfgs_iters = 200
        cfg.track_fit.adam_iters = 150
        cfg.track_fit.lbfgs_iters = 250
        cfg.validation.leave_one_run_out = False
    return cfg


def cmd_fit(args) -> int:
    from .pipeline import run_pipeline
    cfg = _load_cfg(args)
    if cfg.output.out_dir in ("runs/latest", "") and not args.out:
        cfg.output.out_dir = f"runs/{time.strftime('%Y%m%d_%H%M%S')}"
    res = run_pipeline(cfg, base_dir=args.base_dir, make_plots=not args.no_plots,
                       make_report=not args.no_report)
    print()
    print((res.out_dir / "SUMMARY.txt").read_text(encoding="utf-8"))
    if "report" in res.files:
        print(f"\nReport: {res.files['report']}")
    return 0


def cmd_audit(args) -> int:
    from .data import expand_files, load_telemetry, load_ttc
    cfg = _load_cfg(args)
    out = Path(args.out or "runs/audit")
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    ttc_files = expand_files(cfg.ttc.files, args.base_dir)
    if ttc_files:
        ttc = load_ttc(ttc_files, cfg.ttc)
        print("\nTTC files\n" + ttc.summary().to_string(index=False))
        for n in ttc.notes:
            print(f"  [{n['level']}] {n['file']}: {n['msg']}")
        ttc.summary().to_csv(out / "ttc_audit.csv", index=False)
    tel_files = expand_files(cfg.telemetry.files, args.base_dir)
    if tel_files:
        tel = load_telemetry(tel_files, cfg.telemetry, cfg.vehicle)
        cols = ["run", "used", "reason", "segments", "radius_m", "ay_med_g", "steer_rw_med_deg",
                "ackermann_med_deg", "Kus_deg_per_g", "beta_meas_std_deg", "kin_resid_rms"]
        print("\nTelemetry runs\n" + tel.runs[cols].round(3).to_string(index=False))
        for n in tel.notes:
            print(f"  [{n['level']}] {n['file']}: {n['msg']}")
        tel.runs.to_csv(out / "telemetry_audit.csv", index=False)
        tel.segments.to_csv(out / "segments.csv", index=False)
        print(f"\n{len(tel.segments)} quasi-steady segments retained")
    print(f"\nAudit tables written to {out}")
    return 0


def cmd_predict(args) -> int:
    import numpy as np
    import pandas as pd

    from .export.tir import read_tir, tir_fy0
    t = read_tir(args.tir)
    if args.input:
        df = pd.read_csv(args.input)
        a_col = next((c for c in df.columns if c.lower() in ("alpha_deg", "alpha", "sa", "slip_angle_deg")), None)
        f_col = next((c for c in df.columns if c.lower() in ("fz_n", "fz", "load", "load_n")), None)
        if a_col is None or f_col is None:
            raise SystemExit("input CSV needs columns alpha_deg and Fz_N")
        alpha_deg = df[a_col].to_numpy(float)
        Fz = np.abs(df[f_col].to_numpy(float))
    else:
        alpha_deg = np.linspace(args.alpha_min, args.alpha_max, args.n)
        loads = [float(x) for x in args.fz.split(",")]
        df = pd.DataFrame([(a, f) for f in loads for a in alpha_deg], columns=["alpha_deg", "Fz_N"])
        alpha_deg, Fz = df["alpha_deg"].to_numpy(), df["Fz_N"].to_numpy()
    df["Fy_ISO_N"] = tir_fy0(t, np.radians(alpha_deg), Fz)
    df["mu_y"] = np.abs(df["Fy_ISO_N"]) / Fz
    out = args.output or "predictions.csv"
    df.to_csv(out, index=False)
    print(df.head(12).to_string(index=False))
    print(f"... {len(df)} rows -> {out}  (ISO-W sign convention: alpha > 0 -> Fy < 0)")
    return 0


def cmd_init(args) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (use --force)")
    RunConfig().to_yaml(path)
    print(f"Wrote template config to {path}. Edit the vehicle block and the data globs, then:\n"
          f"  mf62pinn fit --config {path}")
    return 0


def cmd_app(args) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("The app needs Streamlit:  pip install \"mf62pinn[app]\"  (or pip install streamlit)")
    import subprocess
    app = Path(__file__).with_name("app") / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--browser.gatherUsageStats", "false", "--theme.primaryColor", "#2a78d6",
           "--theme.base", "light"]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mf62pinn", description="Physics-informed MF6.2 lateral tyre identification")
    p.add_argument("--version", action="version", version=f"mf62pinn {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--config", "-c", help="YAML run configuration")
        sp.add_argument("--ttc", nargs="+", help="TTC files or globs (overrides config)")
        sp.add_argument("--tel", nargs="+", help="telemetry files or globs (overrides config)")
        sp.add_argument("--out", "-o", help="output directory")
        sp.add_argument("--base-dir", default=".", help="directory relative globs are resolved against")
        sp.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="override any config entry, e.g. track_fit.sigma_beta_run_deg=1.0")

    f = sub.add_parser("fit", help="run the full identification pipeline")
    data_args(f)
    f.add_argument("--quick", action="store_true", help="fewer starts/iterations, no cross-validation")
    f.add_argument("--no-track", action="store_true", help="rig (TTC) identification only")
    f.add_argument("--no-plots", action="store_true")
    f.add_argument("--no-report", action="store_true")
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("audit", help="data checks only (no fitting)")
    data_args(a)
    a.set_defaults(func=cmd_audit)

    pr = sub.add_parser("predict", help="evaluate a .tir file at given slip angles and loads")
    pr.add_argument("--tir", required=True)
    pr.add_argument("--input", help="CSV with alpha_deg and Fz_N columns")
    pr.add_argument("--fz", default="300,600,900,1200", help="loads [N] for a grid (without --input)")
    pr.add_argument("--alpha-min", type=float, default=-15.0)
    pr.add_argument("--alpha-max", type=float, default=15.0)
    pr.add_argument("--n", type=int, default=61)
    pr.add_argument("--output", "-o")
    pr.set_defaults(func=cmd_predict)

    i = sub.add_parser("init", help="write an editable configuration template")
    i.add_argument("path", nargs="?", default="my_config.yaml")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)

    ap = sub.add_parser("app", help="launch the graphical tool (Streamlit)")
    ap.add_argument("--port", type=int, default=8501)
    ap.set_defaults(func=cmd_app)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
