"""Minimal Python-API example: run the P26 identification and use the result."""
import numpy as np

from mf62pinn import RunConfig, run_pipeline
from mf62pinn.export.tir import read_tir, tir_fy0

cfg = RunConfig.from_yaml("configs/p26.yaml")
cfg.output.out_dir = "runs/quickstart"
cfg.validation.leave_one_run_out = False          # faster
res = run_pipeline(cfg)

print("\nheadline:")
for k, v in res.headline.items():
    print(f"  {k:28s} {v:.4f}")

# evaluate the exported track tyre with the independent ISO-W evaluator
t = read_tir(res.files["tir_track_front"])
alpha = np.radians(np.arange(-12, 13, 3))
print("\nFy_ISO [N] at 700 N:", np.round(tir_fy0(t, alpha, np.full_like(alpha, 700.0))))
