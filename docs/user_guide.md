# User guide

## 1. Install

```bash
# Python 3.10 – 3.13
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -e ".[app]"          # package + graphical tool
```

PyTorch is installed from PyPI (CPU build). A GPU is not needed: a full run on
the P26 data takes about two minutes on a laptop CPU.

Windows shortcuts in the repository root: **`launch_app.bat`** (opens the
graphical tool) and **`run_p26.bat`** (runs the P26 configuration from the
command line).

## 2. The graphical tool

```bash
mf62pinn app
```

1. **Data** – choose *P26 sample data* or upload your own TTC sweeps (`.csv`,
   `.dat`, `.mat`) and, optionally, telemetry runs. If your telemetry uses other
   column names or units, set them under *Telemetry channels & units*.
2. **Vehicle** – edit the mass, geometry, suspension, aero and steering values,
   or load a whole configuration from a `.yaml` file.
3. **Settings** – *Quick / Standard / Thorough* effort, and the track-calibration
   options (sideslip priors, separate front/rear grip, stiffness scale).
4. **Preview & audit data** shows what will be used and why runs are rejected.
5. **Run identification** – results appear in tabs: Overview (KPIs and findings),
   Rig fit, Track calibration, Vehicle, Data audit, an interactive **Tyre
   explorer**, and **Downloads** (`.tir` files, report, every table, or a zip).

## 3. Command line

```bash
mf62pinn fit -c configs/p26.yaml                      # full run, output in runs/p26
mf62pinn fit --ttc "my_ttc/*.dat" --no-track --quick   # rig only, fast
mf62pinn fit -c configs/p26.yaml --set track_fit.sigma_beta_run_deg=1.0 validation.leave_one_run_out=false
mf62pinn audit -c configs/p26.yaml                    # data checks only
mf62pinn predict --tir runs/p26/tyre_track_front.tir --fz 300,700,1100
mf62pinn predict --tir runs/p26/tyre_rig_TTC.tir --input my_points.csv   # columns alpha_deg, Fz_N
mf62pinn init my_car.yaml                             # editable template with every option
```

`--set` accepts any dotted key of the configuration, so nothing requires editing
code.

## 4. Python API

```python
from mf62pinn import RunConfig, run_pipeline

cfg = RunConfig.from_yaml("configs/p26.yaml")
cfg.output.out_dir = "runs/experiment_1"
res = run_pipeline(cfg)

print(res.headline)                    # key numbers
res.ttc_fit.values                     # MF6.2 coefficients (internal convention)
res.ttc_fit.std                        # 1-sd (Laplace, AR(1)-tempered)
res.track.lambdas                      # LMUY front / rear, LKY
res.track.segments                     # per-segment latent sideslip, slip, Fz, Fy
res.handling_track["curve"]            # steady-state handling diagram
```

Lower-level building blocks:

```python
import torch
from mf62pinn.export.tir import read_tir, tir_fy0
t = read_tir("runs/p26/tyre_track_front.tir")
Fy = tir_fy0(t, alpha=np.radians([2, 4, 6]), Fz=np.array([700, 700, 700]))   # ISO-W
```

## 5. Outputs

| file | content |
|---|---|
| `report.html` | self-contained report: findings, all figures, all tables |
| `tyre_rig_TTC.tir` | MF6.2 property file of the rig tyre (ISO-W) with ±σ comments |
| `tyre_track_front.tir`, `tyre_track_rear.tir` | rig shape with the identified `LMUY` (and `LKY`) |
| `mf62_coefficients.json / .csv` | coefficients, ±σ, identifiability status |
| `characteristics_rig.csv`, `characteristics_track.csv` | μ_peak, α_peak, Kyα, B, C, D, E per load with ±σ |
| `ttc_fit_per_load.csv`, `ttc_multistart.csv`, `ttc_posterior_correlation.csv` | rig-fit diagnostics |
| `segments.csv` | every quasi-steady segment: inputs, latent β, per-wheel α/Fz/Fy, residuals |
| `sideslip_offsets.csv` | per-run sensor correction ±σ |
| `validation_leave_one_run_out.csv`, `validation_grip_limit.csv`, `handling_diagram.csv` | validation |
| `per_wheel_timeseries.csv` | per-sample wheel loads, slip angles and forces (calibrated kinematics) |
| `telemetry_audit.csv` | per-run checks |
| `figures/*.png` | the twelve report figures |
| `metrics.json`, `SUMMARY.txt`, `config_used.yaml`, `model.pt` | machine-readable summary, text summary, exact configuration, model state |

## 6. Adapting to another car or tyre

1. `mf62pinn init my_car.yaml` and edit the `vehicle:` block.
2. Point `ttc.files` and `telemetry.files` at your data; set `telemetry.columns`
   and units (see [data_format.md](data_format.md)).
3. `mf62pinn audit -c my_car.yaml` – fix anything flagged.
4. `mf62pinn fit -c my_car.yaml`.

## 7. Reading the findings

* **good** – a quantity is identified and the evidence for it.
* **info** – what was frozen, rejected or held out.
* **warn** – the data do not support a quantity (e.g. on-track slip angles when the
  car was driven at the limit) or a sensor looks suspect. Use those numbers as
  prior-dominated.
* **error** – a data file is unusable for the intended purpose (e.g. a
  drive/brake test supplied as a cornering sweep).
