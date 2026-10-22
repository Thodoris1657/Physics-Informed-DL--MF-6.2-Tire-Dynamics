# Changelog

## 2.0.0 — 2026-10-22

Complete rewrite as an installable package (`mf62pinn`) with a CLI, a
graphical tool and a test-suite. The physics of v1 (MF6.2 lateral, the
LoadTransfer_v2 port, rigid-body slip kinematics, steady-state balances) is
kept; the estimator around it is new.

### Rig identification
- Robust Student-t likelihood in friction space, equal weight per load, learned noise scale.
- Batched Sobol multi-start (Adam in parallel) + full-batch L-BFGS polish; every end point kept.
- Laplace posterior with AR(1) variance inflation for the 100 Hz serial correlation;
  ±σ on every coefficient, derived quantity and curve (delta method).
- Smooth constrained parameterisation (sigmoid boxes, smooth `Ey ≤ 1`, smooth `Fz` floor):
  no clamps, no zero-gradient attractors.
- Camber terms frozen automatically when the data contain no camber.
- **Result on P26 data: RMSE 41.6 N, R² 0.9993** (v1: 94.9 N / 0.9964 on the same files).

### Track calibration
- Quasi-steady 1-s segments instead of 100 Hz samples.
- Hierarchical errors-in-variables model: latent sideslip per segment with global/run/segment
  priors replaces the single `vy_offset`.
- MF-native scaling: `LMUY` per axle (friction only), optional `LKY`.
- Rig shape fixed or coupled through its Laplace posterior.
- Friction-ellipse de-rating from the rear drive force.
- Multi-start over sideslip basins, Laplace over all free quantities, saturation diagnostic.
- Leave-one-run-out validation, steady-state grip-limit and handling-diagram solver.

### Fixes of defects found in the v1 audit
- Steer-induced yaw moment sign (`½Tf (F_FR − F_FL) sin δ`); one wheel-geometry definition shared by every block.
- TTC SAE conversion `α = −SA, Fy = +FY` (v1 mirrored the asymmetric terms relative to the car).
- Best-state logic no longer compares mini-batch with full-batch losses (all stages are full batch).
- Free-rolling guard against drive/brake files archived as cornering sweeps.
- Steering-sign audit: runs whose steering contradicts the turn direction are quarantined (configurable).
- "NN" labels removed: there is no neural network; the outputs say "MF6.2".

### Outputs
- ISO-W MF6.2 `.tir` export (rig, track-front, track-rear) verified against an independent evaluator.
- Self-contained HTML report, twelve figures, CSV/JSON tables, `metrics.json`, `config_used.yaml`.

## 1.x — 2026 (legacy)
`legacy/main.py`: single-script PINN, Adam + L-BFGS, documented in
`docs/PINN_Lateral_MF62_Identification.pdf`.
