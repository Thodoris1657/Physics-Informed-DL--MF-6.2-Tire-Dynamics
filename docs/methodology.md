# Methodology

This page documents the v2 algorithm. The v1 algorithm and its audit are in
[`PINN_Lateral_MF62_Identification.pdf`](PINN_Lateral_MF62_Identification.pdf);
section numbers below (§) refer to that report.

## 0. Overview

```mermaid
flowchart LR
    TTC[TTC cornering sweeps<br/>SA, FY, FZ] --> A
    TEL[Skidpad telemetry<br/>δ, ax, ay, Vx, Vy, r] --> AUD[Audit<br/>steer sign, kinematic identity,<br/>free-rolling guard] --> SEG[Quasi-steady<br/>segments]
    A[Stage A · rig fit<br/>Student-t likelihood, Sobol multi-start,<br/>AR(1)-tempered Laplace UQ] --> B
    SEG --> B[Stage B · track calibration<br/>hierarchical errors-in-variables MAP<br/>LMUY per axle, latent sideslip]
    B --> V[Validation<br/>leave-one-run-out, grip limit,<br/>handling diagram]
    A --> X[.tir export ISO-W<br/>rig + track front/rear]
    B --> X
    V --> R[HTML report · CSV · figures]
```

Both stages are differentiable PyTorch programs in float64; every gradient,
Hessian and Jacobian is exact (autograd). No step uses finite differences.

## 1. Conventions (one geometry, defined once)

* Vehicle frame: SAE body axes (x forward, y right, z down). Left wheels at
  `y = −T/2`. `r < 0, ay < 0` is a left turn.
* Tyre convention used internally: `α = δ − atan(v_y/v_x)` of the contact point
  (= −α_SAE) and **α > 0 ⇒ Fy > 0**.
* TTC (SAE) → internal: `α = −SA, Fy = +FY, Fz = −FZ`. v1 used `α = +SA, Fy = −FY`,
  which is exact only for an odd curve and mirrors `SHy`, `SVy`, `pEy3` relative to
  the car (§2.1).
* Internal → `.tir` (ISO-W): `PKY1 = −pKy1`, `PVY1..4 = −pVy1..4`, everything else
  unchanged. A reference evaluator written from the published MF6.2 equations
  (`export/tir.py::tir_fy0`) reproduces the PyTorch model to < 0.05 N (unit test).
* Wheel positions are returned by `vehicle.wheel_positions()` and used by the
  slip kinematics, the load transfer and the yaw balance. The steer-induced yaw
  moment is therefore `Mδ = ½Tf (F_FR − F_FL) sin δ`, the sign fix of §6.5.

## 2. Tyre model and parameterisation

MF6.2 pure lateral with the standard scaling factors (`LMUY` on friction and,
through `LMUY' = 10·LMUY/(1+9·LMUY)`, on `SVy`; `LKY` on cornering stiffness).

| v1 | v2 |
|---|---|
| hard clamps on `Cy ∈ [1.4, 1.6]`, `Ey ∈ [−1, 0.99]`, `Fz ≥ 30 N` (zero-gradient attractors, §6.4) | every coefficient mapped from ℝ by a scaled sigmoid; `Ey ≤ 1` by a smooth softplus cap; smooth `Fz` floor |
| `sgn(·) ≈ tanh(30·)` – a ±6° transition | `tanh(α/0.25°)` – asymmetry acts where MF6.2 says it does |
| hand-tuned quadratic walls (weights 0.1 … 5000) | weakly-informative Gaussian priors in physical units (`config.ParamSpec`) |
| camber terms trained with zero gradient | camber terms frozen automatically when the data contain no camber, and reported as structurally non-identifiable |

## 3. Stage A — rig identification

Residual in friction space, `r_i = (Fy_hat − Fy)/F̄z(load_i)`, every load level
weighted equally, robust Student-t likelihood (ν = 4) with a learned scale σ:

```
J(θ, σ) = (1/c) Σ_i w_i [ ρ_t(r_i/σ) + log σ ] + ½ Σ_j ((θ_j − μ_j)/s_j)²
```

* **Serial correlation.** TTC channels are logged at 100 Hz and the residuals of
  one sweep are strongly autocorrelated. `c = (1+ρ₁)/(1−ρ₁)`, the AR(1)
  variance-inflation factor of the residuals, tempers the data term so that the
  posterior reflects the *effective* sample size (≈ 175 of 6 582 rows on the P26
  data). Without it every uncertainty would be ≈ 6× too small.
* **Global search.** 24 scrambled-Sobol starts inside the prior ±2σ box run
  Adam in parallel (one batched tensor program), the best six are polished with
  full-batch strong-Wolfe L-BFGS, and every end point is kept. The fraction of
  starts that reach the optimum is reported; the envelope of near-optimal
  curves (Fig. 05) is the empirical picture of what the rig data do not fix.
* **Uncertainty.** Laplace approximation: exact Hessian of `J` with respect to
  the unconstrained active parameters and `log σ`, eigen-decomposed (directions
  with non-positive curvature are counted and lifted), mapped to physical units
  with the sigmoid Jacobian. Derived quantities (B, C, D, E, `Kyα`, μ_peak,
  α_peak, whole curves) get ±σ by the delta method with autograd Jacobians.
  The peak slip angle is computed exactly by Newton's method on
  `φ(u) = tan(π/2Cy)`, differentiably.

## 4. Stage B — track calibration as an errors-in-variables problem

### Why v1 degenerates

The steady-state balances turn the telemetry into an axle-force sensor
(`F_f ≈ (m ay lr + Iz ṙ)/L`, `F_r ≈ (m ay lf − Iz ṙ)/L`, §2.5). The regressor –
the kinematic slip angle – carries 1–4° of noise and run-to-run bias. With slip
taken at face value, least squares is minimised by moving every operating point
onto the flat post-peak plateau, where force no longer depends on the noisy
regressor (regression dilution, §6.1–6.2). That is the `vy,off = +2.5 m/s`,
`ay R² < 0` signature of the v1 runs.

### v2 formulation

1. **Segments, not samples.** Each run is cut into 1-s windows; windows that are
   quasi-steady (`|mean ṙ| < 0.4 rad/s²`, `std r < 0.25 rad/s`), near pure-lateral
   (`|mean ax| < 2 m/s²`) and cornering (`|ay| > 0.5 g`) are kept as means. The
   within-window spread of measured sideslip, corrected for autocorrelation,
   becomes that segment's measurement error.
2. **Latent sideslip (EIV).** For segment `k` of run `j`:
   `β_k = β_meas,k + b_global + b_run[j] + e_k` with
   `b_global ~ N(0, σ_g²)`, `b_run ~ N(0, σ_r²)`, `e_k ~ N(0, σ_s² + sem_k²)`
   (non-centred parameterisation). The sensor error is absorbed where it occurs
   instead of being traded against the tyre curve.
3. **MF-native grip scaling.** `LMUY` per axle (friction only, cornering
   stiffness preserved – the abrasive TTC belt inflates friction, not carcass
   stiffness), optional `LKY` with a tight log-normal prior. Run E/G of the
   report pointed this way.
4. **Rig shape.** `fixed` (default, as recommended in §8.3) or `laplace`: the
   stage-A posterior `N(θ̂, Σ)` becomes the stage-B prior – the exact Bayesian
   update under the Laplace approximation, without letting 140 segments
   overwrite 6 600 rig samples.
5. **Combined slip.** Rear-wheel drive force `(m ax + ½ρCdA V²)/2` de-rates the
   lateral force through a smooth friction ellipse.
6. **Noise model.** Student-t residuals for both balances with a single learned
   scale and a fixed lateral/yaw ratio. (Learning the two scales independently is
   degenerate in an EIV model: one latent per segment zeroes the yaw residual
   exactly, its scale collapses and the lateral balance is ignored.)
7. **Optimisation.** MAP by Adam → L-BFGS from eight starts on a grid of
   initial sideslip offsets and grip levels; every basin is reported. Laplace
   covariance over all ≈ 150 free quantities (exact Hessian via
   `torch.func.functional_call`).
8. **Optional:** latent steering-zero offsets per run (`sigma_steer_*_deg`), for
   data sets whose steering channel is suspect.

### Diagnostics produced

* `saturated_fraction` – share of wheel-segments beyond the curve peak. Above
  60 % the report states that the telemetry fixes the friction level but not the
  on-track slip angles or cornering stiffness (prior-dominated).
* Multi-start basin table – competing explanations within 5 nats are flagged.
* Per-run sideslip corrections with ±σ, and their m/s equivalents.

## 5. Validation

* **Leave-one-run-out.** Refit without each run; predict the held-out run from
  measured kinematics with only the global offset (no run or segment latent).
  Run means and the spread of `LMUY` across folds are reported.
* **Grip limit.** Steady-state solver: `(β, δ)` solved by damped Newton
  (autograd Jacobian) on a constant radius, continued in speed, the limit
  refined by bisection. Compared with the lateral acceleration actually driven
  at each run's radius.
* **Handling diagram.** `δ − L/R` vs. `ay`, model vs. measured segments, and the
  linear-range understeer gradient.

## 6. What the P26 data support (v2 run)

| quantity | status |
|---|---|
| rig MF6.2 shape and load sensitivity within ±12°, 217–1126 N | identified, R² 0.9993, Laplace ±σ on every coefficient |
| track friction scale `LMUY` (front/rear) | identified, ±0.003, stable across leave-one-run-out folds |
| on-track slip angles, `LKY`, peak slip on track | **not identified** – > 90 % of segments are beyond the peak: the car was driven at its limit; the sideslip channel cannot place the operating points |
| camber terms | not identifiable (no camber in the data) – frozen |

Data that would change the last two rows: an optical/dual-antenna GNSS sideslip
sensor, a fixed steering zero, slow ramp-steer runs through the linear range and
several radii (§8.2).
