# Input data format

## TTC rig data (required)

One file per load level (or any split – rows are grouped by file). Accepted:

* **`.csv`** with Calspan column names (delimiter auto-detected),
* **`.dat`** Calspan text export (line 1 description, line 2 channel names, line 3 units),
* **`.mat`** Calspan MATLAB export (1-D channel variables).

| channel | meaning | unit | required |
|---|---|---|---|
| `SA` | slip angle | deg | ✔ |
| `FY` | lateral force | N | ✔ |
| `FZ` | vertical load (negative in SAE) | N | ✔ |
| `IA` | inclination angle | deg | – (0 if absent) |
| `FX` | longitudinal force – used by the free-rolling guard | N | recommended |
| `P` | pressure | kPa | – (written to the `.tir`) |
| `V` | belt speed | km/h | – (written to the `.tir` as LONGVL) |

`ttc.convention: SAE` (default) is right for raw Calspan data. Use `ISO` for
data already converted to ISO-W.

Rows with `|FX/FZ| > ttc.max_fx_over_fz` (default 0.25) are rejected; if more
than 30 % of a file is rejected the file is flagged as *not a free-rolling
cornering sweep*. Use cornering sweeps (pure lateral), not drive/brake runs.

## Telemetry (optional – enables the track calibration)

One CSV per run (also `.xlsx`, `.parquet`). Map your column names in
`telemetry.columns`:

| canonical | P26 column | meaning | default unit (configurable) |
|---|---|---|---|
| `time` | `time` | time | s |
| `steer` | `steering` | steering-wheel angle (or road-wheel, `steer_is_roadwheel: true`) | deg / rad |
| `ax` | `a_x` | longitudinal acceleration, body frame | m/s² / g |
| `ay` | `a_y` | lateral acceleration, body frame | m/s² / g |
| `vx` | `u_x` | longitudinal velocity | m/s / km/h |
| `vy` | `u_y` | lateral velocity at the sensor | m/s / km/h |
| `yaw_rate` | `yaw_rate` | yaw rate | rad/s / deg/s |

Sign convention: the SAE body frame (y to the right). For a left-hand turn
`ay < 0` and `r < 0`. If your logger uses ISO (y to the left), negate `ay`, `vy`
and `yaw_rate` (and steering) before loading, or pre-process with a short script.

If the file also contains `<name>_raw` columns for all channels, the named ones
are assumed to be already filtered and are not smoothed again (P26 processed
files). Otherwise a Savitzky–Golay filter is applied once.

Set `vehicle.vy_sensor_x` to the sensor's distance ahead of the CoG; the lever
arm `Vy_CoG = Vy_sensor − x_s r` is then removed.

### What makes telemetry informative

The tool will tell you what your data identify. To identify on-track cornering
stiffness and peak slip (not only friction), the data need:

* a sideslip measurement better than ≈ 0.5° (optical or dual-antenna GNSS),
* a steering channel with a verified zero and sign,
* steady-state points below the limit (ramp-steer or several speeds per radius)
  and more than one radius.
