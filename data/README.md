# Data

| folder | content |
|---|---|
| `TTC data/` | Calspan TTC Round 5, Hoosier 43075 16x7.5-10 R25B, 25 mph, 10 psi, 0° IA, free-rolling cornering sweeps (±12°) at 50/100/150/200/250 lbs. SAE convention. |
| `Testing_Data/` | P26 skidpad telemetry, processed (Savitzky–Golay filtered channels + `*_raw` originals), 100 Hz: four runs on the 10.5 m circle, one on 9 m, four on 6 m. |

Known issues (detected automatically by `mf62pinn audit`):

* The four `processed_6m_*` runs have a steering channel whose sign contradicts the
  turn direction; they are quarantined by default (`telemetry.steer_sign_policy`).
* The lateral-velocity channel violates the kinematic identity `ay = dVy/dt + Vx·r`
  by ≈ 2.7 m/s² RMS and carries run-to-run bias – which is why the track calibration
  treats sideslip as a latent variable.

⚠️ TTC data are licensed to TTC members. Keep this folder out of public repositories
(add `data/TTC data/` to `.gitignore` before pushing to a public remote).
