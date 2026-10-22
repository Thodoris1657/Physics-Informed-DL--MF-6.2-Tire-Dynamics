# Legacy v1 code

* `main.py` — the original single-script P26 PINN (MF6.2 + LoadTransfer_v2 + Adam/L-BFGS).
  Documented and audited in [`../docs/PINN_Lateral_MF62_Identification.pdf`](../docs/PINN_Lateral_MF62_Identification.pdf).
* `validate_numerics.py` — the NumPy cross-check of the v1 loaders, load transfer and kinematics.

Kept unchanged for reproducibility of the reference results in
[`../results/reference_v1/`](../results/reference_v1). Run from the repository root:

```bash
python legacy/main.py --out results/reference_v1_rerun
```

New work should use the `mf62pinn` package (see the top-level README). To compare the two
estimators inside one engine, use `configs/v1_compat.yaml`, which reproduces the v1
*formulation* (global offset, whole-force grip scale, all runs kept) in v2.
