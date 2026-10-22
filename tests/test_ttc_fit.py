import numpy as np
from conftest import TRUE

from mf62pinn.config import RunConfig
from mf62pinn.data import load_ttc
from mf62pinn.fit.ttc import fit_ttc


def test_rig_fit_recovers_known_tyre(synth):
    paths, truth = synth
    cfg = RunConfig()
    cfg.ttc_fit.n_starts, cfg.ttc_fit.adam_iters, cfg.ttc_fit.lbfgs_iters = 4, 150, 300
    ttc = load_ttc(paths, cfg.ttc)
    res = fit_ttc(ttc, cfg)
    assert res.metrics["r2"] > 0.999
    assert res.metrics["rmse_N"] < 12.0                     # noise sd is 8 N
    for k in ("pCy1", "pDy1", "pDy2", "pKy1", "pKy2"):
        assert abs(res.values[k] - TRUE[k]) <= max(4 * res.std[k], 0.02 * abs(TRUE[k])), k
    assert set(res.frozen) >= {"pDy3", "pEy4", "pHy3", "pVy3", "pVy4"}


def test_sae_conversion_keeps_asymmetry_consistent(synth):
    """The loader must invert exactly the map used to generate SAE data."""
    paths, truth = synth
    cfg = RunConfig()
    cfg.ttc_fit.n_starts, cfg.ttc_fit.adam_iters, cfg.ttc_fit.lbfgs_iters = 3, 100, 300
    res = fit_ttc(load_ttc(paths, cfg.ttc), cfg)
    assert np.sign(res.values["pVy1"]) == np.sign(TRUE["pVy1"])
    assert abs(res.values["pHy1"] - TRUE["pHy1"]) < 0.002


def test_combined_slip_data_is_flagged(tmp_path):
    import pandas as pd
    sa = np.repeat([0.0, -3.0, -6.0], 200)
    df = pd.DataFrame({"SA": sa, "FY": 300 * sa, "FZ": -660.0, "IA": 0.0,
                       "FX": np.tile(np.linspace(-1200, 900, 200), 3)})
    p = tmp_path / "drive_brake.csv"
    df.to_csv(p, index=False)
    ttc = load_ttc([str(p)], RunConfig().ttc)
    assert any(n["level"] == "error" and "free-rolling" in n["msg"] for n in ttc.notes)
