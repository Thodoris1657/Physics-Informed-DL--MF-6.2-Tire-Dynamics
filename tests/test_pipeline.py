import json

import pytest
from conftest import DATA


@pytest.mark.slow
@pytest.mark.skipif(not (DATA / "TTC data").exists(), reason="bundled data not present")
def test_end_to_end_on_bundled_data(tmp_path):
    from mf62pinn import RunConfig, run_pipeline
    cfg = RunConfig()
    cfg.ttc.files = [str(DATA / "TTC data" / "*.csv")]
    cfg.telemetry.files = [str(DATA / "Testing_Data" / "*.csv")]
    cfg.ttc_fit.n_starts, cfg.ttc_fit.adam_iters, cfg.ttc_fit.lbfgs_iters = 4, 120, 200
    cfg.track_fit.adam_iters, cfg.track_fit.lbfgs_iters = 120, 200
    cfg.validation.leave_one_run_out = False
    cfg.output.out_dir = str(tmp_path / "run")
    res = run_pipeline(cfg, verbose=False, make_plots=False, make_report=True)
    assert res.ttc_fit.metrics["r2"] > 0.995
    assert 0.3 < res.track.lambdas["LMUY_front"] < 0.7
    out = tmp_path / "run"
    for f in ("tyre_rig_TTC.tir", "tyre_track_front.tir", "tyre_track_rear.tir", "report.html",
              "mf62_coefficients.json", "segments.csv", "SUMMARY.txt", "metrics.json"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    assert m["headline"]["ttc_r2"] > 0.995
