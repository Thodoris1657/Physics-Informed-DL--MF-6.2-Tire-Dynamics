from mf62pinn.config import RunConfig, set_by_path


def test_yaml_roundtrip(tmp_path):
    c = RunConfig()
    c.vehicle.Iz = 123.0
    c.tyre.params["pKy1"].prior_sd = 7.0
    c.to_yaml(tmp_path / "c.yaml")
    d = RunConfig.from_yaml(tmp_path / "c.yaml")
    assert d.vehicle.Iz == 123.0 and d.tyre.params["pKy1"].prior_sd == 7.0
    assert d.to_dict() == c.to_dict()


def test_set_by_path_casts_types():
    c = RunConfig()
    set_by_path(c, "track_fit.sigma_beta_run_deg", "0.75")
    set_by_path(c, "validation.leave_one_run_out", "false")
    set_by_path(c, "ttc_fit.n_starts", "3")
    assert c.track_fit.sigma_beta_run_deg == 0.75
    assert c.validation.leave_one_run_out is False and c.ttc_fit.n_starts == 3


def test_unknown_key_is_rejected():
    import pytest
    with pytest.raises(KeyError):
        RunConfig.from_dict({"vehicle": {"not_a_field": 1}})
