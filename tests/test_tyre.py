import math

import numpy as np
import torch
from conftest import TRUE

from mf62pinn.config import TyreConfig
from mf62pinn.tyre import PARAM_NAMES, MF62Lateral, characteristics, mf62_fy


def make(values=TRUE):
    m = MF62Lateral(665.0, TyreConfig(), frozen=("pDy3", "pEy4", "pHy3", "pVy3", "pVy4"))
    m.set_values(values)
    return m


def test_set_values_roundtrip():
    m = make()
    v = m.values()
    for k, val in TRUE.items():
        assert abs(v[k] - val) < 1e-9


def test_parameters_stay_inside_bounds():
    m = make()
    with torch.no_grad():
        m.u.copy_(torch.randn_like(m.u) * 50)
    th = m.theta()
    assert torch.all(th >= m.lo) and torch.all(th <= m.hi)


def test_slope_at_origin_is_cornering_stiffness():
    vals = dict(TRUE, pHy1=0.0, pHy2=0.0, pVy1=0.0, pVy2=0.0)
    m = make(vals)
    for Fz in (300.0, 665.0, 1100.0):
        a = torch.zeros(1, requires_grad=True)
        Fy = m(a, torch.tensor([Fz]))
        (dFy,) = torch.autograd.grad(Fy.sum(), a)
        c = characteristics(m.theta().detach(), torch.tensor(Fz), m.Fz0)
        assert abs(float(dFy) - float(c["Kya_N_per_rad"])) / float(c["Kya_N_per_rad"]) < 1e-6


def test_peak_from_newton_matches_dense_grid():
    m = make()
    a = np.radians(np.linspace(0.01, 25, 200001))
    for Fz in (250.0, 700.0, 1200.0):
        with torch.no_grad():
            Fy = m(torch.tensor(a), torch.full((a.size,), Fz)).numpy()
        c = characteristics(m.theta().detach(), torch.tensor(Fz), m.Fz0, side=1)
        assert abs(math.degrees(a[np.argmax(Fy)]) - float(c["alpha_peak_deg"])) < 0.02
        assert abs(Fy.max() - float(c["Fy_peak_N"])) < 0.05


def test_friction_scaling_preserves_stiffness():
    m = make(dict(TRUE, pHy1=0.0, pHy2=0.0, pVy1=0.0, pVy2=0.0))
    a = torch.tensor([1e-4])
    with torch.no_grad():
        f1 = m(a, torch.tensor([665.0]))
        f2 = m(a, torch.tensor([665.0]), lmuy=0.5)
    assert abs(float(f1 / f2) - 1.0) < 1e-3          # LMUY leaves Kya unchanged
    big = torch.tensor([math.radians(40)])
    with torch.no_grad():
        assert float(m(big, torch.tensor([665.0]), lmuy=0.5)) < 0.55 * float(m(big, torch.tensor([665.0])))


def test_ey_is_capped_below_one():
    m = make(dict(TRUE, pEy1=0.99, pEy2=0.9))
    th = m.theta().detach()
    c = characteristics(th, torch.tensor(1200.0), m.Fz0)
    assert float(c["E"]) < 1.0


def test_frozen_parameters_receive_no_gradient():
    m = make()
    Fy = m(torch.linspace(-0.2, 0.2, 21), torch.full((21,), 700.0))
    Fy.sum().backward()
    for i, n in enumerate(PARAM_NAMES):
        if n in m.frozen:
            assert float(m.u.grad[i]) == 0.0


def test_batched_theta_matches_loop():
    m = make()
    th = m.theta().detach()
    T = torch.stack([th, th * 1.01])
    a = torch.linspace(-0.2, 0.2, 11)
    Fz = torch.full((11,), 600.0)
    Fb = mf62_fy(T.unsqueeze(-2), a, Fz, m.Fz0)
    for k in range(2):
        assert torch.allclose(Fb[k], mf62_fy(T[k], a, Fz, m.Fz0))
