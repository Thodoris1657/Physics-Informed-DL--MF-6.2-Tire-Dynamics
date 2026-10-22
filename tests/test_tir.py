import numpy as np
import torch
from conftest import TRUE

from mf62pinn.config import TyreConfig
from mf62pinn.export.tir import read_tir, tir_fy0, write_tir
from mf62pinn.tyre import MF62Lateral


def test_tir_roundtrip_reproduces_model(tmp_path):
    m = MF62Lateral(665.0, TyreConfig(), frozen=("pDy3", "pEy4", "pHy3", "pVy3", "pVy4"))
    m.set_values(TRUE)
    for lmuy, lky in ((1.0, 1.0), (0.45, 0.9)):
        p = write_tir(tmp_path / "t.tir", m.values(), 665.0, lmuy=lmuy, lky=lky)
        t = read_tir(p)
        assert t["PKY1"] < 0 and t["FITTYP"] == 62
        a = np.radians(np.r_[np.linspace(-15, -1, 40), np.linspace(1, 15, 40)])
        for Fz in (250.0, 665.0, 1200.0):
            iso = tir_fy0(t, a, np.full_like(a, Fz))
            with torch.no_grad():
                internal = m(torch.tensor(a), torch.full((a.size,), Fz), lmuy=lmuy, lky=lky).numpy()
            assert np.max(np.abs(iso + internal)) < 0.05      # Fy_ISO = -Fy_internal
            assert np.all(iso[a > 0.02] < 0)                    # ISO-W: alpha > 0 -> Fy < 0
