from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRUE = {"pCy1": 1.45, "pDy1": 2.5, "pDy2": -0.15, "pEy1": 0.2, "pEy2": -0.4, "pEy3": 0.1,
        "pKy1": 48.0, "pKy2": 1.5, "pHy1": 0.002, "pHy2": -0.001, "pVy1": 0.01, "pVy2": -0.01}


def synthetic_ttc(tmp_path: Path, Fz0=665.0, noise=8.0, seed=0, loads=(220, 440, 660, 885, 1125)):
    """Write SAE-convention CSVs generated from a known internal-convention tyre."""
    from mf62pinn.config import TyreConfig
    from mf62pinn.tyre import MF62Lateral
    m = MF62Lateral(Fz0, TyreConfig())
    m.set_values(TRUE)
    rng = np.random.default_rng(seed)
    paths = []
    for L in loads:
        sa = np.concatenate([np.linspace(0, 12, 150), np.linspace(12, -12, 300), np.linspace(-12, 0, 150)])
        Fz = L + rng.normal(0, 5, sa.size)
        with torch.no_grad():
            Fy_int = m(torch.tensor(np.radians(-sa)), torch.tensor(Fz)).numpy()
        df = pd.DataFrame({"SA": sa, "FY": Fy_int + rng.normal(0, noise, sa.size), "FZ": -Fz,
                           "IA": 0.0, "FX": rng.normal(0, 5, sa.size), "P": 68.9, "V": 40.2})
        p = tmp_path / f"synthetic_{L}N.csv"
        df.to_csv(p, index=False)
        paths.append(str(p))
    return paths, m


@pytest.fixture
def synth(tmp_path):
    return synthetic_ttc(tmp_path)
