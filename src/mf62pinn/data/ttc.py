"""
Calspan Tire Test Consortium (TTC) loader.

Sign convention
---------------
Calspan raw data use the SAE tyre axis system (positive SA -> negative FY).
The vehicle model (see :mod:`mf62pinn.vehicle`) uses ``alpha = -alpha_SAE`` and
``Fy = Fy_SAE``. The loader applies exactly that map::

    alpha = -SA,   Fy = +FY,   Fz = -FZ

v1 used ``alpha = +SA, Fy = -FY``, which is the correct map only for an odd
curve: it mirrors the asymmetric terms (SHy, SVy, pEy3) relative to the car,
so the side of the rig curve that the car actually used was the wrong one.
The two maps fit the rig data equally well; only the new one is consistent
with the wheel kinematics.

Data-provenance guard
---------------------
The v1 audit found a drive/brake (combined-slip) test archived under the same
file names as the cornering sweeps. Rows with ``|FX/FZ|`` above
``max_fx_over_fz`` are rejected, and a file that loses more than
``combined_slip_warn_fraction`` of its rows is flagged as "not a pure lateral
sweep" in the audit.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import TTCConfig
from .io import read_table

LBF_TO_N = 4.4482216152605


@dataclass
class TTCData:
    df: pd.DataFrame                     # alpha[rad], Fz[N], Fy[N], gamma[rad], source, source_id
    sources: List[str]
    notes: List[Dict] = field(default_factory=list)

    @property
    def Fz0_equal_weight(self) -> float:
        return float(self.df.groupby("source")["Fz"].mean().mean())

    def summary(self) -> pd.DataFrame:
        g = self.df.groupby("source")
        return pd.DataFrame({
            "rows": g.size(),
            "Fz_mean_N": g["Fz"].mean().round(1),
            "alpha_min_deg": np.degrees(g["alpha"].min()).round(2),
            "alpha_max_deg": np.degrees(g["alpha"].max()).round(2),
            "camber_mean_deg": np.degrees(g["gamma"].mean()).round(2),
            "mu_peak_raw": (g["Fy"].apply(lambda s: s.abs().quantile(0.995)) / g["Fz"].mean()).round(3),
        }).reset_index()


def _col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n].to_numpy(dtype=float)
        for c in df.columns:
            if c.strip().upper() == n.upper():
                return df[c].to_numpy(dtype=float)
    return None


def load_ttc(paths: List[str], cfg: TTCConfig) -> TTCData:
    frames, notes = [], []
    for sid, path in enumerate(paths):
        raw = read_table(path)
        SA, FY, FZ = _col(raw, "SA"), _col(raw, "FY"), _col(raw, "FZ")
        if SA is None or FY is None or FZ is None:
            notes.append({"file": os.path.basename(path), "level": "error",
                          "msg": "missing SA/FY/FZ columns - skipped"})
            continue
        IA = _col(raw, "IA")
        IA = np.zeros_like(SA) if IA is None else IA
        FX = _col(raw, "FX")
        P = _col(raw, "P")
        V = _col(raw, "V")

        if cfg.convention.upper() == "SAE":
            alpha, Fy, Fz = -np.radians(SA), FY, -FZ
        else:  # ISO-W: alpha_iso = -alpha_SAE, Fy_iso = -FY_SAE  -> internal (alpha, -Fy)
            alpha, Fy, Fz = np.radians(SA), -FY, FZ
        Fz = np.abs(Fz)
        gamma = np.radians(IA)

        ok = np.isfinite(alpha) & np.isfinite(Fy) & np.isfinite(Fz) & (Fz > cfg.fz_min)
        n_ok = int(ok.sum())
        if FX is not None:
            fr = np.abs(FX) / np.maximum(Fz, 1.0) <= cfg.max_fx_over_fz
            rejected = int((ok & ~fr).sum())
            ok &= fr
            frac = rejected / max(n_ok, 1)
            if frac > cfg.combined_slip_warn_fraction:
                notes.append({"file": os.path.basename(path), "level": "error",
                              "msg": f"{100*frac:.0f}% of rows have |FX/FZ| > {cfg.max_fx_over_fz}: "
                                     "this is not a free-rolling cornering sweep (drive/brake test?)"})
            elif rejected:
                notes.append({"file": os.path.basename(path), "level": "info",
                              "msg": f"{rejected} rows rejected by the free-rolling guard"})
        idx = np.flatnonzero(ok)[:: max(cfg.decimate, 1)]
        if idx.size < 20:
            notes.append({"file": os.path.basename(path), "level": "error", "msg": "too few valid rows"})
            continue
        d = pd.DataFrame({
            "alpha": alpha[idx], "Fz": Fz[idx], "Fy": Fy[idx], "gamma": gamma[idx],
            "P_kPa": P[idx] if P is not None else np.nan,
            "V_kph": V[idx] if V is not None else np.nan,
            "source": os.path.basename(path),
        })
        # a-priori one-sidedness check (identifiability of asymmetry terms)
        if (d["alpha"] > np.radians(1)).mean() < 0.05 or (d["alpha"] < -np.radians(1)).mean() < 0.05:
            notes.append({"file": os.path.basename(path), "level": "warn",
                          "msg": "slip angle covers only one sign: asymmetry terms not identifiable"})
        frames.append(d)
    if not frames:
        raise RuntimeError("No usable TTC data")
    df = pd.concat(frames, ignore_index=True)
    # order sources by load so plots / ids are monotone in Fz
    order = df.groupby("source")["Fz"].mean().sort_values().index.tolist()
    df["source_id"] = df["source"].map({s: i for i, s in enumerate(order)}).astype(int)
    return TTCData(df=df, sources=order, notes=notes)


def nominal_load_label(source: str, Fz_mean: float) -> str:
    m = re.search(r"(\d+)\s*lbs", source)
    if m:
        return f"{m.group(1)} lbs ({Fz_mean:.0f} N)"
    return f"{Fz_mean:.0f} N"
