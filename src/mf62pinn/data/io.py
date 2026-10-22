"""File readers: CSV, Calspan/TTC ``.dat`` (3-line header) and MATLAB ``.mat``."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def expand_files(patterns: Iterable[PathLike], base_dir: PathLike | None = None) -> List[str]:
    """Expand globs (relative to ``base_dir``), keep order, drop duplicates."""
    out: List[str] = []
    for patt in patterns:
        patt = str(patt)
        if base_dir is not None and not os.path.isabs(patt):
            patt = os.path.join(str(base_dir), patt)
        hits = sorted(glob.glob(patt)) if any(c in patt for c in "*?[") else [patt]
        for h in hits:
            if os.path.isfile(h) and h not in out:
                out.append(h)
    return out


def read_table(path: PathLike) -> pd.DataFrame:
    """Read a data table, auto-detecting the format from the extension."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return _read_mat(path)
    if ext == ".dat":
        return _read_ttc_dat(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    # csv / txt: sniff delimiter
    return pd.read_csv(path, sep=None, engine="python")


def _read_ttc_dat(path: str) -> pd.DataFrame:
    """TTC ``.dat``: line 1 description, line 2 channel names, line 3 units."""
    with open(path, "r", errors="replace") as f:
        head = [next(f) for _ in range(3)]
    names = head[1].split()
    df = pd.read_csv(path, sep=r"\s+", skiprows=3, header=None, engine="python")
    df.columns = names[: df.shape[1]]
    return df


def _read_mat(path: str) -> pd.DataFrame:
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True)
    cols = {}
    n = None
    for k, v in m.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.ndim == 1 and arr.dtype.kind in "fiu":
            if n is None:
                n = arr.shape[0]
            if arr.shape[0] == n:
                cols[k] = arr.astype(float)
    if not cols:
        raise ValueError(f"No 1-D numeric channels found in {path}")
    return pd.DataFrame(cols)
