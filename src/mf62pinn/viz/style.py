"""Shared figure style: validated categorical palette, ordinal ramp, quiet chrome."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical (fixed order; validated for CVD separation on the adjacent pairlist)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
WHEEL_COLORS = {"FL": BLUE, "FR": ORANGE, "RL": AQUA, "RR": YELLOW}
WHEEL_LS = {"FL": "-", "FR": "-", "RL": "--", "RR": "--"}     # secondary encoding: axle
RIG, TRACK = BLUE, ORANGE
CRITICAL = "#d03b3b"

_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a32c2c"])


def ordinal(n: int):
    """``n`` ordered colours along the blue ramp (light -> dark)."""
    if n <= 1:
        return [BLUE]
    xs = np.linspace(0, len(_RAMP) - 1, n)
    out = []
    for x in xs:
        i = int(np.floor(x))
        j = min(i + 1, len(_RAMP) - 1)
        t = x - i
        a, b = np.array(to_rgb(_RAMP[i])), np.array(to_rgb(_RAMP[j]))
        out.append(tuple(a + (b - a) * t))
    return out


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "font.family": ["DejaVu Sans"], "font.size": 9.5,
        "figure.dpi": 110,
    })


def label_end(ax, x, y, text, color=INK2, dx=4, fontsize=8):
    """Direct label at the end of a line (text in ink, not the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center",
                fontsize=fontsize, color=color)


apply()
