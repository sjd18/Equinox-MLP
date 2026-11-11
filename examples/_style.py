"""Shared matplotlib styling for the example figures."""

import matplotlib as mpl

mpl.use("Agg")

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1-3, validated for colour-vision deficiency (all pairs).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"Pair-potential MLP": BLUE, "Global-potential MLP": ORANGE, "Direct-force MLP": AQUA}

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "font.size": 10,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)
