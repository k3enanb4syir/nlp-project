# src/viz.py
"""
Shared plotting style for the EDA and results notebooks.

Keeping the palette and rcParams in one module means every figure in the repo
reads as one system, and the colour choices are stated once rather than
re-invented per chart.

PALETTE
-------
A validated categorical palette. The first three slots are used for multi-series
charts because that subset clears colour-vision-deficiency separation on an
all-pairs basis; charts needing more categories use a single-hue sequential ramp
instead of extending the categorical set.

Rules followed throughout:
  * Colour encodes identity, never rank -- a series keeps its colour when other
    series are filtered out.
  * Sequential magnitude uses one hue, light to dark. Diverging uses blue/red
    with a neutral grey midpoint, never a rainbow.
  * Every chart with two or more series carries a legend, so identity is never
    conveyed by colour alone.
  * Grid and axes are recessive; marks are thin.
  * No dual-axis charts. Two measures on different scales get two panels.
"""

from __future__ import annotations

import matplotlib as mpl

# Categorical slots (light mode). Fixed order -- assigned, never cycled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Single-hue sequential ramp (blue, light -> dark) for continuous magnitude.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281"]

# Diverging pair with a neutral grey midpoint, for signed quantities.
DIVERGING_LOW = BLUE
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = RED

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"

# Semantic roles used repeatedly across the notebooks.
COLOR_RATE = BLUE
COLOR_NEWS = ORANGE
COLOR_UP = AQUA
COLOR_DOWN = RED

# Split shading, kept deliberately faint so it never competes with the data.
SPLIT_COLORS = {"train": "#eef3fa", "validation": "#fdf1ea", "test": "#eaf7f2"}


def apply_style() -> None:
    """Install the project's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        # Recessive frame: drop the top/right spines entirely.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,

        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.labelsize": 10,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "legend.frameon": False,
        "legend.fontsize": 9,

        "lines.linewidth": 1.6,
        "lines.markersize": 4,

        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 10,

        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def shade_splits(ax, train_end, val_end, label: bool = True) -> None:
    """
    Shade the train / validation / test regions behind a time-series axis.

    Makes the chronological split visible in every temporal chart, which is the
    single most important structural fact about this dataset.
    """
    x0, x1 = ax.get_xlim()
    import matplotlib.dates as mdates

    te = mdates.date2num(train_end)
    ve = mdates.date2num(val_end)

    ax.axvspan(x0, te, color=SPLIT_COLORS["train"], zorder=0)
    ax.axvspan(te, ve, color=SPLIT_COLORS["validation"], zorder=0)
    ax.axvspan(ve, x1, color=SPLIT_COLORS["test"], zorder=0)

    for boundary in (te, ve):
        ax.axvline(boundary, color=TEXT_SECONDARY, lw=0.8, ls="--", alpha=0.55, zorder=1)

    if label:
        # Placed just inside the top of the axes rather than above it -- drawing
        # outside the axes collides with the title.
        for centre, name in (
            ((x0 + te) / 2, "train"),
            ((te + ve) / 2, "validation"),
            ((ve + x1) / 2, "test"),
        ):
            ax.text(centre, 0.97, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=TEXT_SECONDARY)

    ax.set_xlim(x0, x1)


def bar_labels(ax, bars, fmt: str = "{:.3f}", offset: float = 0.002) -> None:
    """Direct-label a small set of bars; avoids needing a value axis lookup."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                fmt.format(height), ha="center", va="bottom",
                fontsize=8, color=TEXT_SECONDARY)
