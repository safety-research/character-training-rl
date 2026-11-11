"""The line-plot look of the reasoning figure: fonts and widths sized for a figure drawn at the paper's text
width, the 0-100% panel, the "lower is better" mark and the panel letters."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from src.plotting import setup_style

TEXT_W = 6.75
"""The paper's printed text width in inches (two-column layout); a figure drawn at this width and included at
the same width prints its text at the sizes set in line_style."""
DPI = 300
PAD = 0.02
"""Blank border kept around a saved figure, in inches."""
PX_PT = 0.397
"""Printed points per design pixel: the sizes below are given in the pixels of the paper's first figure, which
is drawn in pixels and printed at the text width."""
LINE_LW = 2.7 * PX_PT
AXIS_LW = 1.2 * PX_PT
HATCH_LW = 0.5 * PX_PT
"""Printed widths in points of a curve, the axes and a hatch line."""
GRID_Y, GRID_COLOR, GRID_LW = (50, 100), "#E3E3E3", 0.8 * PX_PT
"""Light horizontal gridlines of a 0-100% panel."""
Y_LOW = -4
"""Lower y limit of a 0-100% panel, which keeps a curve at 0 off the x axis."""
LOWER_BETTER = "↓ better"
BETTER_GAP = 4
"""Gap between the top of the y axis and the bottom of lower_better()'s text, in points."""


def line_style() -> None:
    """Font, sizes and axis widths for a line plot drawn at its printed width."""
    setup_style()
    body, tick = 14.5 * PX_PT, 13 * PX_PT
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
            "font.size": body,
            "axes.titlesize": body,
            "axes.labelsize": body,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "legend.fontsize": body,
            "axes.grid": False,
            "axes.linewidth": AXIS_LW,
            "xtick.major.width": AXIS_LW,
            "ytick.major.width": AXIS_LW,
            "xtick.major.size": 5 * PX_PT,
            "ytick.major.size": 5 * PX_PT,
            "lines.linewidth": LINE_LW,
            "hatch.linewidth": HATCH_LW,
        }
    )


def line_axes(ax) -> None:
    """A 0-100% line panel: light gridlines at 50 and 100, y from just below 0, ticks at 0, 50 and 100."""
    for y in GRID_Y:
        ax.axhline(y, color=GRID_COLOR, lw=GRID_LW, zorder=0.2)
    ax.set_ylim(Y_LOW, 100)
    ax.set_yticks((0, 50, 100))


def lower_better(ax) -> float:
    """ "↓ better" as upright text just above the top of the y axis, its left edge at the left edge of the y tick
    labels, at the tick labels' font size. Placed at draw time, so it follows the tick labels through
    bbox_inches="tight". Returns the font size, for lettered_titles()."""

    def anchor(renderer):
        labels = [t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        top = ax.get_window_extent(renderer)
        x0 = min(b.x0 for b in labels) if labels else top.x0
        return Bbox([[x0, top.y0], [top.x1, top.y1]])

    size = ax.yaxis.get_major_ticks()[0].label1.get_fontsize()
    ax.annotate(
        LOWER_BETTER,
        xy=(0, 1),
        xycoords=anchor,
        xytext=(0, BETTER_GAP),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=size,
        annotation_clip=False,
    )
    return size


def lettered_titles(axes, better_size: float) -> None:
    """Put a panel letter (a), (b), ... in each title, and lift the titles of the row above the lower_better()
    text so the two never overlap."""
    lift = BETTER_GAP + 1.25 * better_size
    for i, ax in enumerate(axes):
        ax.set_title(f"({chr(ord('a') + i)}) {ax.get_title()}", pad=plt.rcParams["axes.titlepad"] + lift)
