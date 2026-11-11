"""The grouped bar-chart layout of the two monitor figures.

A panel is groups along the x axis with one bar per series inside each group. A bar's value, in percent, is
(mean, low, high), drawn as the bar and a thin 95% error bar; None leaves the slot empty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from monitors.common import CHARACTER_TRAINED, GPT_OSS_20B
from src.plotting import WONG, setup_style

#: one colour per group: gpt-oss-20b, and the character-trained models pooled (assigned rules and value-grounded
#: rules; teal from Paul Tol's muted scheme)
GROUP_COLOR = {GPT_OSS_20B: WONG[6], CHARACTER_TRAINED: "#44AA99"}
#: share of a group's slot on the x axis that its bars fill
GROUP_WIDTH = 0.8
DPI = 300
#: the paper's printed column width in inches (two-column layout); a figure drawn at this width and included at the
#: same width prints its text at the sizes set in print_style
COL_W = 3.25
#: height kept under the panels for each legend row, in inches
LEGEND_H = 0.22
#: blank border kept around a saved figure, in inches
PAD = 0.02
HIGHER_BETTER = "↑ better"
#: gap between the top of the y axis and the bottom of better()'s text, in points
BETTER_GAP = 4


def print_style() -> None:
    """Font sizes for a figure drawn at its printed width: close to the paper's caption size."""
    setup_style()
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.grid": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.4,
        }
    )


def panel(ax, groups: list[str], series: list[str], values: dict, title: str, colors: dict) -> None:
    """Bars for values[(group, series)]: (mean, low, high), or None for an empty slot."""
    w = GROUP_WIDTH / len(series)
    for i, g in enumerate(groups):
        present = [s for s in series if values.get((g, s)) is not None]
        for k, s in enumerate(present):
            m, lo, hi = values[(g, s)]
            x = i - GROUP_WIDTH / 2 + w * (k + 0.5) + w * (len(series) - len(present)) / 2
            ax.bar(x, m, w * 0.92, color=colors[s], edgecolor="none", lw=0.6)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=1.5)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_title(title)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def higher_better(ax) -> None:
    """ "↑ better" as upright text just above the top of the y axis, its left edge at the left edge of the y
    tick labels, at the tick labels' font size. Placed at draw time, so it follows the tick labels through
    tight_layout and bbox_inches="tight"."""

    def anchor(renderer):
        labels = [t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        top = ax.get_window_extent(renderer)
        x0 = min(b.x0 for b in labels) if labels else top.x0
        return Bbox([[x0, top.y0], [top.x1, top.y1]])

    size = ax.yaxis.get_major_ticks()[0].label1.get_fontsize()
    ax.annotate(
        HIGHER_BETTER,
        xy=(0, 1),
        xycoords=anchor,
        xytext=(0, BETTER_GAP),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=size,
        annotation_clip=False,
    )


def letters(axes) -> None:
    """Panel letters (a), (b), ... in order, for captions to refer to: in the title line of a titled panel, above
    the top-left corner of an untitled one."""
    for i, ax in enumerate(axes):
        letter = f"({chr(ord('a') + i)})"
        if ax.get_title():
            ax.set_title(f"{letter} {ax.get_title()}")
        else:
            ax.text(0, 1.03, letter, transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")


def save(fig, path: Path, legend_rows: int) -> None:
    """Lay out with legend_rows legend rows kept under the panels, and save `path` (a PNG at DPI) with a vector PDF
    beside it. The figure is drawn before the layout and again before saving, so the tick labels exist when the
    layout and higher_better()'s position are computed."""
    fig.canvas.draw()
    fig.tight_layout(rect=(0, LEGEND_H * legend_rows / fig.get_figheight(), 1, 1), pad=0.3, w_pad=0.6, h_pad=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=DPI, bbox_inches="tight", pad_inches=PAD)
        print(target)
