"""The grouped bar-chart layout of the reward-hacking figures.

A panel is groups along the x axis with one bar per series inside each group. A bar's value, in percent, is either
one evaluation's (rate, low, high), drawn as the bar and a thin 95% error bar, or a list of rates, one per repeat
training, drawn as a bar at their mean with one dot per training; None leaves the slot empty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.transforms import Bbox

from src.plotting import setup_style

#: share of a group's slot on the x axis that its bars fill
GROUP_WIDTH = 0.8
#: the paper's printed column width in inches (two-column layout); a figure drawn at this width and included at the
#: same width prints its text at the sizes set in print_style
COL_W = 3.25
#: height kept under the panels for the legend row, in inches
LEGEND_H = 0.22
#: blank border kept around a saved figure, in inches
PAD = 0.02
#: resolution at which a saved figure's bounding box is measured, in dots per inch
DPI = 300
LOWER_BETTER, HIGHER_BETTER = "↓ better", "↑ better"
#: gap between the top of the y axis and the bottom of better()'s text, in points
BETTER_GAP = 4
#: below this relative luminance a bar is dark, and the dots inside it get a white ring of DOT_RING_LW points, drawn
#: at marker area DOT_RING_S so the dark centre stays the size of an unringed dot (area DOT_S)
DARK_BAR_LUMINANCE, DOT_RING_LW, DOT_RING_S, DOT_S = 0.13, 0.4, 9, 7


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


def _luminance(colour: str) -> float:
    """WCAG relative luminance of a colour."""
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def panel(ax, groups: list[str], series: list[str], values: dict, title: str, colors: dict) -> None:
    """Bars for values[(group, series)]: (rate, low, high), a list of rates (one per repeat training), or None."""
    w = GROUP_WIDTH / len(series)
    for i, g in enumerate(groups):
        present = [s for s in series if values.get((g, s)) is not None]
        for k, s in enumerate(present):
            val = values[(g, s)]
            x = i - GROUP_WIDTH / 2 + w * (k + 0.5) + w * (len(series) - len(present)) / 2
            if isinstance(val, list):
                top = sum(val) / len(val)
                ax.bar(x, top, w * 0.92, color=colors[s], edgecolor="none", lw=0.6)
                offs = [(j - (len(val) - 1) / 2) * w * min(0.22, 0.75 / len(val)) for j in range(len(val))]
                # a dot inside a dark bar gets a white ring so it stays visible; dots above the bar stay plain
                dark = _luminance(colors[s]) < DARK_BAR_LUMINANCE
                ring = [dark and y <= top for y in val]
                for ringed in (False, True):
                    pts = [(x + o, y) for o, y, r in zip(offs, val, ring) if r == ringed]
                    if pts:
                        ax.scatter(
                            *zip(*pts),
                            s=DOT_RING_S if ringed else DOT_S,
                            color="#333333",
                            zorder=3,
                            lw=DOT_RING_LW if ringed else 0,
                            edgecolors="white",
                        )
            else:
                m, lo, hi = val
                ax.bar(x, m, w * 0.92, color=colors[s], edgecolor="none", lw=0.6)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=1.5)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_title(title)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def better(ax, lower: bool) -> None:
    """Which direction is better, as upright text ("↓ better" or "↑ better") just above the top of the y
    axis, its left edge at the left edge of the y tick labels, at the tick labels' font size. Placed at draw time, so
    it follows the tick labels through tight_layout and bbox_inches="tight"."""

    def anchor(renderer):
        labels = [t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        top = ax.get_window_extent(renderer)
        x0 = min(b.x0 for b in labels) if labels else top.x0
        return Bbox([[x0, top.y0], [top.x1, top.y1]])

    size = ax.yaxis.get_major_ticks()[0].label1.get_fontsize()
    ax._better_size = size
    ax.annotate(
        LOWER_BETTER if lower else HIGHER_BETTER,
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
    """Panel letters (a), (b), ... in order, in the title line of each panel, for captions to refer to."""
    for i, ax in enumerate(axes):
        ax.set_title(f"({chr(ord('a') + i)}) {ax.get_title()}")


def clear_better(axes) -> None:
    """Lift the titles of `axes` above the better() text of any of them, so the two never overlap. Call after
    letters(), which resets title padding."""
    sizes = [ax._better_size for ax in axes if hasattr(ax, "_better_size")]
    if not sizes:
        return
    lift = BETTER_GAP + 1.25 * max(sizes)  # the better() text's height above the axes top, in points
    for ax in axes:
        ax.set_title(ax.get_title(), pad=plt.rcParams["axes.titlepad"] + lift)


def legend(fig, series: list[str], colors: dict) -> None:
    """One legend row under the panels, a swatch per series."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], ec="none") for s in series]
    fig.legend(
        handles,
        series,
        loc="lower center",
        ncol=len(series),
        frameon=False,
        bbox_to_anchor=(0.5, 0),
        handlelength=1.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def save(fig, path: Path, h_pad: float) -> None:
    """Lay out with one legend row kept under the panels (h_pad, in font-size units, between stacked panels) and save
    `path`, a vector PDF. The figure is drawn before the layout and again before saving, so the tick labels exist
    when the layout and better()'s position are computed."""
    fig.canvas.draw()
    fig.tight_layout(rect=(0, LEGEND_H / fig.get_figheight(), 1, 1), pad=0.3, h_pad=h_pad)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=PAD)
    print(path)
