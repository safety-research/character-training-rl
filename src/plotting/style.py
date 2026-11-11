"""Shared plotting style: house matplotlib settings, colorblind palette, color utilities.

Every figure in this repository uses these settings, so they share one look.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Colorblind-friendly qualitative palette: Paul Tol's "bright" scheme plus an orange. Cycle for
# categorical lines/bars. (The constant keeps the name WONG for its importers.)
WONG = [
    "#4477AA",  # blue
    "#EE6677",  # red/rose
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#EE7733",  # orange
]

# Default figure sizes (inches).
FIG_SINGLE = (8, 5)
FIG_WIDE = (12, 5)
FIG_TALL = (8, 8)
FIG_ROW4 = (20, 4.6)  # a 1x4 row of panels


def setup_style() -> None:
    """Apply the consistent house matplotlib style (call once before plotting)."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def palette_color(i: int) -> str:
    """Categorical color by index, cycling the palette."""
    return WONG[i % len(WONG)]


def lighten(hex_color: str, factor: float = 0.45) -> str:
    """Lighten a hex color toward white by `factor` in [0,1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def darken(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color by multiplying channels by `factor` in [0,1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"
