"""Shared plotting style: restrained palette, redundant encoding (colour + marker + line style), grayscale safe.

The first three LLMs get exactly the encoding used in the manuscript (navy circle solid, amber square dashed,
purple triangle dash-dot); further LLMs continue the cycle.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

INK, GRID, FONT = "#1f1f1f", "#e6e6e6", 9.5
WIDTH = 5.28            # single-column width in inches (13.4 cm)
HEIGHT = 2.65
MS, LW, CAP = 6.5, 1.7, 2.5

COLORS = ["#0B3C8C", "#E69F00", "#8E5EA2", "#00798C", "#B03A6F", "#555555", "#7F9A2E", "#A0522D"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (6, 2, 1, 2))]
HATCH = ["", "////", "xxxx", "....", "\\\\\\\\", "++++", "ooo", "---"]
FASTER, SLOWER = "#00798C", "#B03A6F"
CHANCE_COLOR, CHANCE_LW = "#555555", 1.3
CHANCE = dict(color=CHANCE_COLOR, ls=(0, (1, 1.6)), lw=CHANCE_LW)



def apply_theme() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Charis SIL", "Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
        "axes.linewidth": 0.9, "hatch.linewidth": 0.8, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": INK, "ps.fonttype": 42, "pdf.fonttype": 42,
    })


def styles(runs) -> dict:
    """run name -> dict(color, marker, ls, label, mfc, hatch)."""
    out = {}
    for i, r in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        out[r.name] = dict(color=c, marker=MARKERS[i % len(MARKERS)], ls=LINESTYLES[i % len(LINESTYLES)], label=r.label, mfc=c,
                           hatch=HATCH[i % len(HATCH)])
    return out


def clean(ax, ylabel=None, xlabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


PREFIX = ""   # set by an_report.build_report: every output file is <PREFIX><name>.<ext> in one flat folder


def save(fig, folder, name: str, grayscale: bool = True) -> None:
    """PNG (300 dpi), PDF and EPS; plus a grayscale proof <name>_gray.png in the same folder."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = PREFIX + name
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "eps"):
        fig.savefig(folder / f"{base}.{ext}")
    fig.savefig(folder / f"{base}.png", dpi=300)
    plt.close(fig)
    if grayscale:
        Image.open(folder / f"{base}.png").convert("L").save(folder / f"{base}_gray.png")
