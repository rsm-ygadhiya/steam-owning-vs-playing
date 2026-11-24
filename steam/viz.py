"""Task 9: one consistent figure style. Every figure is drawn in the notebook and
saved to figures/ for the slides.

Color rule (fixed across ALL charts):
  baselines -> grey, course models -> blue, advanced models -> orange.
Titles are the takeaway ("36% of owned games are never played"), not the chart type.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .config import FIG

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY = "#a3a29c"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FAMILY = {
    # baselines
    "Global mean": "baseline", "User mean": "baseline", "Item mean": "baseline",
    "Random": "baseline", "Most popular": "baseline", "Jaccard item-kNN": "baseline",
    "Item rate (baseline)": "baseline", "User rate (baseline)": "baseline",
    # course models
    "Bias model": "course", "Latent factor (MF)": "course", "BPR-MF": "course",
    "ALS (binary)": "course", "item2vec": "course", "Logistic (features)": "course",
    "Logistic (+ CF scores)": "course",
    # advanced
    "Factorization machine": "advanced", "LightGBM (stacked)": "advanced",
    "LightGBM (metadata only)": "advanced", "Hurdle (LightGBM)": "advanced",
    "Ensemble": "advanced", "ALS (playtime-weighted)": "advanced",
    "Content (tf-idf)": "advanced", "Hybrid (ALS + content)": "advanced",
    "Predicted popularity (content)": "advanced", "ALS factors mapped from content": "advanced",
    "Hybrid (mapped ALS + content)": "advanced",
}
FAMILY_COLOR = {"baseline": GREY, "course": BLUE, "advanced": ORANGE}


def color(model: str) -> str:
    return FAMILY_COLOR[FAMILY.get(model, "advanced")]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10.5,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2,
        "patch.linewidth": 0,
    })


def family_legend(ax, families=("baseline", "course", "advanced"), loc="lower right"):
    labels = {"baseline": "Baseline", "course": "Course model", "advanced": "Advanced model"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in families]
    ax.legend(handles, [labels[f] for f in families], loc=loc)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")


def hbar_models(ax, names, values, errs=None, fmt="{:.3f}", ref=None, ref_label=None):
    """Horizontal bars, one per model, colored by family, value labels at bar ends."""
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, xerr=errs, color=[color(n) for n in names], height=0.68,
            error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    span = max(values) - min(0, min(values))
    ax.set_xlim(min(0, min(values)), max(values) + 0.22 * span)
    for yy, v in zip(y, values):
        ax.text(v + 0.01 * span, yy, fmt.format(v), va="center", fontsize=9, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        if ref_label:
            ax.text(ref, len(names) - 0.4, " " + ref_label, color=INK2, fontsize=8.5, va="bottom")


# --------------------------------------------------------------- diagrams
def box_diagram(ax, boxes, arrows, title=None):
    """boxes: list of (x, y, w, h, text, fill); arrows: list of ((x0,y0),(x1,y1))."""
    ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    for x, y, w, h, text, fill in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    fc=fill, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10,
                color=INK if fill not in (BLUE, ORANGE, INK2) else "white", wrap=True)
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    if title:
        ax.set_title(title)
