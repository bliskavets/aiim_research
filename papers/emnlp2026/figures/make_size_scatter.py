"""
Scatter plot of FinOpsBench accuracy vs model size.

Sources:
  Open-weight sizes are vendor-reported total parameters.
  Proprietary sizes come from Li (2026), "Incompressible Knowledge Probes",
  arXiv:2604.24827. Three of our proprietary models (GPT-5, GPT-4.1,
  GPT-5-mini) are listed in the paper's main table. The two o-series and
  GPT-4.1-mini sizes are recovered by inverting the paper's log-linear
  calibration (slope = 14.7 pp/decade, intercept = 13.3, anchored on the
  GPT-5 / GPT-5-mini estimates) applied to their reported IKP scores.

Two figures are emitted:
  fig_accuracy_vs_size           — all 8 models (Llama-3.1-8B included).
  fig_accuracy_vs_size_no_llama  — Llama-3.1-8B removed entirely, fit
                                    over the remaining 7 models.
"""

from __future__ import annotations
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# (model, size_B, ci_low_B, ci_high_B, v1_acc, v2_acc, source)
MODELS = [
    # frontier (proprietary) — estimated sizes
    ("GPT-5",         4100,  1400, 12100, 68.9, 69.6, "ikp"),
    ("GPT-4.1",       2200,   719,  6400, 62.4, 60.6, "ikp"),
    ("o4-mini",        480,   160,  1440, 67.1, 67.3, "ikp"),
    ("GPT-5-mini",     410,   137,  1200, 65.8, 67.5, "ikp"),
    ("GPT-4.1-mini",   213,    71,   639, 61.5, 56.9, "ikp"),
    # open-weight — known sizes
    ("Qwen3-30B-A3B",   30,  None, None, 50.5, 53.0, "open"),
    ("Qwen3-8B",         8,  None, None, 47.6, 44.1, "open"),
    ("Llama-3.1-8B",     8,  None, None, 21.9, 16.3, "open"),
]

COL_V1 = "#3b6fb0"
COL_V2 = "#d97a3f"
EDGE = "#1a1a1a"

LABEL_POS = {
    "GPT-5":         (4100, 73, "right"),
    "GPT-4.1":       (2200, 56, "left"),
    "o4-mini":       (480,  72, "right"),
    "GPT-5-mini":    (410,  62, "right"),
    "GPT-4.1-mini":  (213,  53, "left"),
    "Qwen3-30B-A3B": (30,   57, "left"),
    "Qwen3-8B":      (10,   46, "left"),
    "Llama-3.1-8B":  (10,   16, "left"),
}


def render(models, out_stem: str, fit_excluding=None, title_suffix="",
           title=None):
    """Render the scatter for the given model subset.

    `fit_excluding` is the name of a model to drop from the regression
    (e.g. when Llama is in the dataset but we still want a robust fit).
    Pass None to fit over every passed model.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for model, size, lo, hi, v1, v2, source in models:
        ax.plot([size, size], [v1, v2], color="#888", linewidth=0.7,
                linestyle=":", zorder=1)
        marker = "o" if source == "open" else "D"
        ax.scatter([size], [v1], s=85, marker=marker,
                   facecolor=COL_V1, edgecolor=EDGE, linewidth=0.7, zorder=4)
        ax.scatter([size], [v2], s=85, marker=marker,
                   facecolor=COL_V2, edgecolor=EDGE, linewidth=0.7, zorder=4)

    for model, size, _, _, v1, v2, source in models:
        x, y, ha = LABEL_POS[model]
        label = model + (" (est.)" if source == "ikp" else "")
        ax.annotate(label, (x, y), fontsize=8.5, ha=ha, va="center",
                    color="#222")

    fit_models = [t for t in models if t[0] != fit_excluding]
    xs = np.array([math.log10(s) for _, s, _, _, _, _, _ in fit_models])
    y1 = np.array([v1 for _, _, _, _, v1, _, _ in fit_models])
    y2 = np.array([v2 for _, _, _, _, _, v2, _ in fit_models])
    s1, i1 = np.polyfit(xs, y1, 1)
    s2, i2 = np.polyfit(xs, y2, 1)
    xx = np.linspace(math.log10(5), math.log10(15000), 100)
    ax.plot(10 ** xx, s1 * xx + i1, "--", color=COL_V1, linewidth=1.0,
            alpha=0.55, zorder=1)
    ax.plot(10 ** xx, s2 * xx + i2, "--", color=COL_V2, linewidth=1.0,
            alpha=0.55, zorder=1)
    fit_note = (f" excluding {fit_excluding}"
                if fit_excluding and any(t[0] == fit_excluding for t in models)
                else "")
    print(f"[{out_stem}] v1 fit{fit_note}: "
          f"acc = {s1:.2f} * log10(N_B) + {i1:.2f}")
    print(f"[{out_stem}] v2 fit{fit_note}: "
          f"acc = {s2:.2f} * log10(N_B) + {i2:.2f}")

    ax.set_xscale("log")
    ax.set_xlim(5, 15000)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Model size (B parameters, log scale)")
    ax.set_ylabel("Accuracy (%)")
    if title is None:
        ax.set_title("FinOpsBench accuracy vs. model size" + title_suffix)
    elif title:  # non-empty override
        ax.set_title(title)
    # else: empty string => no title at all

    ticks = [8, 30, 100, 300, 1000, 3000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}B" if t < 1000 else f"{t // 1000}T" for t in ticks])

    legend_handles = [
        mpatches.Patch(facecolor=COL_V1, edgecolor=EDGE, label="FinOpsBench-v1"),
        mpatches.Patch(facecolor=COL_V2, edgecolor=EDGE, label="FinOpsBench-v2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor=EDGE, markersize=8, label="open-weight"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888",
               markeredgecolor=EDGE, markersize=8,
               label="closed (IKP estimates)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False,
              ncol=2, columnspacing=1.2, handletextpad=0.6)

    fig.savefig(f"{out_stem}.png")
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)
    print(f"Saved {out_stem}.{{png,pdf}}")


# Variant 1: all 8 models, fit excludes Llama-3.1-8B (matches the prior
# version of this figure).
render(MODELS,
       out_stem="fig_accuracy_vs_size",
       fit_excluding="Llama-3.1-8B")

# Variant 2: Llama-3.1-8B dropped from the plot entirely; fit runs over the
# remaining 7 models.
render([t for t in MODELS if t[0] != "Llama-3.1-8B"],
       out_stem="fig_accuracy_vs_size_no_llama",
       fit_excluding=None,
       title="")  # empty string => no title rendered on this variant
