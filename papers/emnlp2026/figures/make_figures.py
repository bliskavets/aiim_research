"""
Generate publication-quality figures for the FinAgent-Bench dataset paper.

All input data is the values reported in the paper's tables (Tables 3, 4, 5
and the pipeline funnel from Table 2). Run from this directory:

    python3 make_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Matplotlib defaults for clean academic figures.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Data — verbatim from the paper's results tables.
# ---------------------------------------------------------------------------

# Tables 3 and 4 — accuracy on FinOpsBench-v1 and FinOpsBench-v2.
MODELS = [
    "GPT-5",
    "GPT-5-mini",
    "o4-mini",
    "GPT-4.1",
    "GPT-4.1-mini",
    "Qwen3-30B-A3B",
    "Qwen3-8B",
    "Llama-3.1-8B",
]
ACC_SYNTH = [68.9, 65.8, 67.1, 62.4, 61.5, 50.5, 47.6, 21.9]
ACC_CURATED = [69.6, 67.5, 67.3, 60.6, 56.9, 53.0, 44.1, 16.3]
IS_FRONTIER = [True, True, True, True, True, False, False, False]

# Table 5 — Native vs ReAct on FinOpsBench-v1.
REACT_MODELS = [
    "GPT-5",
    "o4-mini",
    "Qwen3-30B-A3B",
    "Qwen3-8B",
    "GPT-4.1",
    "GPT-4.1-mini",
    "Llama-3.1-8B",
]
REACT_NATIVE = [68.9, 67.1, 50.5, 47.6, 62.4, 61.5, 21.9]
REACT_REACT = [64.6, 64.3, 46.2, 44.3, 63.8, 63.5, 28.3]
REACT_IS_THINKING = [True, True, True, True, False, False, False]

# Pipeline funnel — Table 2.
# Stage 6 directly passes 5,401; the remaining 4,156 enter the improvement
# loop and 2,832 of them pass Stage 9. Total post-judgement: 5,401+2,832 = 8,233.
FUNNEL_STAGES = [
    "Query\nexpansion",
    "Schema\ngeneration",
    "Data\ngeneration",
    "Execution\nvalidation",
    "Agent trace\ngeneration",
    "Committee\njudgement",
    "Final\nfiltering",
]
FUNNEL_COUNTS = [10000, 10000, 10000, 10000, 9557, 8233, 5979]

# Consistent color palette: frontier vs open-source.
COL_FRONTIER = "#3b6fb0"
COL_OPEN = "#d97a3f"
COL_SYNTH = "#5a9bd5"
COL_CURATED = "#f1a340"


# ---------------------------------------------------------------------------
# Figure 1 — Grouped accuracy bars across both benchmarks.
# ---------------------------------------------------------------------------

def fig_accuracy_bars():
    n = len(MODELS)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    bars_s = ax.bar(x - width / 2, ACC_SYNTH, width,
                    label="FinOpsBench-v1", color=COL_SYNTH,
                    edgecolor="black", linewidth=0.4)
    bars_c = ax.bar(x + width / 2, ACC_CURATED, width,
                    label="FinOpsBench-v2", color=COL_CURATED,
                    edgecolor="black", linewidth=0.4)

    # Annotate each bar with its value.
    for bars in (bars_s, bars_c):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.1f}", ha="center", va="bottom",
                    fontsize=7.5)

    # Separator + label for the frontier/open-source split.
    split_at = sum(IS_FRONTIER) - 0.5
    ax.axvline(split_at, color="gray", linestyle=":", linewidth=0.8)
    ax.text(split_at / 2, 86, "frontier",
            ha="center", va="top", fontsize=9, style="italic", color="gray")
    ax.text((split_at + n) / 2, 86, "open-source",
            ha="center", va="top", fontsize=9, style="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylim(0, 90)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)

    fig.savefig("fig_accuracy_bars.png")
    fig.savefig("fig_accuracy_bars.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Native vs ReAct paired bars (FinOpsBench-v1 only).
# ---------------------------------------------------------------------------

def fig_native_vs_react():
    n = len(REACT_MODELS)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    bars_n = ax.bar(x - width / 2, REACT_NATIVE, width,
                    label="Native tool calling",
                    color="#bfbfbf", edgecolor="black", linewidth=0.4)
    bars_r = ax.bar(x + width / 2, REACT_REACT, width,
                    label="ReAct",
                    color="#6cb27b", edgecolor="black", linewidth=0.4)

    # Show the delta on top of each pair.
    for i in range(n):
        nat = REACT_NATIVE[i]
        rea = REACT_REACT[i]
        top = max(nat, rea)
        delta = rea - nat
        sign = "+" if delta >= 0 else ""
        ax.text(x[i], top + 1.4, f"{sign}{delta:.1f}",
                ha="center", va="bottom", fontsize=8,
                color=("#2a7a3a" if delta >= 0 else "#a93030"),
                fontweight="bold")

    # Visual split: thinking vs non-thinking models.
    # The "thinking / non-thinking" annotations were removed per author
    # preference — the LaTeX caption already explains the grouping.
    split_at = sum(REACT_IS_THINKING) - 0.5
    ax.axvline(split_at, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(REACT_MODELS, rotation=20, ha="right")
    ax.set_ylim(0, 90)
    ax.set_ylabel("Accuracy on FinOpsBench-v1 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)

    fig.savefig("fig_native_vs_react.png")
    fig.savefig("fig_native_vs_react.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Synth vs Curated scatter (per-model agreement).
# ---------------------------------------------------------------------------

def fig_synth_vs_curated_scatter():
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # Per-model label offsets, tuned to avoid overlaps.
    label_offsets = {
        "GPT-5":          (1.5,  0.8),
        "GPT-5-mini":     (-3.5, 2.2),
        "o4-mini":        (1.5, -1.0),
        "GPT-4.1":        (1.5,  0.8),
        "GPT-4.1-mini":   (1.5, -1.5),
        "Qwen3-30B-A3B":  (1.5,  0.8),
        "Qwen3-8B":       (1.5,  0.8),
        "Llama-3.1-8B":   (1.5,  0.8),
    }
    for synth, curated, model, frontier in zip(
        ACC_SYNTH, ACC_CURATED, MODELS, IS_FRONTIER
    ):
        color = COL_FRONTIER if frontier else COL_OPEN
        ax.scatter(synth, curated, s=90, color=color,
                   edgecolor="black", linewidth=0.6, zorder=3)
        dx, dy = label_offsets.get(model, (1.5, 0.8))
        ha = "right" if dx < 0 else "left"
        ax.annotate(model, (synth + dx, curated + dy),
                    fontsize=8, va="center", ha=ha)

    # y = x reference line.
    lo, hi = 10, 75
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray",
            linewidth=0.8, zorder=1)
    ax.text(72, 72, "y = x", fontsize=8, color="gray",
            ha="right", va="bottom", rotation=45)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Accuracy on FinOpsBench-v1 (%)")
    ax.set_ylabel("Accuracy on FinOpsBench-v2 (%)")

    legend_handles = [
        mpatches.Patch(facecolor=COL_FRONTIER, edgecolor="black",
                       label="Frontier"),
        mpatches.Patch(facecolor=COL_OPEN, edgecolor="black",
                       label="Open-source"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    ax.set_aspect("equal")

    fig.savefig("fig_synth_vs_curated_scatter.png")
    fig.savefig("fig_synth_vs_curated_scatter.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Pipeline funnel (attrition through stages).
# ---------------------------------------------------------------------------

def fig_pipeline_funnel():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    n = len(FUNNEL_STAGES)
    x = np.arange(n)

    bars = ax.bar(x, FUNNEL_COUNTS, color="#5a8db5",
                  edgecolor="black", linewidth=0.4, width=0.65)

    for i, (b, count) in enumerate(zip(bars, FUNNEL_COUNTS)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontsize=8)
        if i > 0:
            drop = FUNNEL_COUNTS[i - 1] - count
            if drop > 0:
                pct = 100 * drop / FUNNEL_COUNTS[i - 1]
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() / 2,
                        f"-{pct:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(FUNNEL_STAGES, fontsize=8)
    ax.set_ylabel("Number of examples")
    ax.set_ylim(0, 11500)
    ax.grid(axis="x", visible=False)

    fig.savefig("fig_pipeline_funnel.png")
    fig.savefig("fig_pipeline_funnel.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_accuracy_bars()
    fig_native_vs_react()
    fig_synth_vs_curated_scatter()
    fig_pipeline_funnel()
    print("Generated 4 figures (PNG + PDF) in", __file__.rsplit("/", 1)[0])
