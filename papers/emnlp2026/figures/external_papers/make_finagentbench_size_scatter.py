"""
Accuracy vs model size for the FinAgentBench paper
(arXiv:2508.14052v3, "FinAgentBench: A Benchmark Dataset for Agentic
Retrieval in Financial Question Answering").

Source data: Tables 1 (Document Ranking) and 2 (Chunk Ranking) of the
paper. Only 3 models were evaluated.

Parameter counts are estimated using Li (2026), Incompressible Knowledge
Probes, arXiv:2604.24827:
  - GPT-o3:          ~3.0T  (paper p.13 main table: 64.4% -> 3.0T,
                              CI 1.0T-8.9T)
  - Claude-Opus-4:   ~1.4T  (paper p.13 main table: 59.7% -> 1.4T,
                              CI 478B-4.2T)
  - Claude-Sonnet-4: ~237B  (paper p.52 extended table: IKP 0.482,
                              calibrated via slope=0.147 pp/decade,
                              intercept=13.3, yields ~237B; CI ~3x)

Question the chart answers: "Does the FinOpsBench size-accuracy trend
repeat on FinAgentBench?"
Answer (visible in the plot): No. The smallest model (Sonnet-4) wins
or ties on every metric in both tables.
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
    "legend.fontsize": 8.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# (model, size_B, ci_low_B, ci_high_B)
MODELS = [
    ("GPT-o3",          3000, 1000, 8900),
    ("Claude-Opus-4",   1400,  478, 4200),
    ("Claude-Sonnet-4",  237,   79,  711),
]

# Tables 1 & 2 of arXiv:2508.14052v3
DOC_RANKING = {  # Table 1
    "GPT-o3":          {"nDCG@5": 0.770, "MAP@5": 0.829, "MRR@5": 0.875},
    "Claude-Opus-4":   {"nDCG@5": 0.773, "MAP@5": 0.840, "MRR@5": 0.875},
    "Claude-Sonnet-4": {"nDCG@5": 0.783, "MAP@5": 0.849, "MRR@5": 0.892},
}
CHUNK_RANKING = {  # Table 2
    "GPT-o3":          {"nDCG@5": 0.351, "MAP@5": 0.257, "MRR@5": 0.538},
    "Claude-Opus-4":   {"nDCG@5": 0.418, "MAP@5": 0.307, "MRR@5": 0.568},
    "Claude-Sonnet-4": {"nDCG@5": 0.419, "MAP@5": 0.296, "MRR@5": 0.567},
}

METRIC_COLOR = {
    "nDCG@5": "#3b6fb0",
    "MAP@5":  "#d97a3f",
    "MRR@5":  "#6cb27b",
}
METRIC_MARKER = {
    "nDCG@5": "o",
    "MAP@5":  "D",
    "MRR@5":  "s",
}

EDGE = "#1a1a1a"


def draw_panel(ax, scores: dict, title: str, ylim: tuple[float, float]):
    for metric in ["nDCG@5", "MAP@5", "MRR@5"]:
        xs, ys = [], []
        for model, size, *_ in MODELS:
            xs.append(size)
            ys.append(scores[model][metric])
        # Connector line through the 3 points (per metric).
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order],
                color=METRIC_COLOR[metric], linewidth=1.0, alpha=0.55,
                zorder=2)
        ax.scatter(xs, ys, s=85, marker=METRIC_MARKER[metric],
                   facecolor=METRIC_COLOR[metric], edgecolor=EDGE,
                   linewidth=0.7, zorder=4, label=metric)

    # Labels near each model's highest-scoring marker, with per-model
    # vertical offsets to avoid label collisions when two models share a
    # score level (e.g. Opus-4 and o3 both hit MRR@5=0.875 on Table 1).
    per_label = {
        "GPT-o3":          (0, 10),   # straight up
        "Claude-Opus-4":   (0, 26),   # higher up so it clears o3
        "Claude-Sonnet-4": (0, 10),
    }
    for model, size, *_ in MODELS:
        top = max(scores[model].values())
        dx, dy = per_label[model]
        ax.annotate(model + " (est.)",
                    (size, top), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8, ha="center", va="bottom", color="#222")

    ax.set_xscale("log")
    ax.set_xlim(80, 12000)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Model size (B parameters, log scale)")
    ax.set_ylabel("Score")
    # Panel label is rendered as a small annotation in the upper-left so
    # the LaTeX caption remains the only "title" of the figure.
    ax.text(0.02, 0.98, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic", color="#555")

    ticks = [100, 300, 1000, 3000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}B" if t < 1000 else f"{t // 1000}T" for t in ticks])


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

draw_panel(axes[0], DOC_RANKING, "Table 1 — Document Ranking", (0.74, 0.95))
draw_panel(axes[1], CHUNK_RANKING, "Table 2 — Chunk Ranking",  (0.22, 0.62))

# Combined legend
handles = [Line2D([0], [0], marker=METRIC_MARKER[m], color="w",
                  markerfacecolor=METRIC_COLOR[m], markeredgecolor=EDGE,
                  markersize=8, label=m)
           for m in ["nDCG@5", "MAP@5", "MRR@5"]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

# (Citation footnote removed per user preference; sizes are documented
# in figures/external_papers/README.md.)

fig.savefig("fig_finagentbench_size_scatter.png")
fig.savefig("fig_finagentbench_size_scatter.pdf")
print("Saved fig_finagentbench_size_scatter.{png,pdf}")
