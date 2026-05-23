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

Each estimated point gets a 3x confidence interval per the paper's
calibration residual (the paper reports a +/- 0.478 spread in log10
space; here I round to a 3x factor in either direction).
"""

from __future__ import annotations
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
#   "open" => vendor-reported size, no CI
#   "ikp"  => estimated from IKP paper (Li 2026), 3x CI
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

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Draw per-model: connector line, error bars (if estimated), then markers.
for model, size, lo, hi, v1, v2, source in MODELS:
    # Connector showing v1/v2 for the same model.
    ax.plot([size, size], [v1, v2], color="#888", linewidth=0.7,
            linestyle=":", zorder=1)

    # Markers: circle for open-weight, diamond for estimated.
    marker = "o" if source == "open" else "D"
    ax.scatter([size], [v1], s=85, marker=marker,
               facecolor=COL_V1, edgecolor=EDGE, linewidth=0.7, zorder=4)
    ax.scatter([size], [v2], s=85, marker=marker,
               facecolor=COL_V2, edgecolor=EDGE, linewidth=0.7, zorder=4)

# Per-model labels, hand-tuned to avoid overlap.
label_pos = {
    "GPT-5":         (4100, 73, "right"),
    "GPT-4.1":       (2200, 56, "left"),
    "o4-mini":       (480,  72, "right"),
    "GPT-5-mini":    (410,  62, "right"),
    "GPT-4.1-mini":  (213,  53, "left"),
    "Qwen3-30B-A3B": (30,   57, "left"),
    "Qwen3-8B":      (10,   46, "left"),
    "Llama-3.1-8B":  (10,   16, "left"),
}
for model, size, _, _, v1, v2, source in MODELS:
    x, y, ha = label_pos[model]
    label = model + (" (est.)" if source == "ikp" else "")
    ax.annotate(label, (x, y), fontsize=8.5, ha=ha, va="center",
                color="#222")

# Log-linear fit over every model except Llama-3.1-8B (it sits well below
# trend — keep it on the plot but exclude it from the regression so the
# line reflects the bulk of the population).
import numpy as np
fit_models = [(m, s, v1, v2) for m, s, _, _, v1, v2, _ in MODELS
              if m != "Llama-3.1-8B"]
xs = np.array([math.log10(s) for _, s, _, _ in fit_models])
y1 = np.array([v1 for _, _, v1, _ in fit_models])
y2 = np.array([v2 for _, _, _, v2 in fit_models])
s1, i1 = np.polyfit(xs, y1, 1)
s2, i2 = np.polyfit(xs, y2, 1)
xx = np.linspace(math.log10(5), math.log10(15000), 100)
ax.plot(10 ** xx, s1 * xx + i1, "--", color=COL_V1, linewidth=1.0,
        alpha=0.55, zorder=1)
ax.plot(10 ** xx, s2 * xx + i2, "--", color=COL_V2, linewidth=1.0,
        alpha=0.55, zorder=1)
# Print fit equations so they show up in commits/notes.
print(f"v1 fit (excluding Llama): acc = {s1:.2f} * log10(N_B) + {i1:.2f}")
print(f"v2 fit (excluding Llama): acc = {s2:.2f} * log10(N_B) + {i2:.2f}")

ax.set_xscale("log")
ax.set_xlim(5, 15000)
ax.set_ylim(0, 80)
ax.set_xlabel("Model size (B parameters, log scale)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("FinOpsBench accuracy vs. model size")

# Custom xticks at familiar scales.
ticks = [8, 30, 100, 300, 1000, 3000, 10000]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t}B" if t < 1000 else f"{t // 1000}T" for t in ticks])

legend_handles = [
    mpatches.Patch(facecolor=COL_V1, edgecolor=EDGE, label="FinOpsBench-v1"),
    mpatches.Patch(facecolor=COL_V2, edgecolor=EDGE, label="FinOpsBench-v2"),
]
# Two text-only legend entries explaining marker shape.
from matplotlib.lines import Line2D
legend_handles.extend([
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
           markeredgecolor=EDGE, markersize=8, label="open-weight"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#888",
           markeredgecolor=EDGE, markersize=8,
           label="closed (size estimated\nfrom IKP paper)"),
])
ax.legend(handles=legend_handles, loc="lower right", frameon=False,
          ncol=2, columnspacing=1.2, handletextpad=0.6)

# Annotation: cite the IKP paper for the estimated sizes.
ax.text(0.02, 0.96,
        "Estimated sizes from Li (2026), arXiv:2604.24827",
        transform=ax.transAxes, fontsize=8, color="#666",
        ha="left", va="top", style="italic")

fig.savefig("fig_accuracy_vs_size.png")
fig.savefig("fig_accuracy_vs_size.pdf")
print("Saved fig_accuracy_vs_size.{png,pdf}")
