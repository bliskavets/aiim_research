"""
Accuracy vs model size for the Finance Agent Benchmark paper
(arXiv:2508.00828, "Finance Agent Benchmark: Benchmarking LLMs on
Real-world Financial Research Tasks").

Source data: Table 2 of the paper, "Class-Balanced Accuracy" column
(the metric the paper recommends as most representative). All 22
LLM rows; we exclude the "Expert" human-baseline row.

Parameter counts:
  - Open-weight models use vendor-reported total parameters.
  - Proprietary models use Li (2026), Incompressible Knowledge Probes,
    arXiv:2604.24827. Direct quotes come from the IKP main estimates
    table (page 13). Models that only appear in the IKP extended
    table (pages 50-52) are derived by inverting the paper's
    log-linear calibration (slope = 0.147 / decade, intercept = 13.3,
    anchored on the published GPT-5 / GPT-5-Mini estimates) applied
    to their reported IKP raw scores.

A regression line is fit through ALL models (the user didn't ask to
exclude outliers on this dataset).
"""

from __future__ import annotations
import math
import matplotlib.pyplot as plt
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

# (model, size_B, ci_lo, ci_hi, acc, reasoning, source)
#   source = "open" => vendor-reported total; no CI
#            "ikp"  => Li 2026 estimate (with 3x CI when from the calibration)
MODELS = [
    # Top of the table, descending accuracy.
    ("o3",                          3000, 1000, 8900, 46.8, True,  "ikp"),
    ("Claude 3.7 Sonnet (Think)",    626,  209, 1878, 45.9, True,  "ikp"),
    ("Claude 3.7 Sonnet",            676,  225, 2028, 44.3, False, "ikp"),
    ("o4 Mini",                      480,  160, 1440, 37.3, True,  "ikp"),
    ("Grok 3 Mini High Reason.",     314,  105,  942, 30.9, True,  "ikp"),
    ("Gemini 2.5 Pro",              1200,  387, 3400, 28.4, False, "ikp"),
    ("GPT 4.1",                     2200,  719, 6400, 26.7, False, "ikp"),
    ("Grok 3 Beta",                 2100,  715, 6300, 25.8, False, "ikp"),
    ("o1",                          3500, 1200,10300, 21.4, True,  "ikp"),
    ("GPT 4.1 mini",                 213,   71,  639, 20.3, False, "ikp"),
    ("GPT 4o",                       720,  241, 2100, 20.0, False, "ikp"),
    ("Grok 3 Mini Low Reason.",      314,  105,  942, 17.6, True,  "ikp"),
    ("Gemini 2.0 Flash",             233,   78,  699, 14.4, False, "ikp"),
    ("Claude 3.5 Haiku",             158,   53,  470, 13.1, False, "ikp"),
    ("o3 Mini",                      117,   39,  351, 12.8, True,  "ikp"),
    ("GPT 4o Mini",                   66,   22,  198, 10.8, False, "ikp"),
    ("Mistral Small 3.1",             24, None, None, 10.8, False, "open"),
    ("LLaMA 4 Scout",                109, None, None,  5.8, False, "open"),
    ("Command A",                    111, None, None,  4.6, False, "open"),
    ("LLaMA 4 Maverick",             400, None, None,  3.1, False, "open"),
    ("LLaMA 3.3 70B",                 70, None, None,  2.8, False, "open"),
    ("GPT 4.1 nano",                  30,   10,   90,  2.4, False, "ikp"),
]

EDGE = "#1a1a1a"
COL_REASON = "#2a7a3a"
COL_NON    = "#3b6fb0"

fig, ax = plt.subplots(figsize=(9.5, 5.5))

# CI horizontal bars (proprietary only).
for m, s, lo, hi, acc, reason, src in MODELS:
    if src == "ikp":
        ax.plot([lo, hi], [acc, acc], color="#888", linewidth=0.6,
                alpha=0.35, zorder=1)

# Markers.
for m, s, lo, hi, acc, reason, src in MODELS:
    marker = "o" if src == "open" else "D"
    color = COL_REASON if reason else COL_NON
    ax.scatter([s], [acc], s=80, marker=marker,
               facecolor=color, edgecolor=EDGE, linewidth=0.6, zorder=4)

# Per-model labels — hand-tuned to keep things readable.
label_offsets = {
    "o3":                          ( 1.10, "left",  0),
    "Claude 3.7 Sonnet (Think)":   ( 1.10, "left",  0),
    "Claude 3.7 Sonnet":           ( 0.90, "right", 0),
    "o4 Mini":                     ( 1.10, "left",  0),
    "Grok 3 Mini High Reason.":    ( 1.10, "left",  0),
    "Gemini 2.5 Pro":              ( 0.90, "right", 0),
    "GPT 4.1":                     ( 0.90, "right", 0),
    "Grok 3 Beta":                 ( 1.10, "left",  0),
    "o1":                          ( 1.10, "left",  0),
    "GPT 4.1 mini":                ( 1.10, "left",  0),
    "GPT 4o":                      ( 0.90, "right", 0),
    "Grok 3 Mini Low Reason.":     ( 0.90, "right", 0),
    "Gemini 2.0 Flash":            ( 1.10, "left",  0),
    "Claude 3.5 Haiku":            ( 1.10, "left",  0),
    "o3 Mini":                     ( 0.90, "right", 1.5),
    "GPT 4o Mini":                 ( 1.10, "left",  -1.5),
    "Mistral Small 3.1":           ( 1.10, "left",  0),
    "LLaMA 4 Scout":               ( 0.90, "right", 0),
    "Command A":                   ( 1.10, "left",  0),
    "LLaMA 4 Maverick":            ( 1.10, "left",  0),
    "LLaMA 3.3 70B":               ( 1.10, "left",  0),
    "GPT 4.1 nano":                ( 0.90, "right", -1.4),
}
for m, s, *_, acc, reason, src in MODELS:
    x_mul, ha, dy = label_offsets[m]
    label = m + (" (est.)" if src == "ikp" else "")
    ax.annotate(label, (s * x_mul, acc + dy),
                fontsize=7.5, ha=ha, va="center", color="#222")

# Log-linear fit through ALL points.
xs = np.array([math.log10(s) for _, s, *_ in MODELS])
ys = np.array([acc for *_, acc, _, _ in MODELS])
slope, intercept = np.polyfit(xs, ys, 1)
xx = np.linspace(math.log10(15), math.log10(15000), 100)
ax.plot(10 ** xx, slope * xx + intercept, "--", color="#444",
        linewidth=1.1, alpha=0.7, zorder=2,
        label=f"fit (all): acc = {slope:.1f}·log₁₀(N_B) + {intercept:.1f}")

ax.set_xscale("log")
ax.set_xlim(15, 15000)
ax.set_ylim(0, 55)
ax.set_xlabel("Model size (B parameters, log scale)")
ax.set_ylabel("Class-balanced accuracy (%)")
ax.set_title("Finance Agent Benchmark (arXiv:2508.00828, Table 2) "
             "— accuracy vs model size")

ticks = [20, 50, 100, 300, 1000, 3000, 10000]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t}B" if t < 1000 else f"{t // 1000}T" for t in ticks])

# Legend: four entries (reasoning/non x open/closed).
legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_NON,
           markeredgecolor=EDGE, markersize=8, label="non-reasoning, open-weight"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_REASON,
           markeredgecolor=EDGE, markersize=8, label="reasoning, open-weight"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COL_NON,
           markeredgecolor=EDGE, markersize=8, label="non-reasoning, closed (est.)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COL_REASON,
           markeredgecolor=EDGE, markersize=8, label="reasoning, closed (est.)"),
    Line2D([0], [0], color="#444", linewidth=1.2, linestyle="--",
           label=f"linear fit, all 22 models"),
]
ax.legend(handles=legend, loc="upper left", frameon=False,
          ncol=1, handletextpad=0.6, labelspacing=0.35)

ax.text(0.99, 0.02,
        "Sizes from Li (2026), arXiv:2604.24827 (3× CI shown for closed-source).",
        transform=ax.transAxes, fontsize=7.5, color="#666",
        ha="right", va="bottom", style="italic")

fig.savefig("fig_financeagentbench_size_scatter.png")
fig.savefig("fig_financeagentbench_size_scatter.pdf")
print(f"Fit: acc = {slope:.2f}*log10(N_B) + {intercept:.2f}")
print("Saved fig_financeagentbench_size_scatter.{png,pdf}")
