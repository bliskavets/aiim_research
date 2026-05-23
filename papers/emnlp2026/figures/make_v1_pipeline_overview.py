"""
FinOpsBench-v1 construction-pipeline overview figure.

Each of the 9 stages is drawn as a rounded panel with:
  - a coloured title bar (stage number + name)
  - a 1-2 line description of what happens in the stage
  - a small icon glyph that suggests the stage's role

Stages 1-5 form the linear generation phase; Stage 6 is the committee
gate; Stages 7-8 are the improvement loop; Stage 9 is the second
judgement that re-feeds the committee gate; the surviving items
proceed to final filtering. Arrows are coloured-coded:
  - dark grey: the main forward path
  - red dashed: the "rejected" branch (feedback loop)
  - green:       the "accepted" branch into final filtering
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


# (number, title, body, icon, fill, header)
STAGES = {
    1: ("Query generation",
        "12 seed queries are\nexpanded into 10,000\ndiverse questions",
        "speech",   "#dbeafe", "#1d4ed8"),
    2: ("Schema generation",
        "An LLM proposes a\nplausible per-example\ndata schema",
        "table",    "#dbeafe", "#1d4ed8"),
    3: ("Data generation",
        "Realistic rows + distractor\nrows + intended answer\nare drafted",
        "rows",     "#dbeafe", "#1d4ed8"),
    4: ("Execution-based\nvalidation",
        "Schema and data are\nexecuted against SQLite;\nerrors fixed in-loop",
        "check",    "#d1fae5", "#047857"),
    5: ("Agent trace generation",
        "An agent answers the\nquery with the structured-\ndata tool (6 rounds)",
        "agent",    "#ede9fe", "#6d28d9"),
    6: ("Committee judgement",
        "Three LLM judges rate\nthe example on 5 criteria;\n2/3 majority required",
        "judges",   "#ffedd5", "#c2410c"),
    7: ("Feedback\nreconciliation",
        "Judges' critiques are\naggregated into a single\nactionable revision plan",
        "merge",    "#fef3c7", "#a16207"),
    8: ("Feedback application",
        "Agent re-runs the example\nwith feedback in context",
        "loop",     "#fef3c7", "#a16207"),
    9: ("Second judgement",
        "Improved example is\nre-judged; survivors\nproceed to filtering",
        "judges",   "#ffedd5", "#c2410c"),
}


# Layout. Coordinate system: x in [0, 14], y in [0, 9].
# Three horizontal lanes, each pulled wide. Phase labels go in a left
# gutter that doesn't overlap with any panels.
LANE_GEN = 7.4   # top lane: stages 1-5
LANE_GATE = 4.7  # middle lane: stage 6
LANE_LOOP = 2.0  # bottom lane: stages 7, 8, 9

NODES = {
    "seed":   (1.6, LANE_GEN, "circle"),
    1:        (3.8, LANE_GEN, "panel"),
    2:        (6.3, LANE_GEN, "panel"),
    3:        (8.8, LANE_GEN, "panel"),
    4:        (11.3, LANE_GEN, "panel"),
    5:        (13.5, LANE_GATE, "panel"),
    6:        (8.8, LANE_GATE, "panel"),
    7:        (3.8, LANE_LOOP, "panel"),
    8:        (6.8, LANE_LOOP, "panel"),
    9:        (9.8, LANE_LOOP, "panel"),
    "filter": (13.5, LANE_LOOP, "panel_small"),
    "end":    (13.5, 0.3, "pill"),
}

PANEL_W = 2.2
PANEL_H = 1.30
PANEL_SMALL_W = 1.8


# ---------------------------------------------------------------------------
# Icon glyphs.
# ---------------------------------------------------------------------------

def draw_icon(ax, kind: str, cx: float, cy: float, color: str):
    r = 0.18
    lw = 1.4
    if kind == "speech":
        ax.add_patch(FancyBboxPatch(
            (cx - r, cy - r * 0.7), 2 * r, 1.3 * r,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            ec=color, fc="none", lw=lw))
        ax.plot([cx - r * 0.3, cx - r * 0.55, cx - r * 0.15],
                [cy - r * 0.7, cy - r * 1.15, cy - r * 0.7],
                color=color, lw=lw)
        ax.scatter([cx - r * 0.4, cx, cx + r * 0.4], [cy] * 3,
                   color=color, s=4, zorder=3)
    elif kind == "table":
        ax.add_patch(Rectangle((cx - r, cy - r * 0.7), 2 * r, 1.4 * r,
                               ec=color, fc="none", lw=lw))
        ax.plot([cx, cx], [cy - r * 0.7, cy + r * 0.7], color=color, lw=lw)
        ax.plot([cx - r, cx + r], [cy, cy], color=color, lw=lw)
    elif kind == "rows":
        for dy in (0.18, 0, -0.18):
            ax.add_patch(Rectangle((cx - r, cy + dy * r * 1.2 - r * 0.08),
                                   2 * r, r * 0.16,
                                   ec="none", fc=color))
    elif kind == "check":
        ax.add_patch(Circle((cx, cy), r * 1.05, ec=color, fc="none", lw=lw))
        ax.plot([cx - r * 0.45, cx - r * 0.05, cx + r * 0.55],
                [cy + r * 0.05, cy - r * 0.4, cy + r * 0.4],
                color=color, lw=lw + 0.4, solid_capstyle="round")
    elif kind == "agent":
        ax.add_patch(FancyBboxPatch(
            (cx - r * 0.85, cy - r * 0.6), 1.7 * r, 1.2 * r,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            ec=color, fc="none", lw=lw))
        ax.plot([cx, cx], [cy + r * 0.6, cy + r * 1.0], color=color, lw=lw)
        ax.scatter([cx], [cy + r * 1.05], color=color, s=10, zorder=3)
        ax.scatter([cx - r * 0.35, cx + r * 0.35], [cy + r * 0.1] * 2,
                   color=color, s=8, zorder=3)
    elif kind == "judges":
        for dx in (-0.32, 0.0, 0.32):
            ax.add_patch(Circle((cx + dx * r * 1.4, cy + r * 0.05),
                                r * 0.28, ec=color, fc="none", lw=lw))
            ax.plot([cx + dx * r * 1.4 - r * 0.18, cx + dx * r * 1.4 + r * 0.18],
                    [cy - r * 0.45, cy - r * 0.45],
                    color=color, lw=lw)
            ax.plot([cx + dx * r * 1.4, cx + dx * r * 1.4],
                    [cy - r * 0.2, cy - r * 0.45],
                    color=color, lw=lw)
    elif kind == "merge":
        ax.annotate("", xy=(cx, cy), xytext=(cx - r * 0.9, cy + r * 0.55),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        ax.annotate("", xy=(cx, cy), xytext=(cx - r * 0.9, cy - r * 0.55),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        ax.annotate("", xy=(cx + r * 0.85, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    elif kind == "loop":
        ax.add_patch(mpatches.Arc((cx, cy), r * 1.7, r * 1.7,
                                  theta1=30, theta2=330,
                                  ec=color, lw=lw))
        ax.annotate("", xy=(cx + r * 0.85 * 0.866, cy + r * 0.85 * 0.5),
                    xytext=(cx + r * 0.85, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# ---------------------------------------------------------------------------
# Panel + arrows.
# ---------------------------------------------------------------------------

def draw_panel(ax, x, y, w, h, title, body, icon_kind, fill, header_color,
               small=False):
    title_h = 0.36

    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        ec="#374151", fc=fill, lw=0.8, zorder=2))

    ax.add_patch(Rectangle(
        (x - w / 2, y + h / 2 - title_h),
        w, title_h, ec="none", fc=header_color, zorder=3))
    # Round only the top edge of the header strip.
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y + h / 2 - title_h), w, title_h,
        boxstyle="round,pad=0,rounding_size=0.08",
        ec="none", fc=header_color, lw=0, zorder=2.5))

    ax.text(x, y + h / 2 - title_h / 2,
            title, ha="center", va="center",
            fontsize=8.5 if small else 9.2, color="white",
            weight="bold", zorder=4)

    ax.text(x - w / 2 + 0.16, y - title_h / 2 + 0.04,
            body, ha="left", va="top",
            fontsize=7.6, color="#1f2937", zorder=4)

    draw_icon(ax, icon_kind, x + w / 2 - 0.30, y - title_h / 2 - 0.05,
              header_color)


def draw_stage_panel(ax, num):
    x, y, _ = NODES[num]
    title, body, icon, fill, header = STAGES[num]
    draw_panel(ax, x, y, PANEL_W, PANEL_H, f"{num}. {title}",
               body, icon, fill, header)


def draw_circle_node(ax, key, label, color="#0f172a"):
    x, y, _ = NODES[key]
    ax.add_patch(Circle((x, y), 0.55, ec=color, fc="#f8fafc",
                        lw=1.6, zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.3, color=color, weight="bold", zorder=5)


def draw_pill_node(ax, key, label, color="#047857"):
    x, y, _ = NODES[key]
    ax.add_patch(FancyBboxPatch(
        (x - 1.1, y - 0.30), 2.2, 0.6,
        boxstyle="round,pad=0,rounding_size=0.30",
        ec=color, fc="#ecfdf5", lw=1.6, zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, color=color, weight="bold", zorder=5)


def arrow(ax, a, b, color="#374151", style="-|>", lw=1.5, rad=0.0,
          ls="-", label=None, label_pos=0.5, label_dx=0.0, label_dy=0.18):
    if isinstance(a, tuple):
        ax1, ay1 = a
    else:
        ax1, ay1, _ = NODES[a]
    if isinstance(b, tuple):
        bx1, by1 = b
    else:
        bx1, by1, _ = NODES[b]
    arr = FancyArrowPatch(
        (ax1, ay1), (bx1, by1),
        arrowstyle=style, color=color, lw=lw,
        linestyle=ls,
        shrinkA=22, shrinkB=22,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1.5,
        mutation_scale=14)
    ax.add_patch(arr)
    if label is not None:
        mx = ax1 + (bx1 - ax1) * label_pos + label_dx
        my = ay1 + (by1 - ay1) * label_pos + label_dy
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7.5, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.95),
                zorder=3)


# ---------------------------------------------------------------------------
# Build the figure.
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15.5, 9.5))
ax.set_xlim(-0.1, 16.2)
ax.set_ylim(-1.0, 9.6)
ax.axis("off")

# Phase labels in the left gutter, vertically centred on each lane.
def phase_label(y, text, color):
    ax.text(0.15, y, text, ha="left", va="center",
            fontsize=9.5, color=color, weight="bold", style="italic",
            rotation=90, zorder=1)

# Lane backdrops (very subtle).
def lane_band(y0, y1, color):
    ax.add_patch(Rectangle((0.05, y0), 14.4, y1 - y0,
                           ec="none", fc=color, alpha=0.05, zorder=0))

def lane_band2(y0, y1, color):
    ax.add_patch(Rectangle((0.0, y0), 15.9, y1 - y0,
                           ec="none", fc=color, alpha=0.05, zorder=0))

lane_band2(LANE_GEN - 0.8,  LANE_GEN + 0.8,  "#3b82f6")
lane_band2(LANE_GATE - 0.8, LANE_GATE + 0.8, "#f97316")
lane_band2(LANE_LOOP - 0.8, LANE_LOOP + 0.8, "#eab308")

# Phase labels — rotated, in the left gutter so they never collide with panels.
ax.text(0.05, LANE_GEN,  "Generation (1–5)",
        ha="left", va="center", fontsize=9,
        color="#1e3a8a", style="italic", weight="bold",
        rotation=90, zorder=1)
ax.text(0.05, LANE_GATE, "Committee gate (6)",
        ha="left", va="center", fontsize=9,
        color="#9a3412", style="italic", weight="bold",
        rotation=90, zorder=1)
ax.text(0.05, LANE_LOOP, "Improvement loop (7–9)",
        ha="left", va="center", fontsize=9,
        color="#854d0e", style="italic", weight="bold",
        rotation=90, zorder=1)

# Title (well above the top lane so it doesn't clip Stage 1).
ax.text(7.95, 9.15, "FinOpsBench-v1 construction pipeline",
        ha="center", va="center", fontsize=15.5, weight="bold",
        color="#111827")
ax.text(7.95, 8.70,
        "9 stages · 12 seed queries → 5,979 examples in the final dataset",
        ha="center", va="center", fontsize=10, color="#6b7280")

# Nodes.
draw_circle_node(ax, "seed", "Seed\nqueries\n(12)")
for n in range(1, 10):
    draw_stage_panel(ax, n)

draw_panel(ax,
           *NODES["filter"][:2],
           PANEL_SMALL_W, PANEL_H * 0.62,
           "Final filtering",
           "Answer-match and\ntool-use checks",
           "check", "#d1fae5", "#047857", small=True)

draw_pill_node(ax, "end", "Final dataset (5,979 examples)")

# Arrows — forward generation pipeline.
arrow(ax, "seed", 1)
arrow(ax, 1, 2)
arrow(ax, 2, 3)
arrow(ax, 3, 4)

# Stage 4 -> Stage 5 (down and right).
arrow(ax, 4, 5, rad=-0.25)

# Stage 5 -> Stage 6 (left along the gate lane).
arrow(ax, 5, 6)

# Stage 6 branching.
# Rejected (down to Stage 7), red dashed.
arrow(ax, 6, 7, color="#b91c1c", ls=(0, (4, 2)),
      label="rejected", label_pos=0.45, label_dx=-0.20, label_dy=0.05)

# Improvement loop.
arrow(ax, 7, 8)
arrow(ax, 8, 9)

# Stage 9 -> back into Stage 6 (loop-back arrow).
# We approach Stage 6 from the right side; use a curved path.
arrow(ax, 9, 6, color="#b91c1c", ls=(0, (4, 2)),
      rad=-0.30, label="re-judge", label_pos=0.6,
      label_dx=0.45, label_dy=0.15)

# Stage 6 -> Final filtering (accepted, green).
arrow(ax, 6, "filter", color="#047857", rad=0.30,
      label="accepted", label_pos=0.55, label_dx=0.55, label_dy=-0.05)

# Filter -> end (drop down).
arrow(ax, "filter", "end", color="#047857", rad=0.0)

# Drop note for rejects-after-second-judgement, tucked under Stage 8 where
# there's free space (out of the way of every arrow).
x8, y8, _ = NODES[8]
ax.text(x8, y8 - PANEL_H / 2 - 0.30,
        "(second-time rejections are dropped)",
        fontsize=7.7, color="#b91c1c", style="italic",
        ha="center", va="top")

# Legend strip at the bottom-left.
def legend_chip(x, y, label, color):
    ax.add_patch(Rectangle((x, y - 0.10), 0.30, 0.18,
                           ec="none", fc=color))
    ax.text(x + 0.40, y, label, ha="left", va="center",
            fontsize=8.5, color="#1f2937")

ly = -0.75
legend_chip(0.1,  ly, "generation",   "#1d4ed8")
legend_chip(2.0,  ly, "validation",   "#047857")
legend_chip(3.9,  ly, "agent run",    "#6d28d9")
legend_chip(5.8,  ly, "judgement",    "#c2410c")
legend_chip(7.7,  ly, "improvement",  "#a16207")
ax.plot([9.7, 10.2], [ly, ly], color="#b91c1c", lw=1.6,
        linestyle=(0, (4, 2)))
ax.text(10.3, ly, "rejected branch", ha="left", va="center",
        fontsize=8.5, color="#1f2937")
ax.plot([12.0, 12.5], [ly, ly], color="#047857", lw=1.6)
ax.text(12.6, ly, "accepted branch", ha="left", va="center",
        fontsize=8.5, color="#1f2937")

fig.savefig("fig_v1_pipeline_overview.png")
fig.savefig("fig_v1_pipeline_overview.pdf")
print("Saved fig_v1_pipeline_overview.{png,pdf}")
