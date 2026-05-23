"""
FinOpsBench-v2 construction-pipeline overview figure.

v2 has a strictly linear 9-stage pipeline that builds, augments,
and packages a single FinQA item into a runnable agent task. There
is no improvement loop (unlike v1); each stage is validated by
execution, with up to 10 retries on failure.

Three logical phases follow the paper's §4.2 description:
  Phase A — Basic build      (Stages 1-4):
                              initial solution, backing store,
                              basic tools, basic plan
  Phase B — Augmentation     (Stages 5-7):
                              expanded store with distractors,
                              augmented tools, augmented plan
  Phase C — Packaging        (Stages 8-9):
                              system prompt, runnable agent code

Layout: snake-like across 3 lanes (top → ↓ → middle ← ↓ → bottom →
end), matching the visual rhythm of fig_v1_pipeline_overview.

Icons are deliberately omitted; the user will overlay them. The
companion script figures/logos_v2/make_logos.py exports the per-stage
icon glyphs as standalone transparent PNGs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.dpi": 450,
    "savefig.bbox": "tight",
})


# (number, title, body, icon, fill, header)
# Headers/fills reuse the v1 palette: blue=data, orange=tools,
# purple=code/agent, amber=text/prompt, green=final.
STAGES = {
    1: ("Initial solution",
        "Python solution over\nthe original FinQA tables\nestablishes the logic",
        "python",      "#ede9fe", "#6d28d9"),
    2: ("Backing data\nstore",
        "SQLite tables are\ngenerated to model the\nscenario; hold the answer-\nrelevant records",
        "database",    "#dbeafe", "#1d4ed8"),
    3: ("Basic tools",
        "Tool functions are written\nas abstractions over the\nbacking store",
        "toolbox",     "#ffedd5", "#c2410c"),
    4: ("Correct plan\n(basic)",
        "Reference solution written\nusing only the basic tools",
        "python",      "#ede9fe", "#6d28d9"),
    5: ("Augmented\nbacking store",
        "Store is expanded with\nadditional tables, split\ndata, and distractor rows",
        "database",    "#dbeafe", "#1d4ed8"),
    6: ("Augmented tools",
        "Extended toolkit: core\ntools, partial-information\nchains, distractor tools",
        "toolbox",     "#ffedd5", "#c2410c"),
    7: ("Correct plan\n(augmented)",
        "Reference solution rewritten\nover the augmented tools\n(multi-hop path)",
        "python",      "#ede9fe", "#6d28d9"),
    8: ("System prompt",
        "Scenario description and\ntool definitions; the\nbacking table data is\nexcluded",
        "document",    "#fef3c7", "#a16207"),
    9: ("Agent code",
        "Runnable smolagents agent\nready for evaluation",
        "robot",       "#d1fae5", "#047857"),
}


# Lanes (y-coordinates) — same spacing as the v1 figure.
LANE_TOP = 7.0   # Basic build
LANE_MID = 4.8   # Augmentation
LANE_BOT = 2.6   # Packaging

# Shared 5-column grid.
COL_A = 1.6
COL_B = 4.0
COL_C = 6.5
COL_D = 9.0
COL_E = 11.5

NODES = {
    "start": (COL_A, LANE_TOP, "circle"),
    1:       (COL_B, LANE_TOP, "panel"),
    2:       (COL_C, LANE_TOP, "panel"),
    3:       (COL_D, LANE_TOP, "panel"),
    4:       (COL_E, LANE_TOP, "panel"),
    5:       (COL_E, LANE_MID, "panel"),
    6:       (COL_D, LANE_MID, "panel"),
    7:       (COL_C, LANE_MID, "panel"),
    8:       (COL_C, LANE_BOT, "panel"),
    9:       (COL_D, LANE_BOT, "panel"),
    "end":   (COL_E + 1.3, LANE_BOT, "pill"),
}

PANEL_W = 2.2
PANEL_H = 1.50

# Bottom-right insets (kept for future icon overlay placement notes).
ICON_INSET_X = 0.30
ICON_INSET_Y = 0.30


# ---------------------------------------------------------------------------
# Drawing helpers (panel, circle, pill, arrow).
# ---------------------------------------------------------------------------

def draw_panel(ax, x, y, w, h, title, body, fill, header_color):
    title_h = 0.52 if ("\n" in title) else 0.36

    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        ec="#374151", fc=fill, lw=0.9, zorder=2))

    ax.add_patch(Rectangle(
        (x - w / 2, y + h / 2 - title_h),
        w, title_h, ec="none", fc=header_color, zorder=3))
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y + h / 2 - title_h), w, title_h,
        boxstyle="round,pad=0,rounding_size=0.08",
        ec="none", fc=header_color, lw=0, zorder=2.5))

    ax.text(x, y + h / 2 - title_h / 2,
            title, ha="center", va="center",
            fontsize=9.2, color="white", weight="bold", zorder=4)

    ax.text(x, y - title_h / 2 + 0.04,
            body, ha="center", va="center",
            fontsize=7.8, color="#1f2937", zorder=4)


def draw_stage_panel(ax, num):
    x, y, _ = NODES[num]
    title, body, _icon, fill, header = STAGES[num]
    draw_panel(ax, x, y, PANEL_W, PANEL_H, f"{num}. {title}",
               body, fill, header)


def draw_circle_node(ax, key, label, color="#0f172a"):
    x, y, _ = NODES[key]
    ax.add_patch(Circle((x, y), 0.55, ec=color, fc="#f8fafc",
                        lw=1.8, zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.3, color=color, weight="bold", zorder=5)


def draw_pill_node(ax, key, label, color="#047857"):
    x, y, _ = NODES[key]
    pill_w = 3.3
    pill_h = 0.65
    ax.add_patch(FancyBboxPatch(
        (x - pill_w / 2, y - pill_h / 2), pill_w, pill_h,
        boxstyle="round,pad=0,rounding_size=0.32",
        ec=color, fc="#ecfdf5", lw=1.8, zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9.5, color=color, weight="bold", zorder=5)


def arrow(ax, a, b, color="#1f2937", style="-|>", lw=2.4, rad=0.0,
          ls="-", label=None, label_pos=0.5, label_dx=0.0, label_dy=0.20):
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
        arrowstyle=style, color=color, lw=lw, linestyle=ls,
        shrinkA=24, shrinkB=24,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1.5,
        capstyle="round", joinstyle="round",
        mutation_scale=22)
    ax.add_patch(arr)
    if label is not None:
        mx = ax1 + (bx1 - ax1) * label_pos + label_dx
        my = ay1 + (by1 - ay1) * label_pos + label_dy
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.20", fc="white",
                          ec="none", alpha=0.95),
                zorder=3)


# ---------------------------------------------------------------------------
# Build the figure (wrapped so importing this module doesn't render).
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(16.0, 8.4))
    ax.set_xlim(-0.1, 16.7)
    ax.set_ylim(-1.0, 8.1)
    ax.axis("off")

    # Lane backdrops + left-gutter phase labels (matches v1 styling).
    def lane_band(y0, y1, color):
        ax.add_patch(Rectangle((0.0, y0), 16.6, y1 - y0,
                               ec="none", fc=color, alpha=0.06, zorder=0))

    lane_band(LANE_TOP - 0.85, LANE_TOP + 0.85, "#3b82f6")
    lane_band(LANE_MID - 0.85, LANE_MID + 0.85, "#f97316")
    lane_band(LANE_BOT - 0.85, LANE_BOT + 0.85, "#6d28d9")

    ax.text(0.05, LANE_TOP, "Basic build (1–4)",
            ha="left", va="center", fontsize=9.5,
            color="#1e3a8a", style="italic", weight="bold",
            rotation=90, zorder=1)
    ax.text(0.05, LANE_MID, "Augmentation (5–7)",
            ha="left", va="center", fontsize=9.5,
            color="#9a3412", style="italic", weight="bold",
            rotation=90, zorder=1)
    ax.text(0.05, LANE_BOT, "Packaging (8–9)",
            ha="left", va="center", fontsize=9.5,
            color="#5b21b6", style="italic", weight="bold",
            rotation=90, zorder=1)

    # Nodes.
    draw_circle_node(ax, "start", "FinQA\nitem")
    for n in range(1, 10):
        draw_stage_panel(ax, n)
    draw_pill_node(ax, "end", "Final dataset (1,108 examples)")

    # ----- Arrows (snake topology) -----
    arrow(ax, "start", 1)
    arrow(ax, 1, 2)
    arrow(ax, 2, 3)
    arrow(ax, 3, 4)

    # 4 -> 5: drop straight down from top lane to middle lane.
    arrow(ax, 4, 5)

    # 5 -> 6 -> 7: middle lane flows right-to-left.
    arrow(ax, 5, 6)
    arrow(ax, 6, 7)

    # 7 -> 8: drop straight down from middle to bottom lane.
    arrow(ax, 7, 8)

    # 8 -> 9 -> end: bottom lane flows left-to-right into the pill.
    arrow(ax, 8, 9)
    arrow(ax, 9, "end", color="#047857")

    # Note on per-stage validation.
    ax.text(8.0, -0.20,
            "Each stage is validated by execution; failures are retried "
            "up to 10 times before the example is dropped.",
            ha="center", va="center",
            fontsize=8.5, color="#4b5563", style="italic")

    # Legend strip at the bottom.
    def legend_chip(x, y, label, color):
        ax.add_patch(Rectangle((x, y - 0.10), 0.30, 0.18,
                               ec="none", fc=color))
        ax.text(x + 0.40, y, label, ha="left", va="center",
                fontsize=8.5, color="#1f2937")

    ly = -0.85
    legend_chip(0.1,  ly, "data store",       "#1d4ed8")
    legend_chip(2.2,  ly, "tools",            "#c2410c")
    legend_chip(4.0,  ly, "code (plan/sol.)", "#6d28d9")
    legend_chip(6.7,  ly, "system prompt",    "#a16207")
    legend_chip(9.0,  ly, "runnable agent",   "#047857")
    ax.plot([11.4, 12.0], [ly, ly], color="#1f2937", lw=2.2)
    ax.text(12.1, ly, "forward path", ha="left", va="center",
            fontsize=8.5, color="#1f2937")

    fig.savefig("fig_v2_pipeline_overview.png")
    fig.savefig("fig_v2_pipeline_overview.pdf")
    print("Saved fig_v2_pipeline_overview.{png,pdf}")


if __name__ == "__main__":
    main()
