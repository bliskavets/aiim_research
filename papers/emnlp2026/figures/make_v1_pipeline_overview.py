"""
FinOpsBench-v1 construction-pipeline overview figure.

Each of the 9 stages is drawn as a rounded panel with:
  - a coloured title bar (stage number + name)
  - a 1-2 line description of what happens in the stage
  - a sizeable corner icon glyph that suggests the stage's role

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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.dpi": 450,        # high-res so user can rescale freely
    "savefig.bbox": "tight",
})


# (number, title, body, icon, fill, header)
STAGES = {
    1: ("Query generation",
        "12 seed queries are\nexpanded into 10,000\ndiverse questions",
        "seed_to_many", "#dbeafe", "#1d4ed8"),
    2: ("Schema generation",
        "An LLM proposes a\nplausible per-example\ndata schema",
        "schema",       "#dbeafe", "#1d4ed8"),
    3: ("Data generation",
        "Realistic rows + distractor\nrows + intended answer\nare drafted",
        "rows_big",     "#dbeafe", "#1d4ed8"),
    4: ("Execution-based\nvalidation",
        "Schema and data are\nexecuted against SQLite;\nerrors fixed in-loop",
        "gear_check",   "#d1fae5", "#047857"),
    5: ("Agent trace\ngeneration",
        "An agent answers the\nquery with the structured-\ndata tool (6 rounds)",
        "robot",        "#ede9fe", "#6d28d9"),
    6: ("Committee\njudgement",
        "Three LLM judges rate\nthe example on 5 criteria;\n2/3 majority required",
        "scales",       "#ffedd5", "#c2410c"),
    7: ("Feedback\nreconciliation",
        "Judges' critiques are\naggregated into a single\nactionable revision plan",
        "merge_big",    "#fef3c7", "#a16207"),
    8: ("Feedback\napplication",
        "Agent re-runs the example\nwith feedback in context",
        "loop_big",     "#fef3c7", "#a16207"),
    9: ("Second\njudgement",
        "Improved example is\nre-judged; survivors\nproceed to filtering",
        "scales",       "#ffedd5", "#c2410c"),
}


# Lanes (y-coordinates) — pulled closer so there's less whitespace
# between rows but panels still don't touch (gap ≈ 0.5 below).
LANE_GEN  = 6.4
LANE_GATE = 4.4
LANE_LOOP = 2.4

# Grid columns shared with the top row.
COL_A = 1.6   # seed
COL_B = 3.8   # stage 1
COL_C = 6.3   # stage 2
COL_D = 8.8   # stage 3
COL_E = 11.3  # stage 4

# Per request: Committee judgement and Agent trace generation must occupy
# the columns of Schema generation (COL_C) and Data generation (COL_D).
NODES = {
    "seed":   (COL_A, LANE_GEN,  "circle"),
    1:        (COL_B, LANE_GEN,  "panel"),
    2:        (COL_C, LANE_GEN,  "panel"),
    3:        (COL_D, LANE_GEN,  "panel"),
    4:        (COL_E, LANE_GEN,  "panel"),
    5:        (COL_D, LANE_GATE, "panel"),  # was off to the right
    6:        (COL_C, LANE_GATE, "panel"),  # was at COL_D
    7:        (COL_B, LANE_LOOP, "panel"),
    8:        (COL_D, LANE_LOOP, "panel"),
    9:        (11.4,  LANE_LOOP, "panel"),
    "filter": (14.0,  LANE_LOOP, "panel_small"),
    "end":    (14.0,  0.2,       "pill"),
}

PANEL_W = 2.2
PANEL_H = 1.50          # tall enough to fit 2-line title bars when needed
PANEL_SMALL_W = 1.8


# ---------------------------------------------------------------------------
# Bigger, more recognisable icon glyphs.
# All draw centred at (cx, cy) and fit in ~0.55x0.55.
# ---------------------------------------------------------------------------

def draw_icon(ax, kind: str, cx: float, cy: float, color: str):
    R = 0.22  # nominal half-size — sized to fit comfortably in panel corner
    lw = 1.6

    if kind == "seed_to_many":
        # One seed dot on the left, three dots fanned out on the right.
        ax.scatter([cx - R * 0.85], [cy], color=color, s=42, zorder=3)
        for dy in (-R * 0.7, 0, R * 0.7):
            ax.scatter([cx + R * 0.85], [cy + dy], color=color,
                       s=28, zorder=3)
            ax.annotate("", xy=(cx + R * 0.7, cy + dy),
                        xytext=(cx - R * 0.6, cy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    elif kind == "schema":
        # Database cylinder + a few rows underneath (suggestive of schema+rows).
        # Cylinder.
        ax.add_patch(mpatches.Ellipse((cx, cy + R * 0.65), 1.6 * R, 0.40 * R,
                                      ec=color, fc="none", lw=lw))
        ax.plot([cx - R * 0.8, cx - R * 0.8], [cy + R * 0.65, cy - R * 0.10],
                color=color, lw=lw)
        ax.plot([cx + R * 0.8, cx + R * 0.8], [cy + R * 0.65, cy - R * 0.10],
                color=color, lw=lw)
        ax.add_patch(mpatches.Arc((cx, cy - R * 0.10), 1.6 * R, 0.40 * R,
                                  theta1=180, theta2=360, ec=color, lw=lw))
        # Two horizontal lines on the side of the cylinder.
        for dy in (0.30, 0.10):
            ax.add_patch(mpatches.Arc(
                (cx, cy + R * dy), 1.6 * R, 0.40 * R,
                theta1=190, theta2=350, ec=color, lw=0.9, alpha=0.6))

    elif kind == "rows_big":
        # Stack of three rounded rows.
        for dy in (0.50, 0.10, -0.30):
            ax.add_patch(FancyBboxPatch(
                (cx - R * 0.85, cy + R * dy - R * 0.10),
                1.7 * R, 0.22 * R,
                boxstyle="round,pad=0,rounding_size=0.04",
                ec=color, fc="none", lw=lw))

    elif kind == "gear_check":
        # Gear silhouette (octagon with notches) + check mark.
        # Outer gear circle.
        ax.add_patch(Circle((cx - R * 0.35, cy), R * 0.55, ec=color,
                            fc="none", lw=lw))
        # Eight gear teeth as small radial lines.
        import math
        for k in range(8):
            ang = k * math.pi / 4
            x0 = cx - R * 0.35 + R * 0.55 * math.cos(ang)
            y0 = cy + R * 0.55 * math.sin(ang)
            x1 = cx - R * 0.35 + R * 0.78 * math.cos(ang)
            y1 = cy + R * 0.78 * math.sin(ang)
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw)
        # Check mark on the right.
        cx2 = cx + R * 0.55
        ax.plot([cx2 - R * 0.30, cx2 - R * 0.05, cx2 + R * 0.40],
                [cy + R * 0.05, cy - R * 0.30, cy + R * 0.40],
                color=color, lw=lw + 0.6, solid_capstyle="round")

    elif kind == "robot":
        # Robot head (rounded square with eyes, mouth, antenna and arms).
        # Antenna.
        ax.plot([cx, cx], [cy + R * 0.70, cy + R * 1.05], color=color, lw=lw)
        ax.scatter([cx], [cy + R * 1.1], color=color, s=30, zorder=4)
        # Head.
        ax.add_patch(FancyBboxPatch(
            (cx - R * 0.85, cy - R * 0.30), 1.7 * R, 1.0 * R,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            ec=color, fc="none", lw=lw))
        # Eyes.
        ax.add_patch(Circle((cx - R * 0.35, cy + R * 0.25), R * 0.13,
                            ec=color, fc=color, lw=0))
        ax.add_patch(Circle((cx + R * 0.35, cy + R * 0.25), R * 0.13,
                            ec=color, fc=color, lw=0))
        # Mouth.
        ax.plot([cx - R * 0.30, cx + R * 0.30],
                [cy - R * 0.05, cy - R * 0.05], color=color, lw=lw,
                solid_capstyle="round")
        # Side antennae (or "arms").
        ax.plot([cx - R * 0.85, cx - R * 1.10],
                [cy + R * 0.35, cy + R * 0.55], color=color, lw=lw)
        ax.plot([cx + R * 0.85, cx + R * 1.10],
                [cy + R * 0.35, cy + R * 0.55], color=color, lw=lw)

    elif kind == "scales":
        # Scales of justice — base, pole, beam, two pans.
        # Base.
        ax.plot([cx - R * 0.55, cx + R * 0.55],
                [cy - R * 0.85, cy - R * 0.85], color=color, lw=lw + 1)
        # Pole.
        ax.plot([cx, cx], [cy - R * 0.85, cy + R * 0.70], color=color, lw=lw)
        # Beam.
        ax.plot([cx - R * 0.85, cx + R * 0.85],
                [cy + R * 0.70, cy + R * 0.70], color=color, lw=lw)
        # Hanging strings to pans.
        ax.plot([cx - R * 0.85, cx - R * 0.85],
                [cy + R * 0.70, cy + R * 0.20], color=color, lw=lw * 0.7)
        ax.plot([cx + R * 0.85, cx + R * 0.85],
                [cy + R * 0.70, cy + R * 0.20], color=color, lw=lw * 0.7)
        # Pans (small arcs / shallow bowls).
        ax.add_patch(mpatches.Arc((cx - R * 0.85, cy + R * 0.20),
                                  R * 0.85, R * 0.50,
                                  theta1=180, theta2=360, ec=color, lw=lw))
        ax.add_patch(mpatches.Arc((cx + R * 0.85, cy + R * 0.20),
                                  R * 0.85, R * 0.50,
                                  theta1=180, theta2=360, ec=color, lw=lw))

    elif kind == "merge_big":
        # Three arrows merging into one (funnel shape).
        for dy in (-R * 0.75, 0, R * 0.75):
            ax.annotate("", xy=(cx + R * 0.15, cy),
                        xytext=(cx - R * 0.85, cy + dy),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=lw, shrinkA=2, shrinkB=2))
        ax.annotate("", xy=(cx + R * 1.05, cy), xytext=(cx + R * 0.20, cy),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw + 0.4, shrinkA=2, shrinkB=0))

    elif kind == "loop_big":
        # Circular refresh-style arrow.
        ax.add_patch(mpatches.Arc((cx, cy), R * 1.7, R * 1.7,
                                  theta1=30, theta2=330,
                                  ec=color, lw=lw + 0.4))
        # Arrowhead at end of arc.
        import math
        end_angle = math.radians(30)
        ex = cx + (R * 0.85) * math.cos(end_angle)
        ey = cy + (R * 0.85) * math.sin(end_angle)
        ax.annotate("",
                    xy=(ex + R * 0.05, ey + R * 0.18),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw + 0.4,
                                    mutation_scale=14))


# ---------------------------------------------------------------------------
# Panel + arrow drawing.
# ---------------------------------------------------------------------------

ICON_INSET_X = 0.30  # icon-centre distance from panel-right
ICON_INSET_Y = 0.30  # icon-centre distance from panel-bottom


def draw_panel(ax, x, y, w, h, title, body, icon_kind, fill, header_color,
               small=False):
    # Two-line titles need a taller header strip.
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
            fontsize=8.5 if small else 9.2, color="white",
            weight="bold", zorder=4)

    # Body text — anchored top-left of the body area. Icons removed by
    # request; user will overlay their own. Body text is centred a bit
    # lower so the body area doesn't look top-heavy when empty of glyphs.
    ax.text(x, y - title_h / 2 + 0.04,
            body, ha="center", va="center",
            fontsize=7.8, color="#1f2937", zorder=4)


def draw_stage_panel(ax, num):
    x, y, _ = NODES[num]
    title, body, icon, fill, header = STAGES[num]
    draw_panel(ax, x, y, PANEL_W, PANEL_H, f"{num}. {title}",
               body, icon, fill, header)


def draw_circle_node(ax, key, label, color="#0f172a"):
    x, y, _ = NODES[key]
    ax.add_patch(Circle((x, y), 0.55, ec=color, fc="#f8fafc",
                        lw=1.8, zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.3, color=color, weight="bold", zorder=5)


def draw_pill_node(ax, key, label, color="#047857"):
    x, y, _ = NODES[key]
    # Wider pill so the "(5,979 examples)" string fits comfortably.
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
        arrowstyle=style, color=color, lw=lw,
        linestyle=ls,
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
# Build the figure.  Wrapped in main() so importing this module from
# logos/make_logos.py doesn't trigger a render side-effect.
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(16.0, 7.6))
    ax.set_xlim(-0.1, 16.7)
    ax.set_ylim(-1.0, 7.5)
    ax.axis("off")

    # Lane backdrops.
    def lane_band(y0, y1, color):
        ax.add_patch(Rectangle((0.0, y0), 16.6, y1 - y0,
                               ec="none", fc=color, alpha=0.06, zorder=0))

    lane_band(LANE_GEN - 0.85,  LANE_GEN + 0.95,  "#3b82f6")
    lane_band(LANE_GATE - 0.85, LANE_GATE + 0.85, "#f97316")
    lane_band(LANE_LOOP - 0.85, LANE_LOOP + 0.85, "#eab308")

    # Phase labels in the left gutter, rotated.
    ax.text(0.05, LANE_GEN,  "Generation (1–5)",
            ha="left", va="center", fontsize=8.2,
            color="#1e3a8a", style="italic", weight="bold",
            rotation=90, zorder=1)
    ax.text(0.05, LANE_GATE, "Committee (6)",
            ha="left", va="center", fontsize=8.2,
            color="#9a3412", style="italic", weight="bold",
            rotation=90, zorder=1)
    ax.text(0.05, LANE_LOOP, "Improvement (7–9)",
            ha="left", va="center", fontsize=8.2,
            color="#854d0e", style="italic", weight="bold",
            rotation=90, zorder=1)

    # (Figure title removed per request — the caption in the paper will
    # carry that context.)

    # Nodes.
    draw_circle_node(ax, "seed", "Seed\nqueries\n(12)")
    for n in range(1, 10):
        draw_stage_panel(ax, n)

    # Final filtering panel.
    draw_panel(ax,
               *NODES["filter"][:2],
               PANEL_SMALL_W, PANEL_H * 0.72,
               "Final filtering",
               "Answer-match and\ntool-use checks",
               "gear_check", "#d1fae5", "#047857", small=True)

    draw_pill_node(ax, "end", "Final dataset (5,979 examples)")

    # ----- Arrows -----

    # Forward generation pipeline.
    arrow(ax, "seed", 1)
    arrow(ax, 1, 2)
    arrow(ax, 2, 3)
    arrow(ax, 3, 4)

    # Stage 4 (top, col E) -> Stage 5 (middle, col D).
    arrow(ax, 4, 5, rad=-0.20)

    # Stage 5 (col D) -> Stage 6 (col C). Same lane, leftward.
    arrow(ax, 5, 6)

    # Stage 6 -> Stage 7 (rejected branch, red dashed).
    arrow(ax, 6, 7, color="#b91c1c", ls=(0, (5, 3)),
          label="rejected", label_pos=0.535, label_dx=-0.18, label_dy=0.06)

    # Improvement loop.
    arrow(ax, 7, 8)
    arrow(ax, 8, 9)

    # Stage 9 -> Final filtering (accepted, green). Stage 9 at x=11.4 lane3,
    # Filter at x=14.0 lane3, so a short rightward arrow. No "accepted" label
    # here; the corresponding label sits on the long L-shape from Stage 6 below.
    arrow(ax, 9, "filter", color="#047857")

    # Stage 6 -> Final filtering (accepted, green). Route via a hand-built
    # L-shape that goes down out of Stage 6 then right under lane 3 to Filter.
    # (Matplotlib's arc3 alone can't avoid the Stage-5/Stage-8/9 panels cleanly.)
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    elbow_y = 1.10  # horizontal segment between lane 3 (y=2.4) and Final-dataset pill (y=0.2)
    x6_, y6_, _ = NODES[6]
    xf_, yf_, _ = NODES["filter"]
    # Curved corners via small Bezier between the straight runs.
    vertices = [
        (x6_, y6_ - PANEL_H / 2),       # exit bottom of Stage 6
        (x6_, elbow_y + 0.25),          # straight down
        (x6_, elbow_y),                 # rounded corner
        (x6_ + 0.25, elbow_y),
        (xf_ - 0.25, elbow_y),          # straight right
        (xf_, elbow_y),                 # rounded corner
        (xf_, elbow_y + 0.25),
        (xf_, yf_ - PANEL_H * 0.36 / 2 - 0.05),  # arrive at bottom of Filter panel
    ]
    codes = [Path.MOVETO, Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO]
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, ec="#047857", fc="none", lw=2.4,
                           capstyle="round", joinstyle="round", zorder=1.5))
    # Arrowhead on the final upward segment.
    ax.annotate("",
                xy=(xf_, yf_ - PANEL_H * 0.36 / 2 - 0.02),
                xytext=(xf_, elbow_y + 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#047857",
                                lw=2.4, mutation_scale=22,
                                shrinkA=0, shrinkB=0),
                zorder=1.5)
    # "accepted" label on the horizontal segment.
    ax.text((x6_ + xf_) / 2, elbow_y + 0.20, "accepted",
            ha="center", va="bottom", fontsize=8, color="#047857",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none",
                      alpha=0.95), zorder=3)

    # Filter (lane3) -> Final dataset pill (bottom).
    arrow(ax, "filter", "end", color="#047857")

    # Drop note for second-time rejections. Place it above Stage 9 (between
    # lanes 2 and 3) so it never collides with the L-shaped accepted arrow
    # that runs below lane 3.
    x9, y9, _ = NODES[9]
    ax.text(x9, y9 + PANEL_H / 2 + 0.18,
            "(second-time rejections dropped)",
            fontsize=8, color="#b91c1c", style="italic",
            ha="center", va="bottom")

    # Legend strip at the bottom.
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
    ax.plot([9.7, 10.3], [ly, ly], color="#b91c1c", lw=2.2,
            linestyle=(0, (5, 3)))
    ax.text(10.4, ly, "rejected branch", ha="left", va="center",
            fontsize=8.5, color="#1f2937")
    ax.plot([12.2, 12.8], [ly, ly], color="#047857", lw=2.2)
    ax.text(12.9, ly, "accepted branch", ha="left", va="center",
            fontsize=8.5, color="#1f2937")

    fig.savefig("fig_v1_pipeline_overview.png")
    fig.savefig("fig_v1_pipeline_overview.pdf")
    print("Saved fig_v1_pipeline_overview.{png,pdf}")


if __name__ == "__main__":
    main()
