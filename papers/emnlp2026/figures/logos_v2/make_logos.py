"""
Standalone transparent-PNG icons for the FinOpsBench-v2 pipeline.

Eight icons total: some reused from the v1 logos set (database, robot,
gear-check) and three new ones authored here (python, toolbox,
document). Each PNG is 600 dpi with a transparent background.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch,
)
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent

# Reuse the v1 draw_icon to recover database, robot, etc. without
# re-implementing them. Import is via the sibling pipeline module.
sys.path.insert(0, str(HERE.parent))
from make_v1_pipeline_overview import draw_icon as v1_draw_icon  # noqa: E402


# ---------------------------------------------------------------------------
# New v2-only icon glyphs.
# Each draws centred at (cx, cy), fits in a ~0.55x0.55 region.
# ---------------------------------------------------------------------------

def draw_python(ax, cx: float, cy: float, color: str):
    """Stylised code prompt: ">_" plus a couple of code lines."""
    R = 0.22
    lw = 1.7
    # ">" symbol
    ax.plot([cx - R * 1.05, cx - R * 0.55, cx - R * 1.05],
            [cy + R * 0.45, cy, cy - R * 0.45],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round")
    # Cursor underscore.
    ax.plot([cx - R * 0.35, cx + R * 0.30],
            [cy - R * 0.45, cy - R * 0.45],
            color=color, lw=lw + 0.3, solid_capstyle="round")
    # Two horizontal code lines (lighter).
    for dy, length in [(R * 0.35, 1.20), (R * 0.05, 0.90)]:
        ax.plot([cx + R * 0.05, cx + R * 0.05 + length * R],
                [cy + dy, cy + dy], color=color, lw=lw * 0.7, alpha=0.7)


def draw_toolbox(ax, cx: float, cy: float, color: str):
    """Toolbox with a handle on top and a faint divider."""
    R = 0.22
    lw = 1.7
    # Handle (small arc).
    ax.add_patch(mpatches.Arc((cx, cy + R * 0.65), R * 0.95, R * 0.55,
                              theta1=0, theta2=180, ec=color, lw=lw))
    # Box (trapezoid wider at base).
    box_top_w = 1.7 * R
    box_bot_w = 1.95 * R
    box_h = 1.05 * R
    top_y = cy + R * 0.40
    bot_y = top_y - box_h
    polygon = Polygon([
        (cx - box_top_w / 2, top_y),
        (cx + box_top_w / 2, top_y),
        (cx + box_bot_w / 2, bot_y),
        (cx - box_bot_w / 2, bot_y),
    ], closed=True, ec=color, fc="none", lw=lw)
    ax.add_patch(polygon)
    # Latch / divider line.
    ax.plot([cx - box_top_w / 2 + R * 0.10, cx + box_top_w / 2 - R * 0.10],
            [top_y - R * 0.35, top_y - R * 0.35],
            color=color, lw=lw * 0.7, alpha=0.7)
    # Latch knob (small circle at centre).
    ax.add_patch(Circle((cx, top_y - R * 0.35), R * 0.10,
                        ec=color, fc=color, lw=0))


def draw_document(ax, cx: float, cy: float, color: str):
    """Sheet of paper with folded top-right corner and lines of text."""
    R = 0.22
    lw = 1.6
    w = 1.45 * R
    h = 1.8 * R
    fold = 0.45 * R
    # Body: page with a folded corner.
    body = Polygon([
        (cx - w / 2, cy + h / 2),
        (cx + w / 2 - fold, cy + h / 2),
        (cx + w / 2,        cy + h / 2 - fold),
        (cx + w / 2,        cy - h / 2),
        (cx - w / 2,        cy - h / 2),
    ], closed=True, ec=color, fc="none", lw=lw)
    ax.add_patch(body)
    # Fold triangle.
    fold_tri = Polygon([
        (cx + w / 2 - fold, cy + h / 2),
        (cx + w / 2 - fold, cy + h / 2 - fold),
        (cx + w / 2,        cy + h / 2 - fold),
    ], closed=True, ec=color, fc="none", lw=lw)
    ax.add_patch(fold_tri)
    # Three lines of text.
    text_left = cx - w / 2 + R * 0.20
    text_right = cx + w / 2 - R * 0.22
    for dy in (0.35, 0.05, -0.25):
        right = text_right if dy < -0.20 else text_right - R * 0.30
        ax.plot([text_left, right], [cy + dy * R, cy + dy * R],
                color=color, lw=lw * 0.7, alpha=0.7,
                solid_capstyle="round")


# ---------------------------------------------------------------------------
# Render to standalone PNGs.
# ---------------------------------------------------------------------------

NEW_ICONS = [
    ("python",      "#6d28d9", draw_python),
    ("toolbox",     "#c2410c", draw_toolbox),
    ("document",    "#a16207", draw_document),
]

# Re-export a small set of v1 icons that v2 also uses, so the user has a
# single folder with every v2-relevant logo.
V1_REUSE = [
    ("database_v2", "#1d4ed8", "schema"),    # data store
    ("robot_v2",    "#047857", "robot"),     # runnable agent
    ("loop_v2",     "#4b5563", "loop_big"),  # validation/retry badge
]


def _save(kind: str, color: str, out_path: Path, draw_fn):
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_fn(ax, 0, 0, color)
    fig.savefig(out_path, dpi=600, bbox_inches="tight",
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    for kind, color, fn in NEW_ICONS:
        _save(kind, color, HERE / f"icon_{kind}.png", fn)
    for name, color, v1_kind in V1_REUSE:
        _save(name, color, HERE / f"icon_{name}.png",
              lambda ax, x, y, c, k=v1_kind: v1_draw_icon(ax, k, x, y, c))
    total = len(NEW_ICONS) + len(V1_REUSE)
    print(f"\n{total} icons saved to {HERE}/")


if __name__ == "__main__":
    main()
