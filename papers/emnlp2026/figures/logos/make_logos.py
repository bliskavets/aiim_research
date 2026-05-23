"""
Export each of the matplotlib-drawn icon glyphs from the v1 pipeline
overview as a standalone transparent-background PNG, suitable for
manual placement on top of the figure.

The icons live in make_v1_pipeline_overview.py's draw_icon(); we
import that function and call it onto a per-icon transparent figure.

Each PNG is rendered at high resolution (600 dpi) and is uniformly
sized so the user can drop any of them onto the pipeline figure
without rescaling differences.

Usage:
    cd papers/emnlp2026/figures/logos
    python3 make_logos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Import draw_icon from the sibling pipeline-overview script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_v1_pipeline_overview import draw_icon  # noqa: E402

# (icon_kind, suggested colour to render in).
ICONS = [
    ("seed_to_many", "#1d4ed8"),
    ("schema",       "#1d4ed8"),
    ("rows_big",     "#1d4ed8"),
    ("gear_check",   "#047857"),
    ("robot",        "#6d28d9"),
    ("scales",       "#c2410c"),
    ("merge_big",    "#a16207"),
    ("loop_big",     "#a16207"),
]


def export(kind: str, color: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_icon(ax, kind, 0, 0, color)
    fig.savefig(out_path, dpi=600, bbox_inches="tight",
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    for kind, color in ICONS:
        export(kind, color, HERE / f"icon_{kind}.png")
    print(f"\n{len(ICONS)} icons saved to {HERE}/")


if __name__ == "__main__":
    main()
