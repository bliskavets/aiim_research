"""
Stitch the FinOpsBench-v1 and FinOpsBench-v2 pipeline overviews into
a single vertical figure: v1 on top, v2 on the bottom, separated by a
thin divider and short subtitle for each half.

Inputs (must exist before running this script):
  fig_v1_pipeline_overview.png
  fig_v2_pipeline_overview.png

Outputs:
  fig_combined_pipeline.png  (vertical stack, transparent background
                              preserved if present)
  fig_combined_pipeline.pdf  (same content, via matplotlib for vector
                              fidelity of the source rasters)

Usage:
    cd papers/emnlp2026/figures
    python3 make_v1_pipeline_overview.py
    python3 make_v2_pipeline_overview.py
    python3 make_combined_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

V1 = HERE / "fig_v1_pipeline_overview.png"
V2 = HERE / "fig_v2_pipeline_overview.png"
OUT_PNG = HERE / "fig_combined_pipeline.png"
OUT_PDF = HERE / "fig_combined_pipeline.pdf"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common DejaVu paths; fall back to the bitmap default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def combine_png():
    a = Image.open(V1).convert("RGBA")
    b = Image.open(V2).convert("RGBA")

    # Match widths by rescaling the narrower image up; preserve aspect.
    if a.size[0] != b.size[0]:
        target_w = max(a.size[0], b.size[0])
        def rescale(img):
            scale = target_w / img.size[0]
            return img.resize((target_w, int(img.size[1] * scale)),
                              Image.LANCZOS)
        a, b = rescale(a), rescale(b)

    width = a.size[0]
    band_h = 130          # subtitle band above each pipeline
    gap_h  = 22           # blank gap separating the two pipelines
    total_h = band_h + a.size[1] + gap_h + band_h + b.size[1]

    canvas = Image.new("RGBA", (width, total_h), (255, 255, 255, 255))

    title_font = _load_font(58)
    sub_font   = _load_font(28)

    draw = ImageDraw.Draw(canvas)

    def banner(y_top: int, title: str, subtitle: str, color: tuple):
        # Title.
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) / 2, y_top + 18), title,
                  font=title_font, fill=color)
        # Subtitle.
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox[2] - bbox[0]
        draw.text(((width - sw) / 2, y_top + 85), subtitle,
                  font=sub_font, fill=(100, 116, 139, 255))

    # v1 banner + image.
    banner(0,
           "FinOpsBench-v1 construction pipeline",
           "9 stages · 12 seed queries → 5,979 examples in the final dataset",
           (17, 24, 39, 255))
    canvas.paste(a, (0, band_h), a)

    # v2 banner + image.
    y_off = band_h + a.size[1] + gap_h
    banner(y_off,
           "FinOpsBench-v2 construction pipeline",
           "9 stages · 1,247 FinQA items → 1,108 examples in the final dataset",
           (17, 24, 39, 255))
    canvas.paste(b, (0, y_off + band_h), b)

    canvas.save(OUT_PNG, optimize=True)
    print(f"Saved {OUT_PNG.name}  ({width}x{total_h} px)")


def combine_pdf():
    """Emit a vector-friendly PDF by re-pasting via matplotlib so the
    file is usable as a paper-figure target if desired later."""
    img = Image.open(OUT_PNG)
    w_px, h_px = img.size
    dpi = 220
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(OUT_PDF, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {OUT_PDF.name}")


if __name__ == "__main__":
    if not (V1.exists() and V2.exists()):
        raise SystemExit(
            "Missing input(s). Render the per-pipeline figures first:\n"
            "  python3 make_v1_pipeline_overview.py\n"
            "  python3 make_v2_pipeline_overview.py")
    combine_png()
    combine_pdf()
