#!/usr/bin/env python3
"""Draw the rel00005 learned dynamic-K depth-allocation figure."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "depth_allocation_rel00005.png"
OUT_PDF = ROOT / "figures" / "depth_allocation_rel00005.pdf"

DATA = [
    {
        "name": "Reacher",
        "success": 86,
        "mean_k": 1.08,
        "k": [92.19, 7.65, 0.16, 0.007],
        "deepened": 7.81,
    },
    {
        "name": "Cube Single",
        "success": 84,
        "mean_k": 1.00,
        "k": [99.987, 0.013, 0.0, 0.0],
        "deepened": 0.013,
    },
    {
        "name": "Cube Double",
        "success": 72,
        "mean_k": 1.10,
        "k": [91.20, 7.95, 0.78, 0.070],
        "deepened": 8.80,
    },
    {
        "name": "Cube Triple",
        "success": 78,
        "mean_k": 1.06,
        "k": [95.20, 3.24, 1.50, 0.049],
        "deepened": 4.80,
    },
]

COLORS = {
    "K1": "#2f6ee4",
    "K2": "#23a79a",
    "K3": "#f59e0b",
    "K4": "#ef4444",
    "axis": "#334155",
    "grid": "#e2e8f0",
    "text": "#0f172a",
    "muted": "#64748b",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_PANEL = font(29, True)
F_LABEL = font(26)
F_AXIS = font(20)
F_SMALL = font(19)
F_SMALL_BOLD = font(19, True)


def draw_center(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill=COLORS["text"]):
    draw.text(xy, text, font=fnt, fill=fill, anchor="mm")


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int):
    items = [("K=1", COLORS["K1"]), ("K=2", COLORS["K2"]), ("K=3", COLORS["K3"]), ("K=4", COLORS["K4"])]
    cursor = x
    for label, color in items:
        draw.rounded_rectangle((cursor, y, cursor + 34, y + 18), radius=4, fill=color)
        draw.text((cursor + 44, y + 9), label, font=F_SMALL, fill=COLORS["text"], anchor="lm")
        cursor += 120


def grid(draw: ImageDraw.ImageDraw, box, x_max: float, ticks):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=COLORS["axis"], width=2)
    for tick in ticks:
        x = x0 + tick / x_max * (x1 - x0)
        draw.line((x, y0, x, y1), fill=COLORS["grid"], width=1)
        draw.text((x, y1 + 14), f"{tick:g}", font=F_SMALL, fill=COLORS["muted"], anchor="mt")
    return lambda v: x0 + v / x_max * (x1 - x0)


def draw_stacked_bar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y: int,
    width: int,
    height: int,
    values,
    scale: float,
    include_k1: bool,
):
    keys = ["K1", "K2", "K3", "K4"]
    start = x0
    for key, value in zip(keys, values):
        if key == "K1" and not include_k1:
            continue
        if value <= 0:
            continue
        w = max(1, int(round(width * value / scale)))
        draw.rectangle((start, y, start + w, y + height), fill=COLORS[key])
        start += w


def main():
    img = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(img)

    draw_center(
        draw,
        (900, 72),
        "Learned dynamic K uses extra refinement sparsely",
        F_TITLE,
    )
    draw_center(
        draw,
        (900, 114),
        "Distribution of selected recurrent depth over all imagined CEM transitions",
        F_SMALL,
        COLORS["muted"],
    )
    draw_legend(draw, 660, 150)

    left = (210, 245, 930, 750)
    right = (1100, 245, 1665, 750)
    draw_center(draw, ((left[0] + left[2]) / 2, 215), "All selected depths", F_PANEL)
    draw_center(draw, ((right[0] + right[2]) / 2, 215), "Zoom: transitions refined beyond K=1", F_PANEL)

    lx = grid(draw, left, 100, [0, 20, 40, 60, 80, 100])
    rx = grid(draw, right, 12, [0, 2, 4, 6, 8, 10, 12])
    bar_h = 56
    row_gap = 96
    y_start = 292

    for i, row in enumerate(DATA):
        y = y_start + i * row_gap
        center_y = y + bar_h / 2
        draw.text((left[0] - 28, center_y), row["name"], font=F_LABEL, fill=COLORS["text"], anchor="rm")
        draw_stacked_bar(draw, left[0], y, left[2] - left[0], bar_h, row["k"], 100, include_k1=True)
        draw.text(
            (left[2] - 18, center_y),
            f"{row['success']}%, mean K={row['mean_k']:.2f}",
            font=F_SMALL_BOLD,
            fill="white",
            anchor="rm",
        )

        draw_stacked_bar(draw, right[0], y, right[2] - right[0], bar_h, row["k"], 12, include_k1=False)
        label = f"{row['deepened']:.2f}%"
        if row["deepened"] < 0.05:
            draw.text((right[0] + 18, center_y), label, font=F_SMALL_BOLD, fill=COLORS["muted"], anchor="lm")
        else:
            x_label = min(rx(row["deepened"]) + 14, right[2] - 4)
            draw.text((x_label, center_y), label, font=F_SMALL_BOLD, fill=COLORS["text"], anchor="lm")

    draw_center(draw, ((left[0] + left[2]) / 2, 830), "Fraction of imagined transitions (%)", F_AXIS, COLORS["axis"])
    draw_center(draw, ((right[0] + right[2]) / 2, 830), "Fraction refined beyond K=1 (%)", F_AXIS, COLORS["axis"])

    note = "Takeaway: the learned selector improves success while sending only a small fraction of transitions past K=1."
    draw_center(draw, (900, 910), note, F_SMALL, COLORS["muted"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    img.save(OUT_PDF, "PDF", resolution=180.0)
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
