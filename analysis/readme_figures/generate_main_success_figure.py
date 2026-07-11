from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("main_success_vs_lewm.png")

TASKS = ["Reacher", "Cube Single", "Cube Double", "Cube Triple"]
METHODS = ["LeWM", "K1", "K2", "K3", "K4", "Dynamic"]
MEANS = [
    [81.3, 84.0, 82.0, 83.3, 82.7, 85.3],
    [72.0, 79.3, 78.7, 78.0, 78.0, 79.3],
    [74.7, 72.0, 72.0, 72.7, 72.0, 74.0],
    [74.0, 74.0, 74.0, 73.3, 74.0, 77.3],
]
STDS = [
    [4.2, 5.3, 2.0, 1.2, 1.2, 4.2],
    [12.0, 8.1, 7.6, 8.7, 8.7, 8.1],
    [7.6, 3.5, 5.3, 5.0, 5.3, 3.5],
    [8.0, 4.0, 0.0, 5.0, 6.0, 7.6],
]
DYNAMIC_DEPTHS = [1.03, 1.00, 1.26, 1.22]
COLORS = ["#6B7280", "#2563EB", "#0F9D8B", "#E69F00", "#D55E5E", "#7C3AED"]

WIDTH, HEIGHT = 1800, 920
LEFT, RIGHT, TOP, BOTTOM = 190, 70, 190, 150
Y_MIN, Y_MAX = 50.0, 100.0


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def y_px(value):
    plot_height = HEIGHT - TOP - BOTTOM
    return TOP + (Y_MAX - value) / (Y_MAX - Y_MIN) * plot_height


image = Image.new("RGB", (WIDTH, HEIGHT), "white")
draw = ImageDraw.Draw(image)

title_font = font(39, bold=True)
subtitle_font = font(21)
axis_font = font(20)
label_font = font(18, bold=True)
small_font = font(16)

title = "Dynamic K versus LeWM and every fixed refinement depth"
subtitle = "Mean success across 3 paired train/eval seeds; error bars show seed standard deviation"
draw.text((WIDTH / 2, 42), title, fill="#111827", font=title_font, anchor="ma")
draw.text((WIDTH / 2, 96), subtitle, fill="#526079", font=subtitle_font, anchor="ma")

legend_y = 142
legend_width = 0
for method in METHODS:
    legend_width += 24 + 10 + draw.textlength(method, font=axis_font) + 34
legend_x = (WIDTH - legend_width) / 2
for method, color in zip(METHODS, COLORS):
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), radius=4, fill=color)
    draw.text((legend_x + 34, legend_y + 12), method, fill="#263247", font=axis_font, anchor="lm")
    legend_x += 24 + 10 + draw.textlength(method, font=axis_font) + 34

plot_bottom = HEIGHT - BOTTOM
plot_right = WIDTH - RIGHT
for tick in range(50, 101, 10):
    y = y_px(tick)
    draw.line((LEFT, y, plot_right, y), fill="#D7DEE8", width=2)
    draw.text((LEFT - 22, y), str(tick), fill="#526079", font=axis_font, anchor="rm")

draw.line((LEFT, TOP, LEFT, plot_bottom), fill="#4B5563", width=2)
draw.line((LEFT, plot_bottom, plot_right, plot_bottom), fill="#4B5563", width=2)
draw.text((18, (TOP + plot_bottom) / 2), "Success rate (%)", fill="#111827", font=axis_font, anchor="lm")

group_width = (plot_right - LEFT) / len(TASKS)
bar_width = 46
bar_gap = 9
bars_width = len(METHODS) * bar_width + (len(METHODS) - 1) * bar_gap

for task_index, task in enumerate(TASKS):
    group_center = LEFT + group_width * (task_index + 0.5)
    start_x = group_center - bars_width / 2
    for method_index, (method, color) in enumerate(zip(METHODS, COLORS)):
        mean = MEANS[task_index][method_index]
        std = STDS[task_index][method_index]
        x0 = start_x + method_index * (bar_width + bar_gap)
        x1 = x0 + bar_width
        top = y_px(mean)
        draw.rounded_rectangle((x0, top, x1, plot_bottom), radius=5, fill=color)

        err_top = y_px(min(Y_MAX, mean + std))
        err_bottom = y_px(max(Y_MIN, mean - std))
        mid_x = (x0 + x1) / 2
        draw.line((mid_x, err_top, mid_x, err_bottom), fill="#111827", width=2)
        draw.line((mid_x - 8, err_top, mid_x + 8, err_top), fill="#111827", width=2)
        draw.line((mid_x - 8, err_bottom, mid_x + 8, err_bottom), fill="#111827", width=2)

        value_label = f"{mean:g}"
        if method == "Dynamic":
            value_label = f"{mean:g}\nK={DYNAMIC_DEPTHS[task_index]:.2f}"
        box = draw.multiline_textbbox((0, 0), value_label, font=small_font, spacing=2, align="center")
        label_height = box[3] - box[1]
        label_y = max(TOP + 2, err_top - label_height - 8)
        draw.multiline_text((mid_x, label_y), value_label, fill="#111827", font=small_font, spacing=2, anchor="ma", align="center")

    draw.text((group_center, plot_bottom + 44), task, fill="#111827", font=label_font, anchor="ma")

draw.text(
    (WIDTH / 2, HEIGHT - 48),
    "Bars report seed means; individual seeds contain 50 evaluation episodes.",
    fill="#526079",
    font=small_font,
    anchor="ma",
)

image.save(OUT, optimize=True)
print(OUT)
