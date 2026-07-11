from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("main_success_vs_lewm.png")

METHODS = ["LeWM", "K1", "K2", "K3", "K4", "Dynamic"]
COLORS = ["#6B7280", "#2563EB", "#0F9D8B", "#E69F00", "#D55E5E", "#7C3AED"]

MAIN_TASKS = ["Reacher", "Cube Single", "Cube Double", "Cube Triple"]
MAIN_MEANS = [
    [81.3, 84.0, 82.0, 83.3, 82.7, 85.3],
    [72.0, 79.3, 78.7, 78.0, 78.0, 79.3],
    [74.7, 72.0, 72.0, 72.7, 72.0, 74.0],
    [74.0, 74.0, 74.0, 73.3, 74.0, 77.3],
]
MAIN_STDS = [
    [4.2, 5.3, 2.0, 1.2, 1.2, 4.2],
    [12.0, 8.1, 7.6, 8.7, 8.7, 8.1],
    [7.6, 3.5, 5.3, 5.0, 5.3, 3.5],
    [8.0, 4.0, 0.0, 5.0, 6.0, 7.6],
]
MAIN_DYNAMIC_DEPTHS = [1.03, 1.00, 1.26, 1.22]

PUSHT_TASKS = ["PushT H=10\ngoal offset=50"]
PUSHT_MEANS = [[11.3, 13.3, 14.7, 15.3, 12.7, 17.3]]
PUSHT_STDS = [[5.0, 6.1, 6.4, 8.1, 8.3, 7.6]]
PUSHT_DYNAMIC_DEPTHS = [1.02]

WIDTH, HEIGHT = 2000, 980
TOP, BOTTOM = 210, 155


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


image = Image.new("RGB", (WIDTH, HEIGHT), "white")
draw = ImageDraw.Draw(image)

title_font = font(40, bold=True)
subtitle_font = font(22)
panel_font = font(23, bold=True)
axis_font = font(19)
label_font = font(18, bold=True)
small_font = font(15)

draw.text(
    (WIDTH / 2, 42),
    "Dynamic K versus LeWM and every fixed refinement depth",
    fill="#111827",
    font=title_font,
    anchor="ma",
)
draw.text(
    (WIDTH / 2, 98),
    "Mean success across 3 paired train/eval seeds; error bars show seed standard deviation",
    fill="#526079",
    font=subtitle_font,
    anchor="ma",
)

legend_y = 146
legend_width = sum(24 + 10 + draw.textlength(m, font=axis_font) + 34 for m in METHODS)
legend_x = (WIDTH - legend_width) / 2
for method, color in zip(METHODS, COLORS):
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), radius=4, fill=color)
    draw.text((legend_x + 34, legend_y + 12), method, fill="#263247", font=axis_font, anchor="lm")
    legend_x += 24 + 10 + draw.textlength(method, font=axis_font) + 34


def draw_panel(
    *,
    left,
    right,
    title,
    tasks,
    means,
    stds,
    dynamic_depths,
    y_min,
    y_max,
    ticks,
    bar_width,
    bar_gap,
    show_ylabel=False,
):
    plot_bottom = HEIGHT - BOTTOM
    plot_height = plot_bottom - TOP

    def y_px(value):
        return TOP + (y_max - value) / (y_max - y_min) * plot_height

    draw.text(((left + right) / 2, TOP - 24), title, fill="#263247", font=panel_font, anchor="ms")
    for tick in ticks:
        y = y_px(tick)
        draw.line((left, y, right, y), fill="#D7DEE8", width=2)
        draw.text((left - 16, y), str(tick), fill="#526079", font=axis_font, anchor="rm")
    draw.line((left, TOP, left, plot_bottom), fill="#4B5563", width=2)
    draw.line((left, plot_bottom, right, plot_bottom), fill="#4B5563", width=2)
    if show_ylabel:
        draw.text((18, (TOP + plot_bottom) / 2), "Success rate (%)", fill="#111827", font=axis_font, anchor="lm")

    group_width = (right - left) / len(tasks)
    bars_width = len(METHODS) * bar_width + (len(METHODS) - 1) * bar_gap
    for task_index, task in enumerate(tasks):
        group_center = left + group_width * (task_index + 0.5)
        start_x = group_center - bars_width / 2
        for method_index, (method, color) in enumerate(zip(METHODS, COLORS)):
            mean = means[task_index][method_index]
            std = stds[task_index][method_index]
            x0 = start_x + method_index * (bar_width + bar_gap)
            x1 = x0 + bar_width
            top = y_px(mean)
            draw.rounded_rectangle((x0, top, x1, plot_bottom), radius=4, fill=color)

            err_top = y_px(min(y_max, mean + std))
            err_bottom = y_px(max(y_min, mean - std))
            mid_x = (x0 + x1) / 2
            draw.line((mid_x, err_top, mid_x, err_bottom), fill="#111827", width=2)
            draw.line((mid_x - 7, err_top, mid_x + 7, err_top), fill="#111827", width=2)
            draw.line((mid_x - 7, err_bottom, mid_x + 7, err_bottom), fill="#111827", width=2)

            value_label = f"{mean:g}"
            if method == "Dynamic":
                value_label = f"{mean:g}\nK={dynamic_depths[task_index]:.2f}"
            box = draw.multiline_textbbox((0, 0), value_label, font=small_font, spacing=2, align="center")
            label_height = box[3] - box[1]
            label_y = max(TOP + 2, err_top - label_height - 7)
            draw.multiline_text(
                (mid_x, label_y),
                value_label,
                fill="#111827",
                font=small_font,
                spacing=2,
                anchor="ma",
                align="center",
            )

        draw.multiline_text(
            (group_center, plot_bottom + 44),
            task,
            fill="#111827",
            font=label_font,
            spacing=3,
            anchor="ma",
            align="center",
        )


draw_panel(
    left=190,
    right=1510,
    title="Four primary tasks",
    tasks=MAIN_TASKS,
    means=MAIN_MEANS,
    stds=MAIN_STDS,
    dynamic_depths=MAIN_DYNAMIC_DEPTHS,
    y_min=50,
    y_max=100,
    ticks=range(50, 101, 10),
    bar_width=38,
    bar_gap=6,
    show_ylabel=True,
)
draw_panel(
    left=1630,
    right=1950,
    title="Long-horizon setting",
    tasks=PUSHT_TASKS,
    means=PUSHT_MEANS,
    stds=PUSHT_STDS,
    dynamic_depths=PUSHT_DYNAMIC_DEPTHS,
    y_min=0,
    y_max=30,
    ticks=range(0, 31, 5),
    bar_width=38,
    bar_gap=6,
)

draw.line((1568, TOP - 12, 1568, HEIGHT - BOTTOM + 20), fill="#C6CEDA", width=2)
draw.text(
    (WIDTH / 2, HEIGHT - 40),
    "Bars report seed means; individual seeds contain 50 evaluation episodes. PushT uses a separate y-axis.",
    fill="#526079",
    font=small_font,
    anchor="ma",
)

image.save(OUT, optimize=True)
print(OUT)
