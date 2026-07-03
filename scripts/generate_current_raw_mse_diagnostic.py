#!/usr/bin/env python3
"""Generate current-checkpoint raw target-MSE diagnostic tables and figures."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "analysis" / "k_refinement_rel00005_current"
README_FIG_DIR = ROOT / "analysis" / "readme_figures"
PAPER_FIG_DIR = ROOT / "analysis" / "paper1_figures" / "png_direct"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    csv_name: str
    lewm: float
    fixed_k1: float
    fixed_k4: float


DATASETS = [
    DatasetSpec(
        "reacher",
        "Reacher",
        "reacher_rel00005_current_rows.csv",
        80.0,
        80.0,
        82.0,
    ),
    DatasetSpec(
        "cube_single",
        "Cube Single",
        "cube_single_rel00005_current_rows.csv",
        72.0,
        84.0,
        82.0,
    ),
    DatasetSpec(
        "cube_double",
        "Cube Double",
        "cube_double_rel00005_current_rows.csv",
        66.0,
        70.0,
        68.0,
    ),
    DatasetSpec(
        "cube_triple",
        "Cube Triple",
        "cube_triple_rel00005_current_rows.csv",
        74.0,
        74.0,
        74.0,
    ),
]


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
            continue
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUBTITLE = font(18)
F_AXIS = font(17)
F_LABEL = font(18)
F_SMALL = font(14)
F_SMALL_BOLD = font(14, True)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_rows(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(IN_DIR / spec.csv_name)
    df["k1_success"] = bool_series(df["k1_success"])
    df["k4_success"] = bool_series(df["k4_success"])
    df["improvement_abs"] = df["latent_mse_k1"] - df["latent_mse_k4"]
    return df


def pct(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def depth(value: float) -> str:
    return f"K{value:.2f}"


def diagnostic_sweep(df: pd.DataFrame) -> pd.DataFrame:
    positives = sorted({float(v) for v in df["improvement_abs"] if float(v) > 0.0})
    thresholds = [0.0]
    thresholds.extend(positives)
    if positives:
        thresholds.append(max(positives) + 1e-12)
    rows = []
    for tol in thresholds:
        select_k4 = df["improvement_abs"] > tol
        success = np.where(select_k4, df["k4_success"], df["k1_success"]).mean() * 100.0
        mean_k = 1.0 + 3.0 * select_k4.mean()
        beneficial = (~df["k1_success"]) & df["k4_success"]
        selected_beneficial = (select_k4 & beneficial).sum()
        selected = int(select_k4.sum())
        rows.append(
            {
                "tolerance": tol,
                "success": float(success),
                "mean_k": float(mean_k),
                "selected_k4": selected,
                "beneficial_total": int(beneficial.sum()),
                "beneficial_selected": int(selected_beneficial),
                "precision": float(selected_beneficial / selected) if selected else math.nan,
                "recall": float(selected_beneficial / beneficial.sum()) if beneficial.sum() else math.nan,
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["success", "mean_k", "selected_k4"])
    return out.sort_values(["mean_k", "success"]).reset_index(drop=True)


def best_row_text(sweep: pd.DataFrame) -> tuple[float, float, float, str]:
    best_success = float(sweep["success"].max())
    tied = sweep[np.isclose(sweep["success"], best_success)]
    min_k = float(tied["mean_k"].min())
    max_k = float(tied["mean_k"].max())
    if abs(min_k - max_k) < 0.005:
        text = f"{pct(best_success)}@{depth(min_k)}"
    else:
        text = f"{pct(best_success)}@{depth(min_k)} to {depth(max_k)}"
    return best_success, min_k, max_k, text


def outcome_upper_bound(df: pd.DataFrame) -> tuple[float, float, int]:
    beneficial = (~df["k1_success"]) & df["k4_success"]
    success = (df["k1_success"] | df["k4_success"]).mean() * 100.0
    mean_k = 1.0 + 3.0 * beneficial.mean()
    return float(success), float(mean_k), int(beneficial.sum())


def write_summary() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summary_rows = []
    sweeps = {}
    frames = {}
    for spec in DATASETS:
        df = load_rows(spec)
        sweep = diagnostic_sweep(df)
        sweeps[spec.key] = sweep
        frames[spec.key] = df
        best_success, min_k, max_k, best_text = best_row_text(sweep)
        upper_success, upper_k, helped = outcome_upper_bound(df)
        summary_rows.append(
            {
                "dataset": spec.label,
                "lewm": spec.lewm,
                "fixed_k1": spec.fixed_k1,
                "fixed_k4": spec.fixed_k4,
                "mse_diagnostic": best_text,
                "mse_diagnostic_success": best_success,
                "mse_diagnostic_mean_k_min": min_k,
                "mse_diagnostic_mean_k_max": max_k,
                "outcome_upper_bound": f"{pct(upper_success)}@{depth(upper_k)}",
                "outcome_upper_bound_success": upper_success,
                "outcome_upper_bound_mean_k": upper_k,
                "k1_fail_k4_success": helped,
                "n": len(df),
            }
        )
        sweep.insert(0, "dataset", spec.label)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(IN_DIR / "raw_mse_current_summary.csv", index=False)
    pd.concat(sweeps.values(), ignore_index=True).to_csv(
        IN_DIR / "raw_mse_current_sweep.csv", index=False
    )
    return summary, sweeps, frames


def draw_axes(draw: ImageDraw.ImageDraw, box, xlim, ylim, xlabel, ylabel):
    x0, y0, x1, y1 = box
    axis = "#475569"
    grid = "#e2e8f0"
    draw.line((x0, y1, x1, y1), fill=axis, width=2)
    draw.line((x0, y0, x0, y1), fill=axis, width=2)
    for tick in np.linspace(ylim[0], ylim[1], 6):
        y = y1 - (tick - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=grid, width=1)
        draw.text((x0 - 12, y - 8), f"{tick:.0f}", font=F_SMALL, fill="#475569", anchor="ra")
    for tick in [1.0, 2.0, 3.0, 4.0]:
        x = x0 + (tick - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        draw.line((x, y1, x, y1 + 6), fill=axis, width=2)
        draw.text((x, y1 + 12), f"{tick:.0f}", font=F_SMALL, fill="#475569", anchor="mt")
    draw.text(((x0 + x1) / 2, y1 + 48), xlabel, font=F_AXIS, fill="#334155", anchor="mm")
    draw.text((x0 - 78, (y0 + y1) / 2), ylabel, font=F_AXIS, fill="#334155", anchor="mm")

    def project(x, y):
        px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        py = y1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        return px, py

    return project


def line_plot(summary: pd.DataFrame, sweeps: dict[str, pd.DataFrame]):
    README_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((750, 54), "Current rel00005 target-MSE diagnostic", font=F_TITLE, fill="#0f172a", anchor="mm")
    draw.text((750, 92), "Post-hoc K1/K4 choice: select K4 only when target-latent MSE improves enough", font=F_SUBTITLE, fill="#64748b", anchor="mm")
    box = (150, 160, 1250, 740)
    project = draw_axes(draw, box, (0.95, 4.05), (64, 88), "Mean selected depth", "Success rate (%)")
    colors = {
        "reacher": "#2563eb",
        "cube_single": "#14b8a6",
        "cube_double": "#f97316",
        "cube_triple": "#9333ea",
    }
    for spec in DATASETS:
        sw = sweeps[spec.key].sort_values("mean_k")
        pts = [project(float(r.mean_k), float(r.success)) for r in sw.itertuples()]
        if len(pts) > 1:
            draw.line(pts, fill=colors[spec.key], width=4)
        for x, y in pts:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[spec.key])
        fixed1 = project(1.0, spec.fixed_k1)
        fixed4 = project(4.0, spec.fixed_k4)
        draw.rectangle((fixed1[0] - 7, fixed1[1] - 7, fixed1[0] + 7, fixed1[1] + 7), outline=colors[spec.key], width=3)
        draw.rectangle((fixed4[0] - 7, fixed4[1] - 7, fixed4[0] + 7, fixed4[1] + 7), outline=colors[spec.key], width=3)
    lx, ly = 1280, 175
    for i, spec in enumerate(DATASETS):
        y = ly + i * 38
        draw.line((lx, y, lx + 34, y), fill=colors[spec.key], width=5)
        draw.text((lx + 44, y), spec.label, font=F_LABEL, fill="#0f172a", anchor="lm")
    draw.text((1280, 350), "Squares mark fixed K1/K4.", font=F_SMALL, fill="#64748b")
    draw.text((1280, 372), "Lines sweep MSE tolerance.", font=F_SMALL, fill="#64748b")
    out = README_FIG_DIR / "raw_mse_tolerance_pareto.png"
    img.save(out)
    shutil.copy2(out, PAPER_FIG_DIR / out.name)


def precision_recall_plot(frames: dict[str, pd.DataFrame]):
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    draw.text((750, 54), "Can target MSE identify K4-helped episodes?", font=F_TITLE, fill="#0f172a", anchor="mm")
    draw.text((750, 92), "Tolerance 0 diagnostic: select K4 when MSE(K4) < MSE(K1)", font=F_SUBTITLE, fill="#64748b", anchor="mm")
    left, top, right, bottom = 150, 160, 1320, 680
    draw.line((left, bottom, right, bottom), fill="#475569", width=2)
    draw.line((left, top, left, bottom), fill="#475569", width=2)
    for pct_tick in range(0, 101, 20):
        y = bottom - pct_tick / 100 * (bottom - top)
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        draw.text((left - 12, y), str(pct_tick), font=F_SMALL, fill="#475569", anchor="rm")
    width = 58
    group_w = (right - left) / len(DATASETS)
    colors = {"recall": "#16a34a", "precision": "#2563eb", "select_rate": "#ef4444"}
    for i, spec in enumerate(DATASETS):
        df = frames[spec.key]
        select = df["improvement_abs"] > 0.0
        positive = (~df["k1_success"]) & df["k4_success"]
        selected_pos = int((select & positive).sum())
        selected = int(select.sum())
        pos = int(positive.sum())
        vals = [
            selected_pos / pos if pos else math.nan,
            selected_pos / selected if selected else math.nan,
            selected / len(df),
        ]
        cx = left + group_w * (i + 0.5)
        for j, (name, val) in enumerate(zip(["recall", "precision", "select_rate"], vals)):
            x = cx + (j - 1) * (width + 10)
            if math.isnan(val):
                draw.text((x, bottom - 18), "n/a", font=F_SMALL_BOLD, fill="#64748b", anchor="mm")
                continue
            h = val * (bottom - top)
            draw.rounded_rectangle((x - width/2, bottom - h, x + width/2, bottom), radius=5, fill=colors[name])
            draw.text((x, bottom - h - 10), f"{val*100:.0f}", font=F_SMALL_BOLD, fill="#0f172a", anchor="mm")
        draw.text((cx, bottom + 28), spec.label, font=F_LABEL, fill="#0f172a", anchor="mm")
    lx, ly = 1010, 150
    legend = [("recall", "Recall of helped cases"), ("precision", "Precision among K4 selections"), ("select_rate", "Overall K4 selection rate")]
    for i, (name, text) in enumerate(legend):
        y = ly + i * 32
        draw.rounded_rectangle((lx, y, lx + 26, y + 18), radius=4, fill=colors[name])
        draw.text((lx + 36, y + 9), text, font=F_SMALL, fill="#0f172a", anchor="lm")
    out = README_FIG_DIR / "raw_mse_precision_recall_failure.png"
    img.save(out)
    shutil.copy2(out, PAPER_FIG_DIR / out.name)


def cube_triple_confusion(frames: dict[str, pd.DataFrame]):
    df = frames["cube_triple"]
    select = df["improvement_abs"] > 0.0
    categories = [
        ("Beneficial: K1 fail, K4 success", (~df["k1_success"]) & df["k4_success"], "#16a34a"),
        ("Harmful: K1 success, K4 fail", df["k1_success"] & (~df["k4_success"]), "#ef4444"),
        ("Redundant: both success", df["k1_success"] & df["k4_success"], "#2563eb"),
        ("Insufficient: both fail", (~df["k1_success"]) & (~df["k4_success"]), "#64748b"),
    ]
    rows = []
    for label, mask, _ in categories:
        rows.append(
            {
                "category": label,
                "selected_k1": int((mask & (~select)).sum()),
                "selected_k4": int((mask & select).sum()),
                "total": int(mask.sum()),
            }
        )
    pd.DataFrame(rows).to_csv(README_FIG_DIR / "cube_triple_raw_mse_allocation_confusion.csv", index=False)
    img = Image.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(img)
    draw.text((750, 54), "Cube Triple target-MSE allocation confusion", font=F_TITLE, fill="#0f172a", anchor="mm")
    draw.text((750, 92), "Current rel00005 checkpoint, tolerance 0", font=F_SUBTITLE, fill="#64748b", anchor="mm")
    x0, y0, x1 = 360, 170, 1280
    max_total = max(r["total"] for r in rows)
    for i, (r, (_, _, color)) in enumerate(zip(rows, categories)):
        y = y0 + i * 115
        draw.text((x0 - 30, y + 22), r["category"], font=F_LABEL, fill="#0f172a", anchor="rm")
        w_total = (r["total"] / max_total) * (x1 - x0)
        w_k4 = (r["selected_k4"] / max_total) * (x1 - x0)
        draw.rounded_rectangle((x0, y, x0 + w_total, y + 44), radius=8, fill="#e2e8f0")
        if r["selected_k4"]:
            draw.rounded_rectangle((x0, y, x0 + w_k4, y + 44), radius=8, fill=color)
        draw.text((x0 + w_total + 12, y + 22), f"{r['selected_k4']}/{r['total']} sent to K4", font=F_SMALL_BOLD, fill="#334155", anchor="lm")
    draw.rounded_rectangle((895, 650, 925, 672), radius=5, fill="#e2e8f0")
    draw.text((935, 661), "selected K1", font=F_SMALL, fill="#334155", anchor="lm")
    for j, c in enumerate(["#16a34a", "#ef4444", "#2563eb", "#64748b"]):
        draw.rounded_rectangle((1045 + j * 18, 650, 1060 + j * 18, 672), radius=4, fill=c)
    draw.text((1130, 661), "selected K4", font=F_SMALL, fill="#334155", anchor="lm")
    out = README_FIG_DIR / "cube_triple_raw_mse_allocation_confusion.png"
    img.save(out)
    shutil.copy2(out, PAPER_FIG_DIR / out.name)


def print_markdown(summary: pd.DataFrame):
    print("| Dataset | Fixed K1 | Fixed K4 | MSE diagnostic | Hindsight K1/K4 chooser | K1 fail / K4 success |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary.itertuples():
        print(
            f"| {row.dataset} | {pct(row.fixed_k1)}@K1.00 | {pct(row.fixed_k4)}@K4.00 | "
            f"{row.mse_diagnostic} | {row.outcome_upper_bound} | "
            f"{int(row.k1_fail_k4_success)} / {int(row.n)} |"
        )


def main():
    summary, sweeps, frames = write_summary()
    line_plot(summary, sweeps)
    precision_recall_plot(frames)
    cube_triple_confusion(frames)
    print_markdown(summary)


if __name__ == "__main__":
    main()
