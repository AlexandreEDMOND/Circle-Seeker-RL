from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ppo import load_checkpoint


PLOT_SPECS = [
    ("mean_return", "Mean return"),
    ("success_rate", "Success rate"),
    ("entropy", "Entropy"),
    ("value_loss", "Value loss"),
    ("policy_loss", "Policy loss"),
    ("approx_kl", "Approx KL"),
    ("clip_fraction", "Clip fraction"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO training metrics from a checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/training_curves.svg"),
        help="SVG output path.",
    )
    return parser.parse_args()


def metric_points(metrics: dict[str, Any]) -> list[dict[str, float]]:
    updates = metrics.get("updates")
    if isinstance(updates, list) and updates:
        return [
            {
                key: float(value)
                for key, value in update.items()
                if isinstance(value, (int, float))
            }
            for update in updates
        ]

    fallback = {
        "global_step": float(metrics.get("timesteps", 0.0)),
    }
    for key, _ in PLOT_SPECS:
        fallback[key] = float(metrics.get(key, 0.0))
    return [fallback]


def build_svg(points: list[dict[str, float]]) -> str:
    panel_width = 380
    panel_height = 220
    margin = 42
    gap = 24
    columns = 2
    rows = (len(PLOT_SPECS) + columns - 1) // columns
    width = columns * panel_width + (columns + 1) * gap
    height = rows * panel_height + (rows + 1) * gap

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#12161c"/>',
    ]
    for index, (key, title) in enumerate(PLOT_SPECS):
        column = index % columns
        row = index // columns
        x = gap + column * (panel_width + gap)
        y = gap + row * (panel_height + gap)
        parts.append(draw_panel(points, key, title, x, y, panel_width, panel_height, margin))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def draw_panel(
    points: list[dict[str, float]],
    key: str,
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
    margin: int,
) -> str:
    xs = [point.get("global_step", float(index)) for index, point in enumerate(points)]
    ys = [point.get(key, 0.0) for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        padding = max(abs(y_min) * 0.1, 1.0)
        y_min -= padding
        y_max += padding

    left = x + margin
    right = x + width - 18
    top = y + 38
    bottom = y + height - margin

    coords = [
        (
            scale(value_x, x_min, x_max, left, right),
            scale(value_y, y_min, y_max, bottom, top),
        )
        for value_x, value_y in zip(xs, ys)
    ]
    point_attr = " ".join(f"{point_x:.1f},{point_y:.1f}" for point_x, point_y in coords)
    escaped_title = html.escape(title)

    panel = [
        f'<g transform="translate(0 0)">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="#1b2129"/>',
        f'<text x="{x + 16}" y="{y + 24}" fill="#ebeef5" font-family="Arial" font-size="15">{escaped_title}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#6d7684" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#6d7684" stroke-width="1"/>',
        f'<text x="{left}" y="{bottom + 22}" fill="#aeb7c4" font-family="Arial" font-size="11">step {x_min:.0f}</text>',
        f'<text x="{right - 58}" y="{bottom + 22}" fill="#aeb7c4" font-family="Arial" font-size="11">step {x_max:.0f}</text>',
        f'<text x="{x + 10}" y="{top + 4}" fill="#aeb7c4" font-family="Arial" font-size="11">{y_max:.3g}</text>',
        f'<text x="{x + 10}" y="{bottom}" fill="#aeb7c4" font-family="Arial" font-size="11">{y_min:.3g}</text>',
    ]
    if len(coords) == 1:
        point_x, point_y = coords[0]
        panel.append(f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="4" fill="#5aa5ff"/>')
    else:
        panel.append(
            f'<polyline points="{point_attr}" fill="none" stroke="#5aa5ff" stroke-width="2.5"/>'
        )
    panel.append("</g>")
    return "\n".join(panel)


def scale(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def main() -> None:
    args = parse_args()
    _, _, metrics = load_checkpoint(args.checkpoint)
    points = metric_points(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(points), encoding="utf-8")
    print(f"Saved training curves: {args.output}")


if __name__ == "__main__":
    main()
