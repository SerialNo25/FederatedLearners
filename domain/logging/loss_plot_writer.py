"""Dependency-free SVG writer for federated loss curves."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path


class LossPlotWriter:
    """Writes a basic SVG chart for train and validation loss over rounds."""

    @classmethod
    def write(
        cls,
        output_path: Path,
        rounds: Sequence[int],
        train_losses: Sequence[float],
        val_losses: Sequence[float],
    ) -> Path:
        svg = _build_line_chart_svg(
            title="Federated Loss Over Rounds",
            x_label="Round",
            y_label="Loss",
            rounds=rounds,
            series=(
                ("Train Loss", "#1d4ed8", train_losses),
                ("Validation Loss", "#dc2626", val_losses),
            ),
        )
        output_path.write_text(svg, encoding="utf-8")
        return output_path


def _build_line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    rounds: Sequence[int],
    series: Sequence[tuple[str, str, Sequence[float]]],
) -> str:
    width = 800
    height = 480
    left = 70
    right = 30
    top = 60
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_values = [value for _, _, values in series for value in values]
    if not rounds or not all_values:
        return _empty_chart_svg(width=width, height=height, title=title)

    min_round = min(rounds)
    max_round = max(rounds)
    min_value = min(all_values)
    max_value = max(all_values)

    if min_round == max_round:
        max_round = min_round + 1
    if min_value == max_value:
        padding = 1.0 if min_value == 0 else abs(min_value) * 0.1
        min_value -= padding
        max_value += padding
    else:
        padding = (max_value - min_value) * 0.1
        min_value -= padding
        max_value += padding

    def x_pos(round_index: int) -> float:
        return left + ((round_index - min_round) / (max_round - min_round)) * plot_width

    def y_pos(value: float) -> float:
        return top + (1 - ((value - min_value) / (max_value - min_value))) * plot_height

    grid_lines: list[str] = []
    y_ticks = 5
    for tick_index in range(y_ticks + 1):
        ratio = tick_index / y_ticks
        y = top + ratio * plot_height
        value = max_value - ratio * (max_value - min_value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#374151">{value:.4f}</text>'
        )

    x_labels = []
    for round_index in rounds:
        x = x_pos(round_index)
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle" '
            f'font-size="11" fill="#374151">{round_index}</text>'
        )

    plotted_series = []
    legend_items = []
    legend_x = left
    legend_y = 28
    for label, color, values in series:
        points = " ".join(
            f"{x_pos(round_index):.2f},{y_pos(value):.2f}"
            for round_index, value in zip(rounds, values)
        )
        markers = "".join(
            f'<circle cx="{x_pos(round_index):.2f}" cy="{y_pos(value):.2f}" r="3.2" fill="{color}" />'
            for round_index, value in zip(rounds, values)
        )
        plotted_series.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />'
            + markers
        )
        legend_items.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3" />'
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-size="12" fill="#111827">'
            f"{escape(label)}</text>"
        )
        legend_x += 150

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{width / 2:.2f}" y="18" text-anchor="middle" font-size="18" fill="#111827">{escape(title)}</text>',
            *legend_items,
            *grid_lines,
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="1.5" />',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="1.5" />',
            *x_labels,
            *plotted_series,
            f'<text x="{width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#111827">{escape(x_label)}</text>',
            f'<text x="20" y="{height / 2:.2f}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 20 {height / 2:.2f})">{escape(y_label)}</text>',
            "</svg>",
        ]
    )


def _empty_chart_svg(*, width: int, height: int, title: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{width / 2:.2f}" y="24" text-anchor="middle" font-size="18" fill="#111827">{escape(title)}</text>',
            f'<text x="{width / 2:.2f}" y="{height / 2:.2f}" text-anchor="middle" font-size="14" fill="#6b7280">No data available</text>',
            "</svg>",
        ]
    )
