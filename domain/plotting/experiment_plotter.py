"""SVG-based experiment plotting utilities without external plotting dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from statistics import mean

from domain.metrics.evaluation import InstitutionMetrics


@dataclass(frozen=True)
class LineSeries:
    label: str
    x_values: list[float]
    y_values: list[float]
    color: str


@dataclass(frozen=True)
class BoxplotPanel:
    title: str
    y_label: str
    groups: list[tuple[str, list[float], str]]


@dataclass(frozen=True)
class LocalEpochRecord:
    epoch: int
    train_loss: float
    validation: InstitutionMetrics


@dataclass(frozen=True)
class FederatedRoundRecord:
    round_index: int
    global_evaluations: list[InstitutionMetrics]
    local_evaluations: list[InstitutionMetrics]
    local_train_loss: dict[str, float]


class ExperimentPlotter:
    _PALETTE = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#65a30d",
    ]
    _MAX_LINE_POINTS = 1_200
    _MAX_POINT_MARKERS = 240

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def write_local_epoch_plots(
        self,
        institution_id: str,
        records: list[LocalEpochRecord],
    ) -> None:
        if not records:
            return

        epochs = [record.epoch for record in records]
        self._write_line_chart(
            title=f"Loss plot — {institution_id}",
            x_label="Epoch",
            y_label="Loss",
            file_name="loss_plot.svg",
            series=[
                LineSeries("Train loss", epochs, [record.train_loss for record in records], self._PALETTE[0]),
                LineSeries(
                    "Validation loss",
                    epochs,
                    [record.validation.loss for record in records],
                    self._PALETTE[1],
                ),
            ],
        )
        self._write_line_chart(
            title=f"PR-AUC — {institution_id}",
            x_label="Epoch",
            y_label="PR-AUC",
            file_name="pr_auc.svg",
            series=[
                LineSeries(
                    "Validation PR-AUC",
                    epochs,
                    [record.validation.pr_auc for record in records],
                    self._PALETTE[2],
                )
            ],
        )
        self._write_line_chart(
            title=f"F1-score at optimal threshold — {institution_id}",
            x_label="Epoch",
            y_label="Best F1",
            file_name="f1_optimal_threshold.svg",
            series=[
                LineSeries(
                    "Validation best F1",
                    epochs,
                    [record.validation.best_f1 for record in records],
                    self._PALETTE[3],
                )
            ],
        )
        self._write_line_chart(
            title=f"Convergence — {institution_id}",
            x_label="Epoch",
            y_label="Loss",
            file_name="global_vs_local_convergence.svg",
            series=[
                LineSeries("Train loss", epochs, [record.train_loss for record in records], self._PALETTE[0]),
                LineSeries(
                    "Validation loss",
                    epochs,
                    [record.validation.loss for record in records],
                    self._PALETTE[1],
                ),
            ],
        )

    def write_local_summary_plots(
        self,
        institution_id: str,
        records: list[LocalEpochRecord],
    ) -> None:
        if not records:
            return

        final_metrics = records[-1].validation
        self._write_line_chart(
            title=f"Threshold curves — {institution_id}",
            x_label="Threshold",
            y_label="Score",
            file_name="threshold_curves.svg",
            series=[
                LineSeries("Precision", final_metrics.thresholds, final_metrics.precision_curve, self._PALETTE[0]),
                LineSeries("Recall", final_metrics.thresholds, final_metrics.recall_curve, self._PALETTE[1]),
                LineSeries("F1", final_metrics.thresholds, final_metrics.f1_curve, self._PALETTE[2]),
            ],
        )
        self._write_boxplot_panels(
            file_name="per_client_performance_boxplots.svg",
            panels=[
                BoxplotPanel(
                    title="Validation loss",
                    y_label="Loss",
                    groups=[(institution_id, [record.validation.loss for record in records], self._PALETTE[0])],
                ),
                BoxplotPanel(
                    title="Validation PR-AUC",
                    y_label="PR-AUC",
                    groups=[(institution_id, [record.validation.pr_auc for record in records], self._PALETTE[1])],
                ),
                BoxplotPanel(
                    title="Best F1",
                    y_label="F1",
                    groups=[(institution_id, [record.validation.best_f1 for record in records], self._PALETTE[2])],
                ),
            ],
        )

    def write_federated_round_plots(self, records: list[FederatedRoundRecord]) -> None:
        if not records:
            return

        rounds = [record.round_index for record in records]
        client_ids = self._client_ids(records)
        self._write_line_chart(
            title="Loss plot — federated training",
            x_label="Round",
            y_label="Loss",
            file_name="loss_plot.svg",
            series=[
                LineSeries(
                    "Global mean eval loss",
                    rounds,
                    [mean(metric.loss for metric in record.global_evaluations) for record in records],
                    self._PALETTE[0],
                ),
                LineSeries(
                    "Local mean train loss",
                    rounds,
                    [mean(record.local_train_loss.values()) for record in records],
                    self._PALETTE[1],
                ),
            ],
        )
        self._write_line_chart(
            title="PR-AUC — federated training",
            x_label="Round",
            y_label="PR-AUC",
            file_name="pr_auc.svg",
            series=[
                LineSeries(
                    "Global mean PR-AUC",
                    rounds,
                    [mean(metric.pr_auc for metric in record.global_evaluations) for record in records],
                    self._PALETTE[2],
                ),
                LineSeries(
                    "Local mean PR-AUC",
                    rounds,
                    [mean(metric.pr_auc for metric in record.local_evaluations) for record in records],
                    self._PALETTE[3],
                ),
            ],
        )
        self._write_line_chart(
            title="F1-score at optimal threshold — federated training",
            x_label="Round",
            y_label="Best F1",
            file_name="f1_optimal_threshold.svg",
            series=[
                LineSeries(
                    "Global mean best F1",
                    rounds,
                    [mean(metric.best_f1 for metric in record.global_evaluations) for record in records],
                    self._PALETTE[4],
                ),
                LineSeries(
                    "Local mean best F1",
                    rounds,
                    [mean(metric.best_f1 for metric in record.local_evaluations) for record in records],
                    self._PALETTE[5],
                ),
            ],
        )
        convergence_series = [
            LineSeries(
                "Global mean eval loss",
                rounds,
                [mean(metric.loss for metric in record.global_evaluations) for record in records],
                self._PALETTE[0],
            )
        ]
        for index, client_id in enumerate(client_ids, start=1):
            convergence_series.append(
                LineSeries(
                    f"Local {client_id} loss",
                    rounds,
                    [record.local_train_loss[client_id] for record in records],
                    self._color_for_index(index),
                )
            )
        self._write_line_chart(
            title="Global vs local convergence",
            x_label="Round",
            y_label="Loss",
            file_name="global_vs_local_convergence.svg",
            series=convergence_series,
        )

    def write_federated_summary_plots(self, records: list[FederatedRoundRecord]) -> None:
        if not records:
            return

        client_ids = self._client_ids(records)
        final_record = records[-1]
        threshold_grid = self._shared_threshold_grid(
            [*final_record.global_evaluations, *final_record.local_evaluations]
        )
        self._write_line_chart(
            title="Threshold curves — federated training",
            x_label="Threshold",
            y_label="Score",
            file_name="threshold_curves.svg",
            series=[
                LineSeries(
                    "Global precision",
                    threshold_grid,
                    self._mean_curve(final_record.global_evaluations, threshold_grid, "precision_curve"),
                    self._PALETTE[0],
                ),
                LineSeries(
                    "Global recall",
                    threshold_grid,
                    self._mean_curve(final_record.global_evaluations, threshold_grid, "recall_curve"),
                    self._PALETTE[1],
                ),
                LineSeries(
                    "Global F1",
                    threshold_grid,
                    self._mean_curve(final_record.global_evaluations, threshold_grid, "f1_curve"),
                    self._PALETTE[2],
                ),
                LineSeries(
                    "Local F1",
                    threshold_grid,
                    self._mean_curve(final_record.local_evaluations, threshold_grid, "f1_curve"),
                    self._PALETTE[3],
                ),
            ],
        )
        self._write_boxplot_panels(
            file_name="per_client_performance_boxplots.svg",
            panels=[
                BoxplotPanel(
                    title="Global PR-AUC by client",
                    y_label="PR-AUC",
                    groups=[
                        (
                            client_id,
                            [
                                self._metric_for_client(record.global_evaluations, client_id).pr_auc
                                for record in records
                            ],
                            self._color_for_index(index),
                        )
                        for index, client_id in enumerate(client_ids)
                    ],
                ),
                BoxplotPanel(
                    title="Global best F1 by client",
                    y_label="F1",
                    groups=[
                        (
                            client_id,
                            [
                                self._metric_for_client(record.global_evaluations, client_id).best_f1
                                for record in records
                            ],
                            self._color_for_index(index),
                        )
                        for index, client_id in enumerate(client_ids)
                    ],
                ),
                BoxplotPanel(
                    title="Local train loss by client",
                    y_label="Loss",
                    groups=[
                        (
                            client_id,
                            [record.local_train_loss[client_id] for record in records],
                            self._color_for_index(index),
                        )
                        for index, client_id in enumerate(client_ids)
                    ],
                ),
            ],
        )

    @staticmethod
    def _client_ids(records: list[FederatedRoundRecord]) -> list[str]:
        return sorted(
            {
                metric.institution_id
                for record in records
                for metric in [*record.global_evaluations, *record.local_evaluations]
            }
        )

    def _write_line_chart(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        file_name: str,
        series: list[LineSeries],
    ) -> None:
        width = 960
        height = 540
        margin_left = 80
        margin_right = 220
        margin_top = 70
        margin_bottom = 70
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        all_x = [value for line in series for value in line.x_values]
        all_y = [value for line in series for value in line.y_values]
        min_x, max_x = self._axis_bounds(all_x)
        min_y, max_y = self._axis_bounds(all_y, padding=0.08)

        elements = [
            f'<rect width="{width}" height="{height}" fill="#ffffff" />',
            f'<text x="{width / 2}" y="32" font-size="22" text-anchor="middle" fill="#111827">{self._escape(title)}</text>',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="2" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="2" />',
            f'<text x="{margin_left + plot_width / 2}" y="{height - 20}" font-size="16" text-anchor="middle" fill="#111827">{self._escape(x_label)}</text>',
            f'<text x="22" y="{margin_top + plot_height / 2}" font-size="16" text-anchor="middle" fill="#111827" transform="rotate(-90 22 {margin_top + plot_height / 2})">{self._escape(y_label)}</text>',
        ]

        for tick_index in range(5):
            tick_ratio = tick_index / 4
            x_value = min_x + (max_x - min_x) * tick_ratio
            x_coord = margin_left + plot_width * tick_ratio
            y_value = min_y + (max_y - min_y) * (1.0 - tick_ratio)
            y_coord = margin_top + plot_height * tick_ratio
            elements.append(
                f'<line x1="{x_coord}" y1="{margin_top}" x2="{x_coord}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1" />'
            )
            elements.append(
                f'<text x="{x_coord}" y="{margin_top + plot_height + 22}" font-size="12" text-anchor="middle" fill="#374151">{x_value:.2f}</text>'
            )
            elements.append(
                f'<line x1="{margin_left}" y1="{y_coord}" x2="{margin_left + plot_width}" y2="{y_coord}" stroke="#e5e7eb" stroke-width="1" />'
            )
            elements.append(
                f'<text x="{margin_left - 10}" y="{y_coord + 4}" font-size="12" text-anchor="end" fill="#374151">{y_value:.2f}</text>'
            )

        for index, line in enumerate(series):
            line_points = self._prepare_line_points(line)
            points = []
            for point_index, (x_value, y_value) in enumerate(line_points):
                x_coord = self._map_value(x_value, min_x, max_x, margin_left, margin_left + plot_width)
                y_coord = self._map_value(y_value, min_y, max_y, margin_top + plot_height, margin_top)
                points.append(f"{x_coord:.2f},{y_coord:.2f}")
                if self._should_draw_marker(len(line_points), point_index):
                    elements.append(
                        f'<circle cx="{x_coord:.2f}" cy="{y_coord:.2f}" r="3.5" fill="{line.color}" />'
                    )
            elements.append(
                f'<polyline fill="none" stroke="{line.color}" stroke-width="2.5" points="{" ".join(points)}" />'
            )
            legend_y = margin_top + 24 + index * 24
            legend_x = width - margin_right + 24
            elements.append(
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{line.color}" stroke-width="3" />'
            )
            elements.append(
                f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-size="13" fill="#111827">{self._escape(line.label)}</text>'
            )

        self._write_svg(file_name, width, height, elements)

    def _write_boxplot_panels(self, *, file_name: str, panels: list[BoxplotPanel]) -> None:
        width = 1260
        height = 480
        panel_width = 360
        panel_height = 340
        panel_margin_x = 40
        panel_margin_top = 80
        elements = [
            f'<rect width="{width}" height="{height}" fill="#ffffff" />',
            f'<text x="{width / 2}" y="34" font-size="22" text-anchor="middle" fill="#111827">Per-client performance boxplots</text>',
        ]
        for panel_index, panel in enumerate(panels):
            origin_x = panel_margin_x + panel_index * (panel_width + 40)
            origin_y = panel_margin_top
            elements.extend(self._boxplot_panel_elements(panel, origin_x, origin_y, panel_width, panel_height))
        self._write_svg(file_name, width, height, elements)

    def _boxplot_panel_elements(
        self,
        panel: BoxplotPanel,
        origin_x: int,
        origin_y: int,
        panel_width: int,
        panel_height: int,
    ) -> list[str]:
        margin_left = 55
        margin_bottom = 50
        margin_top = 28
        plot_width = panel_width - margin_left - 20
        plot_height = panel_height - margin_top - margin_bottom
        plot_x0 = origin_x + margin_left
        plot_y0 = origin_y + margin_top

        values = [value for _, group_values, _ in panel.groups for value in group_values]
        min_y, max_y = self._axis_bounds(values, padding=0.1)
        elements = [
            f'<rect x="{origin_x}" y="{origin_y}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#d1d5db" stroke-width="1" rx="8" />',
            f'<text x="{origin_x + panel_width / 2}" y="{origin_y + 18}" font-size="16" text-anchor="middle" fill="#111827">{self._escape(panel.title)}</text>',
            f'<line x1="{plot_x0}" y1="{plot_y0 + plot_height}" x2="{plot_x0 + plot_width}" y2="{plot_y0 + plot_height}" stroke="#111827" stroke-width="1.5" />',
            f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_height}" stroke="#111827" stroke-width="1.5" />',
            f'<text x="{origin_x + 20}" y="{origin_y + panel_height / 2}" font-size="13" text-anchor="middle" fill="#111827" transform="rotate(-90 {origin_x + 20} {origin_y + panel_height / 2})">{self._escape(panel.y_label)}</text>',
        ]
        for tick_index in range(5):
            tick_ratio = tick_index / 4
            y_value = min_y + (max_y - min_y) * (1.0 - tick_ratio)
            y_coord = plot_y0 + plot_height * tick_ratio
            elements.append(
                f'<line x1="{plot_x0}" y1="{y_coord}" x2="{plot_x0 + plot_width}" y2="{y_coord}" stroke="#e5e7eb" stroke-width="1" />'
            )
            elements.append(
                f'<text x="{plot_x0 - 8}" y="{y_coord + 4}" font-size="11" text-anchor="end" fill="#374151">{y_value:.2f}</text>'
            )

        slot_width = plot_width / max(len(panel.groups), 1)
        for index, (label, group_values, color) in enumerate(panel.groups):
            center_x = plot_x0 + slot_width * (index + 0.5)
            stats = self._quartiles(group_values)
            min_coord = self._map_value(stats[0], min_y, max_y, plot_y0 + plot_height, plot_y0)
            q1_coord = self._map_value(stats[1], min_y, max_y, plot_y0 + plot_height, plot_y0)
            median_coord = self._map_value(stats[2], min_y, max_y, plot_y0 + plot_height, plot_y0)
            q3_coord = self._map_value(stats[3], min_y, max_y, plot_y0 + plot_height, plot_y0)
            max_coord = self._map_value(stats[4], min_y, max_y, plot_y0 + plot_height, plot_y0)
            box_width = min(42, slot_width * 0.45)
            elements.extend(
                [
                    f'<line x1="{center_x}" y1="{min_coord}" x2="{center_x}" y2="{max_coord}" stroke="#374151" stroke-width="1.5" />',
                    f'<line x1="{center_x - box_width / 2}" y1="{min_coord}" x2="{center_x + box_width / 2}" y2="{min_coord}" stroke="#374151" stroke-width="1.5" />',
                    f'<line x1="{center_x - box_width / 2}" y1="{max_coord}" x2="{center_x + box_width / 2}" y2="{max_coord}" stroke="#374151" stroke-width="1.5" />',
                    f'<rect x="{center_x - box_width / 2}" y="{q3_coord}" width="{box_width}" height="{q1_coord - q3_coord}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.5" />',
                    f'<line x1="{center_x - box_width / 2}" y1="{median_coord}" x2="{center_x + box_width / 2}" y2="{median_coord}" stroke="{color}" stroke-width="2" />',
                    f'<text x="{center_x}" y="{plot_y0 + plot_height + 22}" font-size="11" text-anchor="middle" fill="#111827">{self._escape(label)}</text>',
                ]
            )
        return elements

    def _shared_threshold_grid(self, evaluations: list[InstitutionMetrics]) -> list[float]:
        thresholds = sorted({0.0, 0.5, 1.0, *[value for metric in evaluations for value in metric.thresholds]})
        return thresholds

    def _prepare_line_points(self, line: LineSeries) -> list[tuple[float, float]]:
        paired_points = list(zip(line.x_values, line.y_values))
        if len(paired_points) <= self._MAX_LINE_POINTS:
            return paired_points
        step = max(1, ceil(len(paired_points) / self._MAX_LINE_POINTS))
        sampled_points = paired_points[::step]
        if sampled_points[-1] != paired_points[-1]:
            sampled_points.append(paired_points[-1])
        return sampled_points

    def _should_draw_marker(self, point_count: int, point_index: int) -> bool:
        if point_count <= self._MAX_POINT_MARKERS:
            return True
        marker_step = max(1, ceil(point_count / self._MAX_POINT_MARKERS))
        return point_index % marker_step == 0 or point_index == point_count - 1


    def _mean_curve(
        self,
        evaluations: list[InstitutionMetrics],
        threshold_grid: list[float],
        field_name: str,
    ) -> list[float]:
        interpolated_curves = []
        for evaluation in evaluations:
            source_thresholds = evaluation.thresholds
            source_values = getattr(evaluation, field_name)
            interpolated_curves.append(
                [self._interpolate(source_thresholds, source_values, threshold) for threshold in threshold_grid]
            )
        return [mean(values) for values in zip(*interpolated_curves)]

    @staticmethod
    def _interpolate(x_values: list[float], y_values: list[float], target_x: float) -> float:
        if target_x <= x_values[0]:
            return y_values[0]
        if target_x >= x_values[-1]:
            return y_values[-1]
        for left_index in range(len(x_values) - 1):
            left_x = x_values[left_index]
            right_x = x_values[left_index + 1]
            if left_x <= target_x <= right_x:
                if right_x == left_x:
                    return y_values[left_index]
                ratio = (target_x - left_x) / (right_x - left_x)
                return y_values[left_index] + ratio * (y_values[left_index + 1] - y_values[left_index])
        return y_values[-1]

    @staticmethod
    def _metric_for_client(evaluations: list[InstitutionMetrics], client_id: str) -> InstitutionMetrics:
        for evaluation in evaluations:
            if evaluation.institution_id == client_id:
                return evaluation
        raise KeyError(f"Missing metrics for client '{client_id}'")

    def _color_for_index(self, index: int) -> str:
        return self._PALETTE[index % len(self._PALETTE)]

    @staticmethod
    def _axis_bounds(values: list[float], padding: float = 0.05) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            delta = 1.0 if min_value == 0 else abs(min_value) * 0.1
            return min_value - delta, max_value + delta
        value_range = max_value - min_value
        return min_value - value_range * padding, max_value + value_range * padding

    @staticmethod
    def _map_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
        if src_max == src_min:
            return (dst_min + dst_max) / 2.0
        ratio = (value - src_min) / (src_max - src_min)
        return dst_min + (dst_max - dst_min) * ratio

    @staticmethod
    def _quartiles(values: list[float]) -> tuple[float, float, float, float, float]:
        sorted_values = sorted(values) if values else [0.0]

        def _median(sequence: list[float]) -> float:
            length = len(sequence)
            midpoint = length // 2
            if length % 2 == 0:
                return (sequence[midpoint - 1] + sequence[midpoint]) / 2.0
            return sequence[midpoint]

        median = _median(sorted_values)
        midpoint = len(sorted_values) // 2
        if len(sorted_values) % 2 == 0:
            lower_half = sorted_values[:midpoint]
            upper_half = sorted_values[midpoint:]
        else:
            lower_half = sorted_values[:midpoint]
            upper_half = sorted_values[midpoint + 1 :]
        q1 = _median(lower_half or sorted_values)
        q3 = _median(upper_half or sorted_values)
        return sorted_values[0], q1, median, q3, sorted_values[-1]

    def _write_svg(self, file_name: str, width: int, height: int, elements: list[str]) -> None:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>"
        )
        (self.experiment_dir / file_name).write_text(svg, encoding="utf-8")

    @staticmethod
    def _escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
