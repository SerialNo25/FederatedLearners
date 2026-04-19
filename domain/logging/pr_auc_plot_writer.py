"""Dependency-free SVG writer for federated PR-AUC trends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from domain.logging.loss_plot_writer import _build_line_chart_svg


class PRAUCPlotWriter:
    """Writes a basic SVG chart for PR-AUC over federated rounds."""

    @classmethod
    def write(
        cls,
        output_path: Path,
        rounds: Sequence[int],
        pr_auc_values: Sequence[float],
    ) -> Path:
        svg = _build_line_chart_svg(
            title="Federated PR-AUC Over Rounds",
            x_label="Round",
            y_label="PR-AUC",
            rounds=rounds,
            series=(("PR-AUC", "#059669", pr_auc_values),),
        )
        output_path.write_text(svg, encoding="utf-8")
        return output_path
