"""Stage that builds harmonized institution datasets from raw CSVs."""

from __future__ import annotations

from pathlib import Path

from domain.dataset.dataset_loader import load_institution_dataset
from domain.harmonization.raw_data_harmonizer import RawDataHarmonizationService, RawDatasetSource
from stages.harmonized_data.config import HarmonizedDataConfig
from stages.stage import Stage


class HarmonizedDataStage(Stage):
    def __init__(
        self,
        config: HarmonizedDataConfig,
        harmonizer: RawDataHarmonizationService,
    ) -> None:
        self.config = config
        self.harmonizer = harmonizer

    def execute(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for dataset in self.config.datasets:
            summary = self.harmonizer.harmonize(
                RawDatasetSource(
                    institution_id=dataset.institution_id,
                    bank_kind=dataset.bank_kind,
                    raw_path=dataset.raw_path,
                    output_filename=dataset.output_filename,
                ),
                self.config.output_dir,
            )
            validated = load_institution_dataset(summary.institution_id, summary.output_path)
            print(
                f"{summary.institution_id} ({summary.bank_kind}): "
                f"{summary.row_count:,} rows | fraud {summary.fraud_count:,} | "
                f"validated features {len(validated.features[0]) if validated.features else 0}"
            )
            print(f"  -> {summary.output_path}")

        return self.config.output_dir
