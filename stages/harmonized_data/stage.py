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
            summary = self.harmonizer.harmonize_train_test_split(
                RawDatasetSource(
                    institution_id=dataset.institution_id,
                    bank_kind=dataset.bank_kind,
                    raw_path=dataset.raw_path,
                    output_filename=dataset.output_filename,
                ),
                self.config.output_dir,
                test_fraction=self.config.test_fraction,
            )
            train_validated = load_institution_dataset(
                summary.institution_id,
                summary.train.output_path,
            )
            test_validated = load_institution_dataset(
                summary.institution_id,
                summary.test.output_path,
            )
            print(
                f"{summary.institution_id} ({summary.bank_kind}): "
                f"train {summary.train.row_count:,} rows ({summary.train.fraud_count:,} fraud) | "
                f"test {summary.test.row_count:,} rows ({summary.test.fraud_count:,} fraud) | "
                f"validated features train={len(train_validated.features[0]) if train_validated.features else 0} "
                f"test={len(test_validated.features[0]) if test_validated.features else 0}"
            )
            print(f"  -> {summary.train.output_path}")
            print(f"  -> {summary.test.output_path}")
            print(f"  -> {summary.preprocessing_artifact_path}")

        return self.config.output_dir
