"""Stage that materializes the exact local-training validation split for ensemble fitting."""

from __future__ import annotations

import json
from pathlib import Path

from domain.dataset.dataset_loader import (
    DEFAULT_LOCAL_TRAINING_VALIDATION_FRACTION,
    load_institution_dataset,
    split_dataset,
    split_for_local_training,
    write_institution_dataset,
)
from stages.ensemble_validation_split.config import EnsembleValidationSplitConfig
from stages.stage import Stage


class EnsembleValidationSplitStage(Stage):
    def __init__(self, config: EnsembleValidationSplitConfig) -> None:
        self.config = config

    def execute(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for institution in self.config.institutions:
            dataset = load_institution_dataset(
                institution_id=institution.institution_id,
                csv_path=institution.dataset_path,
            )
            validation_dataset = self._validation_subset(dataset)

            output_path = self.config.output_dir / f"{institution.institution_id}_validation.csv"
            write_institution_dataset(validation_dataset, output_path)
            output_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "institution_id": institution.institution_id,
                        "source_dataset_path": str(institution.dataset_path),
                        "seed": self.config.seed,
                        "validation_fraction": self.config.validation_fraction,
                        "rows_total": len(dataset.labels),
                        "rows_validation": len(validation_dataset.labels),
                        "fraud_count_validation": sum(validation_dataset.labels),
                        "matches_local_training_split": (
                            self.config.validation_fraction
                            == DEFAULT_LOCAL_TRAINING_VALIDATION_FRACTION
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"{institution.institution_id}: validation {len(validation_dataset.labels):,} "
                f"of {len(dataset.labels):,} rows -> {output_path}"
            )

        return self.config.output_dir

    def _validation_subset(self, dataset):
        if self.config.validation_fraction == DEFAULT_LOCAL_TRAINING_VALIDATION_FRACTION:
            _, validation_dataset = split_for_local_training(dataset, seed=self.config.seed)
            return validation_dataset

        _, validation_dataset = split_dataset(
            dataset,
            val_fraction=self.config.validation_fraction,
            seed=self.config.seed,
        )
        return validation_dataset
