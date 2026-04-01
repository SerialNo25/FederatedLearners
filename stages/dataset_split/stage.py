"""Stage that splits institution datasets into train and test CSV files."""

from __future__ import annotations

from pathlib import Path

from domain.dataset.dataset_loader import (
    load_institution_dataset,
    split_dataset,
    write_institution_dataset,
)
from stages.dataset_split.config import DatasetSplitConfig
from stages.stage import Stage


class DatasetSplitStage(Stage):
    def __init__(self, config: DatasetSplitConfig) -> None:
        self.config = config

    def execute(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for inst in self.config.institutions:
            dataset = load_institution_dataset(
                institution_id=inst.institution_id,
                csv_path=inst.dataset_path,
            )

            train_set, test_set = split_dataset(
                dataset,
                val_fraction=self.config.test_fraction,
                seed=self.config.seed,
            )

            train_path = self.config.output_dir / f"{inst.institution_id}_train.csv"
            test_path = self.config.output_dir / f"{inst.institution_id}_test.csv"

            write_institution_dataset(train_set, train_path)
            write_institution_dataset(test_set, test_path)

            n_total = len(dataset.labels)
            n_train_fraud = sum(train_set.labels)
            n_test_fraud = sum(test_set.labels)
            print(
                f"{inst.institution_id}: {n_total:,} total -> "
                f"train {len(train_set.labels):,} ({n_train_fraud} fraud) | "
                f"test {len(test_set.labels):,} ({n_test_fraud} fraud)"
            )
            print(f"  -> {train_path}")
            print(f"  -> {test_path}")

        return self.config.output_dir
