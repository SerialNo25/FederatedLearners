"""Stage that samples from institution datasets and creates a mixed dataset."""

from __future__ import annotations

from pathlib import Path

from domain.dataset.dataset_loader import (
    load_institution_dataset,
    merge_datasets,
    sample_dataset_rows,
    write_dataset_mix_metadata,
    write_institution_dataset,
)
from stages.dataset_mixer.config import DatasetMixerConfig
from stages.stage import Stage


class DatasetMixerStage(Stage):
    def __init__(self, config: DatasetMixerConfig) -> None:
        self.config = config

    def execute(self) -> Path:
        sampled_datasets = []
        for offset, institution in enumerate(self.config.institutions):
            dataset = load_institution_dataset(
                institution_id=institution.institution_id,
                csv_path=institution.dataset_path,
            )
            sampled_datasets.append(
                sample_dataset_rows(
                    dataset=dataset,
                    sample_size=institution.sample_size,
                    seed=self.config.seed + offset,
                )
            )

        mixed_dataset = merge_datasets(
            [sample.dataset for sample in sampled_datasets],
            mixed_institution_id=self.config.mixed_institution_id,
            seed=self.config.seed,
        )

        metadata_path = self.config.output_path.with_suffix(".json")
        write_institution_dataset(mixed_dataset, self.config.output_path)
        write_dataset_mix_metadata(
            mixed_dataset=mixed_dataset,
            path=metadata_path,
            seed=self.config.seed,
            source_samples=sampled_datasets,
        )

        for sample in sampled_datasets:
            print(
                f"{sample.dataset.institution_id}: sampled {sample.sampled_rows:,} rows "
                f"({sample.fraud_count:,} fraud)"
            )
        print(
            f"mixed dataset {mixed_dataset.institution_id}: "
            f"{len(mixed_dataset.labels):,} rows ({sum(mixed_dataset.labels):,} fraud)"
        )
        print(f"  -> {self.config.output_path}")
        print(f"  -> {metadata_path}")
        return self.config.output_path
