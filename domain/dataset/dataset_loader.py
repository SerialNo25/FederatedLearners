"""Dataset loading, validation, and sampling for institution-level datasets."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from domain.dataset.schema import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class InstitutionDataset:
    institution_id: str
    features: list[list[float]]
    labels: list[int]


class DatasetValidationError(ValueError):
    """Raised when an institution dataset does not satisfy required schema constraints."""


@dataclass(frozen=True)
class SampledInstitutionDataset:
    dataset: InstitutionDataset
    requested_rows: int
    sampled_rows: int
    fraud_count: int


def load_institution_dataset(institution_id: str, csv_path: str | Path) -> InstitutionDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Institution dataset not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if not all(col in header for col in ALL_COLUMNS):
            missing = [col for col in ALL_COLUMNS if col not in header]
            raise DatasetValidationError(
                f"{path} is missing required columns: {missing}"
            )

        features: list[list[float]] = []
        labels: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                feature_row = [float(row[column]) for column in FEATURE_COLUMNS]
                label = int(float(row[TARGET_COLUMN]))
            except (ValueError, TypeError) as exc:
                raise DatasetValidationError(
                    f"{path}:{row_number} contains non-numeric value"
                ) from exc

            if label not in (0, 1):
                raise DatasetValidationError(
                    f"{path}:{row_number} contains invalid class label {label}"
                )

            features.append(feature_row)
            labels.append(label)

    return InstitutionDataset(
        institution_id=institution_id,
        features=features,
        labels=labels,
    )


def split_dataset(
    dataset: InstitutionDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[InstitutionDataset, InstitutionDataset]:
    """Stratified train/val split preserving class ratios."""
    rng = random.Random(seed)

    indices_by_class: dict[int, list[int]] = {}
    for i, label in enumerate(dataset.labels):
        indices_by_class.setdefault(label, []).append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in indices_by_class.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    def _subset(idx: list[int]) -> InstitutionDataset:
        return InstitutionDataset(
            institution_id=dataset.institution_id,
            features=[dataset.features[i] for i in idx],
            labels=[dataset.labels[i] for i in idx],
        )

    return _subset(train_indices), _subset(val_indices)


def write_institution_dataset(dataset: InstitutionDataset, path: Path) -> None:
    """Write a dataset back to CSV with the standard schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for features, label in zip(dataset.features, dataset.labels):
            row = {col: features[i] for i, col in enumerate(FEATURE_COLUMNS)}
            row[TARGET_COLUMN] = label
            writer.writerow(row)


def sample_dataset_rows(
    dataset: InstitutionDataset,
    sample_size: int,
    seed: int,
) -> SampledInstitutionDataset:
    """Randomly sample rows without replacement from a validated institution dataset."""
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size > len(dataset.labels):
        raise ValueError(
            f"{dataset.institution_id} requested {sample_size} rows, "
            f"but only {len(dataset.labels)} are available"
        )

    if sample_size == 0:
        empty_dataset = InstitutionDataset(
            institution_id=dataset.institution_id,
            features=[],
            labels=[],
        )
        return SampledInstitutionDataset(
            dataset=empty_dataset,
            requested_rows=0,
            sampled_rows=0,
            fraud_count=0,
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset.labels)), sample_size)
    sampled_dataset = InstitutionDataset(
        institution_id=dataset.institution_id,
        features=[dataset.features[i] for i in indices],
        labels=[dataset.labels[i] for i in indices],
    )
    return SampledInstitutionDataset(
        dataset=sampled_dataset,
        requested_rows=sample_size,
        sampled_rows=sample_size,
        fraud_count=sum(sampled_dataset.labels),
    )


def merge_datasets(
    datasets: list[InstitutionDataset],
    mixed_institution_id: str,
    seed: int,
) -> InstitutionDataset:
    """Combine multiple institution datasets and shuffle row order reproducibly."""
    if not datasets:
        raise ValueError("At least one dataset is required to create a mixed dataset")

    rows: list[tuple[list[float], int]] = []
    for dataset in datasets:
        rows.extend(zip(dataset.features, dataset.labels))

    rng = random.Random(seed)
    rng.shuffle(rows)

    return InstitutionDataset(
        institution_id=mixed_institution_id,
        features=[features for features, _ in rows],
        labels=[label for _, label in rows],
    )


def write_dataset_mix_metadata(
    *,
    mixed_dataset: InstitutionDataset,
    path: Path,
    seed: int,
    source_samples: list[SampledInstitutionDataset],
) -> None:
    """Persist reproducibility metadata for a mixed dataset artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mixed_institution_id": mixed_dataset.institution_id,
        "seed": seed,
        "total_rows": len(mixed_dataset.labels),
        "fraud_count": sum(mixed_dataset.labels),
        "sources": [
            {
                "institution_id": sample.dataset.institution_id,
                "requested_rows": sample.requested_rows,
                "sampled_rows": sample.sampled_rows,
                "fraud_count": sample.fraud_count,
            }
            for sample in source_samples
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
