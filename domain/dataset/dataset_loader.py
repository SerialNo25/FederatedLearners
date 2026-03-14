"""Dataset loading and validation for institution-level dataset silos."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from domain.dataset.schema import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class InstitutionDataset:
    institution_id: str
    features: list[list[float]]
    labels: list[int]


class DatasetValidationError(ValueError):
    """Raised when an institution dataset does not satisfy required schema constraints."""


def load_institution_dataset(institution_id: str, csv_path: str | Path) -> InstitutionDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Institution dataset not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if header != ALL_COLUMNS:
            raise DatasetValidationError(
                f"{path} has invalid columns. Expected exact order: {ALL_COLUMNS}"
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
