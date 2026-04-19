"""Configuration schema for the dataset split stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class InstitutionSplitConfig(BaseModel):
    institution_id: str
    dataset_path: Path


class DatasetSplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["dataset_split"] = "dataset_split"
    output_dir: Path = Path("train_test_splits")
    test_fraction: float = 0.2
    seed: int = 42
    institutions: list[InstitutionSplitConfig]

    @field_validator("test_fraction")
    @classmethod
    def _validate_test_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("test_fraction must be between 0 and 1 (exclusive)")
        return value

    @field_validator("institutions")
    @classmethod
    def _validate_institutions(cls, value: list[InstitutionSplitConfig]) -> list[InstitutionSplitConfig]:
        if not value:
            raise ValueError("At least one institution must be configured")
        ids = [inst.institution_id for inst in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetSplitConfig":
        return cls.model_validate(payload)
