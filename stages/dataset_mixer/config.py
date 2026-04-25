"""Configuration schema for the dataset mixer stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class DatasetMixSourceConfig(BaseModel):
    institution_id: str
    dataset_path: Path
    sample_size: int

    @field_validator("sample_size")
    @classmethod
    def _validate_sample_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sample_size must be non-negative")
        return value


class DatasetMixerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["dataset_mixer"] = "dataset_mixer"
    output_path: Path = Path("data/mixed_datasets/mixed_dataset.csv")
    mixed_institution_id: str = "mixed_dataset"
    seed: int = 42
    institutions: list[DatasetMixSourceConfig]

    @field_validator("institutions")
    @classmethod
    def _validate_institutions(
        cls,
        value: list[DatasetMixSourceConfig],
    ) -> list[DatasetMixSourceConfig]:
        if not value:
            raise ValueError("At least one institution must be configured")
        ids = [institution.institution_id for institution in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetMixerConfig":
        return cls.model_validate(payload)
