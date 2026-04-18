"""Configuration schema for raw dataset harmonization stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class RawDatasetConfig(BaseModel):
    institution_id: str
    bank_kind: Literal["sparkov", "banksim", "ccfraud"]
    raw_path: Path
    output_filename: str


class HarmonizedDataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["harmonized_data"] = "harmonized_data"
    output_dir: Path = Path("data/harmonized")
    seed: int = 42
    sparkov_target_size: int = 500_000
    datasets: list[RawDatasetConfig]

    @field_validator("sparkov_target_size")
    @classmethod
    def _validate_target_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sparkov_target_size must be positive")
        return value

    @field_validator("datasets")
    @classmethod
    def _validate_datasets(cls, value: list[RawDatasetConfig]) -> list[RawDatasetConfig]:
        if not value:
            raise ValueError("At least one dataset must be configured")
        ids = [dataset.institution_id for dataset in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset institution IDs must be unique")
        filenames = [dataset.output_filename for dataset in value]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Dataset output filenames must be unique")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "HarmonizedDataConfig":
        return cls.model_validate(payload)
