"""Configuration schema for ensemble evaluation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["ensemble"] = "ensemble"
    experiment_name: str = "ensemble"
    output_dir: Path = Path("data/experiments")
    local_model_path: Path
    federated_model_path: Path
    dataset_path: Path
    ensemble_weight: float = 0.5
    classification_threshold: float = 0.5

    @field_validator("ensemble_weight")
    @classmethod
    def _validate_ensemble_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("ensemble_weight must be between 0 and 1 (inclusive)")
        return value

    @field_validator("classification_threshold")
    @classmethod
    def _validate_classification_threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("classification_threshold must be between 0 and 1 (exclusive)")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "EnsembleConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
