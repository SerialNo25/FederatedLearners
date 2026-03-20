"""Configuration schema for model evaluation stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str = "evaluation"
    output_dir: Path = Path("data/experiments")
    model_path: Path
    dataset_path: Path
    classification_threshold: float = 0.5

    @field_validator("classification_threshold")
    @classmethod
    def _validate_classification_threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("classification_threshold must be between 0 and 1 (exclusive)")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "EvaluationConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
