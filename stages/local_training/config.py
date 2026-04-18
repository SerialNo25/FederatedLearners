"""Configuration schema for single-institution local training stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from domain.models.model_config import ModelConfig, validate_model_config


class LocalTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["local_training"] = "local_training"
    experiment_name: str = "local_single_institution"
    output_dir: Path = Path("data/experiments")
    institution_id: str
    dataset_path: Path
    local_epochs: int
    learning_rate: float
    fraud_weight: float = 100.0
    batch_size: int = 256
    classification_threshold: float = 0.5
    seed: int = 42
    model: ModelConfig

    @field_validator("local_epochs")
    @classmethod
    def _validate_local_epochs(cls, value: int) -> int:
        if value < 1:
            raise ValueError("local_epochs must be >= 1")
        return value

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        return value

    @field_validator("fraud_weight")
    @classmethod
    def _validate_fraud_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("fraud_weight must be > 0")
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("batch_size must be >= 1")
        return value

    @field_validator("classification_threshold")
    @classmethod
    def _validate_classification_threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("classification_threshold must be between 0 and 1 (exclusive)")
        return value

    @field_validator("model")
    @classmethod
    def _validate_model_type(cls, value: ModelConfig) -> ModelConfig:
        return validate_model_config(value)

    @classmethod
    def from_dict(cls, payload: dict) -> "LocalTrainingConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
