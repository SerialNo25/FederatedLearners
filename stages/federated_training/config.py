"""Configuration schema for federated training stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from domain.models.model_config import ModelConfig, validate_model_config
from stages.local_training.config import LocalTrainingConfig


class FederatedTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["federated_training"] = "federated_training"
    experiment_name: str = "federated_global"
    output_dir: Path = Path("data/experiments")
    num_rounds: int
    proximal_mu: float = 0.0
    model: ModelConfig
    institutions: list[LocalTrainingConfig]

    @field_validator("num_rounds")
    @classmethod
    def _validate_num_rounds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_rounds must be >= 1")
        return value

    @field_validator("proximal_mu")
    @classmethod
    def _validate_proximal_mu(cls, value: float) -> float:
        if value < 0:
            raise ValueError("proximal_mu must be >= 0")
        return value

    @field_validator("model")
    @classmethod
    def _validate_model_type(cls, value: ModelConfig) -> ModelConfig:
        return validate_model_config(value)

    @model_validator(mode="after")
    def _validate_institutions(self) -> "FederatedTrainingConfig":
        if not self.institutions:
            raise ValueError("At least one institution must be configured")
        ids = [inst.institution_id for inst in self.institutions]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")
        return self

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
