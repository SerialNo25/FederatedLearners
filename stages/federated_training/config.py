"""Configuration schema for federated training stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.models.model_config import ModelConfig, validate_model_config
from stages.local_training.config import LocalTrainingConfig


class FederatedLocalTrainingOverrides(BaseModel):
    """Optional federated-stage overrides for per-client local training."""

    model_config = ConfigDict(frozen=True)

    local_epochs: int | None = None
    learning_rate: float | None = None
    fraud_weight: float | None = None
    batch_size: int | None = None
    classification_threshold: float | None = None

    @field_validator("local_epochs")
    @classmethod
    def _validate_local_epochs(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("local_epochs override must be >= 1")
        return value

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("learning_rate override must be > 0")
        return value

    @field_validator("fraud_weight")
    @classmethod
    def _validate_fraud_weight(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("fraud_weight override must be > 0")
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("batch_size override must be >= 1")
        return value

    @field_validator("classification_threshold")
    @classmethod
    def _validate_classification_threshold(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError(
                "classification_threshold override must be between 0 and 1 (exclusive)"
            )
        return value


class FederatedTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["federated_training"] = "federated_training"
    experiment_name: str = "federated_global"
    output_dir: Path = Path("data/experiments")
    num_rounds: int
    proximal_mu: float = 0.0
    local_training_overrides: dict[str, FederatedLocalTrainingOverrides] = Field(
        default_factory=dict
    )
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
        unknown_override_ids = set(self.local_training_overrides) - set(ids)
        if unknown_override_ids:
            raise ValueError(
                "local_training_overrides contains unknown institution IDs: "
                + ", ".join(sorted(unknown_override_ids))
            )
        return self

    def local_epochs_for(self, institution: LocalTrainingConfig) -> int:
        return self._override_for(institution).local_epochs or institution.local_epochs

    def learning_rate_for(self, institution: LocalTrainingConfig) -> float:
        return self._override_for(institution).learning_rate or institution.learning_rate

    def fraud_weight_for(self, institution: LocalTrainingConfig) -> float:
        return self._override_for(institution).fraud_weight or institution.fraud_weight

    def batch_size_for(self, institution: LocalTrainingConfig) -> int:
        return self._override_for(institution).batch_size or institution.batch_size

    def classification_threshold_for(self, institution: LocalTrainingConfig) -> float:
        return (
            self._override_for(institution).classification_threshold
            or institution.classification_threshold
        )

    def _override_for(self, institution: LocalTrainingConfig) -> FederatedLocalTrainingOverrides:
        return self.local_training_overrides.get(
            institution.institution_id,
            FederatedLocalTrainingOverrides(),
        )

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
