"""Configuration schema for single-institution local training stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.models.model_registry import MODEL_REGISTRY
from stages.federated_training.config import InstitutionConfig, ModelConfig


class LocalTrainingConfig(BaseModel):
    """Local-training configuration that is intentionally compatible with federated config files."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    experiment_name: str = "local_single_institution"
    output_dir: Path = Path("data/experiments")
    institutions: list[InstitutionConfig] = Field(default_factory=list)
    local_epochs: int
    learning_rate: float
    model: ModelConfig
    fraud_weight: float = 100.0
    classification_threshold: float = 0.5
    validation_fraction: float = 0.2
    seed: int = 42
    local_institution_id: str | None = None

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

    @field_validator("validation_fraction")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        return value

    @field_validator("model")
    @classmethod
    def _validate_model_type(cls, value: ModelConfig) -> ModelConfig:
        if not MODEL_REGISTRY.has(value.model_type):
            valid_model_types = ", ".join(MODEL_REGISTRY.list_model_types())
            raise ValueError(f"model_type must be one of: {valid_model_types}")
        return value

    @model_validator(mode="after")
    def _validate_institutions(self) -> "LocalTrainingConfig":
        if not self.institutions:
            raise ValueError("At least one institution must be configured")

        ids = [institution.institution_id for institution in self.institutions]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")

        if self.local_institution_id is not None and self.local_institution_id not in ids:
            raise ValueError(
                f"local_institution_id '{self.local_institution_id}' does not match any configured institution"
            )

        return self

    @property
    def selected_institution(self) -> InstitutionConfig:
        if self.local_institution_id is None:
            return self.institutions[0]
        return next(
            institution
            for institution in self.institutions
            if institution.institution_id == self.local_institution_id
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "LocalTrainingConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
