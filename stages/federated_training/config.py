"""Configuration schema for federated training stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.models.model_registry import MODEL_REGISTRY


class InstitutionConfig(BaseModel):
    institution_id: str
    dataset_path: Path


class FederatedTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str = "federated_global"
    output_dir: Path = Path("data/experiments")
    num_institutions: int
    institutions: list[InstitutionConfig] = Field(default_factory=list)
    num_rounds: int
    local_epochs: int
    learning_rate: float
    proximal_mu: float
    model_type: str
    tabnet_decision_dim: int
    tabnet_attention_dim: int
    tabnet_steps: int
    tabnet_relaxation_factor: float
    tabnet_sparsity_weight: float

    @field_validator("num_rounds", "local_epochs")
    @classmethod
    def _validate_positive_training_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_rounds and local_epochs must be >= 1")
        return value

    @field_validator("num_institutions")
    @classmethod
    def _validate_num_institutions(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_institutions must be >= 1")
        return value

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        return value

    @field_validator("proximal_mu")
    @classmethod
    def _validate_proximal_mu(cls, value: float) -> float:
        if value < 0:
            raise ValueError("proximal_mu must be >= 0")
        return value

    @field_validator("tabnet_steps")
    @classmethod
    def _validate_tabnet_steps(cls, value: int) -> int:
        if value < 1:
            raise ValueError("tabnet_steps must be >= 1")
        return value

    @field_validator("model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        if not MODEL_REGISTRY.has(value):
            valid_model_types = ", ".join(MODEL_REGISTRY.list_model_types())
            raise ValueError(f"model_type must be one of: {valid_model_types}")
        return value

    @model_validator(mode="after")
    def _validate_institutions(self) -> "FederatedTrainingConfig":
        if len(self.institutions) != self.num_institutions:
            raise ValueError(
                "Configured institutions count must match num_institutions"
            )

        ids = [institution.institution_id for institution in self.institutions]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")

        return self

    @classmethod
    def from_dict(cls, payload: dict) -> "FederatedTrainingConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
