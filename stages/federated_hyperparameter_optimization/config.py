"""Configuration schema for Optuna-backed federated hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.models.model_config import ModelConfig, validate_model_config
from stages.federated_training.config import FederatedLocalTrainingOverrides
from stages.hyperparameter_optimization.config import (
    FloatSearchSpace,
    IntChoices,
    IntSearchSpace,
    TabNetSearchSpace,
)
from stages.local_training.config import LocalTrainingConfig


class FederatedParameterSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    proximal_mu: FloatSearchSpace | None = None
    num_rounds: IntSearchSpace | None = None


class FederatedLocalTrainingSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: FloatSearchSpace | None = None
    fraud_weight: FloatSearchSpace | None = None
    local_epochs: IntSearchSpace | None = None
    batch_size: IntChoices | None = None
    classification_threshold: FloatSearchSpace | None = None


class FederatedHyperparameterSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    federated: FederatedParameterSearchSpace = FederatedParameterSearchSpace()
    local_training: FederatedLocalTrainingSearchSpace = FederatedLocalTrainingSearchSpace()
    tabnet: TabNetSearchSpace = TabNetSearchSpace()


class FederatedHyperparameterOptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["federated_hyperparameter_optimization"] = (
        "federated_hyperparameter_optimization"
    )
    experiment_name: str = "hpo_federated_global"
    output_dir: Path = Path("data/experiments")
    n_trials: int = 10
    timeout_seconds: int | None = None
    study_name: str | None = None
    storage_url: str | None = None
    load_if_exists: bool = True
    objective_metric: Literal["val_pr_auc", "val_loss"] = "val_pr_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    seed: int = 42
    num_rounds: int = 5
    proximal_mu: float = 0.0
    local_training_overrides: dict[str, FederatedLocalTrainingOverrides] = Field(
        default_factory=dict
    )
    model: ModelConfig
    institutions: list[LocalTrainingConfig]
    search_space: FederatedHyperparameterSearchSpace = FederatedHyperparameterSearchSpace()

    @field_validator("n_trials")
    @classmethod
    def _validate_n_trials(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_trials must be >= 1")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout_seconds(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("timeout_seconds must be >= 1 when provided")
        return value

    @field_validator("study_name", "storage_url")
    @classmethod
    def _validate_optional_non_empty_string(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("study_name and storage_url must not be empty when provided")
        return value

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
    def _validate_config(self) -> "FederatedHyperparameterOptimizationConfig":
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
        if self.objective_metric == "val_loss" and self.direction != "minimize":
            raise ValueError("direction must be 'minimize' when objective_metric is val_loss")
        if self.objective_metric != "val_loss" and self.direction != "maximize":
            raise ValueError("direction must be 'maximize' for metric objectives")
        return self

    @classmethod
    def from_dict(cls, payload: dict) -> "FederatedHyperparameterOptimizationConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
