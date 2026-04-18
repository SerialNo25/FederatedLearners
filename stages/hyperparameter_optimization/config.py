"""Configuration schema for Optuna-backed local hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from domain.models.model_config import ModelConfig, validate_model_config


class FloatSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    low: float
    high: float
    log: bool = False
    step: float | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> "FloatSearchSpace":
        if self.low >= self.high:
            raise ValueError("float search low must be lower than high")
        if self.log and (self.low <= 0 or self.high <= 0):
            raise ValueError("log-scaled float search bounds must be > 0")
        if self.step is not None and self.step <= 0:
            raise ValueError("float search step must be > 0")
        if self.log and self.step is not None:
            raise ValueError("Optuna float log search cannot use step")
        return self


class IntSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def _validate_range(self) -> "IntSearchSpace":
        if self.low > self.high:
            raise ValueError("int search low must be <= high")
        if self.low < 1:
            raise ValueError("int search low must be >= 1")
        if self.step < 1:
            raise ValueError("int search step must be >= 1")
        if self.log and self.step != 1:
            raise ValueError("Optuna integer log search requires step=1")
        return self


class IntChoices(BaseModel):
    model_config = ConfigDict(frozen=True)

    choices: list[int]

    @field_validator("choices")
    @classmethod
    def _validate_choices(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("choices must not be empty")
        if any(choice < 1 for choice in value):
            raise ValueError("integer choices must be >= 1")
        return value


class TrainingSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: FloatSearchSpace | None = None
    fraud_weight: FloatSearchSpace | None = None
    local_epochs: IntSearchSpace | None = None
    batch_size: IntChoices | None = None
    classification_threshold: FloatSearchSpace | None = None


class TabNetSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision_dim: IntChoices | None = None
    attention_dim: IntChoices | None = None
    steps: IntSearchSpace | None = None
    relaxation_factor: FloatSearchSpace | None = None
    sparsity_weight: FloatSearchSpace | None = None


class HyperparameterSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    training: TrainingSearchSpace = TrainingSearchSpace()
    tabnet: TabNetSearchSpace = TabNetSearchSpace()


class HyperparameterOptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["hyperparameter_optimization"] = "hyperparameter_optimization"
    experiment_name: str = "local_hyperparameter_optimization"
    output_dir: Path = Path("data/experiments")
    institution_id: str
    dataset_path: Path
    n_trials: int = 10
    timeout_seconds: int | None = None
    study_name: str | None = None
    storage_url: str | None = None
    load_if_exists: bool = True
    validation_fraction: float = 0.2
    objective_metric: Literal["val_pr_auc", "val_roc_auc", "val_f1", "val_loss"] = "val_pr_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    seed: int = 42
    local_epochs: int = 5
    learning_rate: float = 0.01
    fraud_weight: float = 100.0
    batch_size: int = 256
    classification_threshold: float = 0.5
    model: ModelConfig
    search_space: HyperparameterSearchSpace = HyperparameterSearchSpace()

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

    @field_validator("validation_fraction")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1 (exclusive)")
        return value

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

    @model_validator(mode="after")
    def _validate_objective_direction(self) -> "HyperparameterOptimizationConfig":
        if self.objective_metric == "val_loss" and self.direction != "minimize":
            raise ValueError("direction must be 'minimize' when objective_metric is val_loss")
        if self.objective_metric != "val_loss" and self.direction != "maximize":
            raise ValueError("direction must be 'maximize' for metric objectives")
        return self

    @classmethod
    def from_dict(cls, payload: dict) -> "HyperparameterOptimizationConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
