"""Configuration schema for Optuna-backed ensemble-weight optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from stages.hyperparameter_optimization.config import FloatSearchSpace
from stages.model_matrix_evaluation.config import CheckpointRunRef


class EnsembleWeightSearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    ensemble_weight: FloatSearchSpace = FloatSearchSpace(low=0.0, high=1.0, step=0.01)


class EnsembleWeightOptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["ensemble_weight_optimization"] = "ensemble_weight_optimization"
    experiment_name: str = "ensemble_weight_optimization"
    output_dir: Path = Path("data/experiments")
    institution_id: str
    dataset_path: Path
    local_model: CheckpointRunRef
    federated_model: CheckpointRunRef
    n_trials: int = 25
    timeout_seconds: int | None = None
    study_name: str | None = None
    storage_url: str | None = None
    load_if_exists: bool = True
    objective_metric: Literal["val_pr_auc", "val_roc_auc", "val_f1", "val_loss"] = "val_pr_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    seed: int = 42
    classification_threshold: float = 0.5
    search_space: EnsembleWeightSearchSpace = EnsembleWeightSearchSpace()

    @field_validator("institution_id", "study_name", "storage_url")
    @classmethod
    def _validate_optional_non_empty_string(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("institution_id, study_name, and storage_url must not be empty")
        return value

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

    @field_validator("classification_threshold")
    @classmethod
    def _validate_classification_threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("classification_threshold must be between 0 and 1 (exclusive)")
        return value

    @model_validator(mode="after")
    def _validate_config(self) -> "EnsembleWeightOptimizationConfig":
        if self.objective_metric == "val_loss" and self.direction != "minimize":
            raise ValueError("direction must be 'minimize' when objective_metric is val_loss")
        if self.objective_metric != "val_loss" and self.direction != "maximize":
            raise ValueError("direction must be 'maximize' for metric objectives")
        if self.local_model.model_id == self.federated_model.model_id:
            raise ValueError("local_model and federated_model model_id values must be unique")
        return self

    @classmethod
    def from_dict(cls, payload: dict) -> "EnsembleWeightOptimizationConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
