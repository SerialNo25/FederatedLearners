"""Configuration schema for evaluating many model runs on many datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    path: Path


class CheckpointRunRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    base_path: Path
    run_number: int = Field(gt=0)
    checkpoint_name: str = "model.pt"

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if not value:
            raise ValueError("model_id must not be empty")
        return value

    @field_validator("checkpoint_name")
    @classmethod
    def _validate_checkpoint_name(cls, value: str) -> str:
        if not value or "/" in value:
            raise ValueError("checkpoint_name must be a file name")
        return value

    @property
    def run_dir_name(self) -> str:
        return f"run_{self.run_number:03d}"

    @property
    def checkpoint_path(self) -> Path:
        return self.base_path / self.run_dir_name / self.checkpoint_name


class EnsembleRunRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    local_model_id: str
    federated_model_id: str
    ensemble_weight: float | None = None

    @field_validator("ensemble_weight")
    @classmethod
    def _validate_optional_ensemble_weight(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("ensemble_weight must be between 0 and 1 (inclusive)")
        return value


class ModelMatrixConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["model_matrix_evaluation"] = "model_matrix_evaluation"
    experiment_name: str = "model_matrix_evaluation"
    output_dir: Path = Path("data/experiments")
    classification_threshold: float = 0.5
    ensemble_weight: float = 0.5
    datasets: list[DatasetRef]
    local_models: list[CheckpointRunRef]
    global_federated_model: CheckpointRunRef
    exclusive_federated_models: list[CheckpointRunRef]
    exclusive_ensembles: list[EnsembleRunRef]
    inclusive_ensembles: list[EnsembleRunRef]

    @field_validator("classification_threshold")
    @classmethod
    def _validate_classification_threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("classification_threshold must be between 0 and 1 (exclusive)")
        return value

    @field_validator("ensemble_weight")
    @classmethod
    def _validate_ensemble_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("ensemble_weight must be between 0 and 1 (inclusive)")
        return value

    @model_validator(mode="after")
    def _validate_model_references(self) -> "ModelMatrixConfig":
        checkpoint_model_ids = {
            self.global_federated_model.model_id,
            *(model.model_id for model in self.local_models),
            *(model.model_id for model in self.exclusive_federated_models),
        }
        if len(checkpoint_model_ids) != (
            1 + len(self.local_models) + len(self.exclusive_federated_models)
        ):
            raise ValueError("checkpoint model_id values must be unique")

        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("dataset_id values must be unique")

        ensemble_ids = [
            *(ensemble.model_id for ensemble in self.exclusive_ensembles),
            *(ensemble.model_id for ensemble in self.inclusive_ensembles),
        ]
        if len(ensemble_ids) != len(set(ensemble_ids)):
            raise ValueError("ensemble model_id values must be unique")

        for ensemble in [*self.exclusive_ensembles, *self.inclusive_ensembles]:
            if ensemble.local_model_id not in checkpoint_model_ids:
                raise ValueError(
                    f"ensemble '{ensemble.model_id}' references unknown local model "
                    f"'{ensemble.local_model_id}'"
                )
            if ensemble.federated_model_id not in checkpoint_model_ids:
                raise ValueError(
                    f"ensemble '{ensemble.model_id}' references unknown federated model "
                    f"'{ensemble.federated_model_id}'"
                )

        return self

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelMatrixConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
