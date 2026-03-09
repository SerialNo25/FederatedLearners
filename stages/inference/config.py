"""Configuration schema for model inference stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from domain.dataset.schema import FEATURE_COLUMNS
from domain.models.model_registry import MODEL_REGISTRY


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str = "inference"
    output_dir: Path = Path("data/experiments")
    checkpoint_path: Path
    model_type: str
    input_data_path: Path
    label_column: str | None = None
    feature_columns: list[str] = FEATURE_COLUMNS
    tabnet_decision_dim: int = 16
    tabnet_attention_dim: int = 16
    tabnet_steps: int = 3
    tabnet_relaxation_factor: float = 1.5
    tabnet_sparsity_weight: float = 1e-4
    tabnet_device: str | None = None

    @field_validator("model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        if not MODEL_REGISTRY.has(value):
            valid_model_types = ", ".join(MODEL_REGISTRY.list_model_types())
            raise ValueError(f"model_type must be one of: {valid_model_types}")
        return value

    @field_validator("feature_columns")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("feature_columns must contain at least one column name")
        if len(value) != len(set(value)):
            raise ValueError("feature_columns must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_label_column(self) -> "InferenceConfig":
        if self.label_column and self.label_column in self.feature_columns:
            raise ValueError("label_column cannot also be present in feature_columns")
        return self

    @classmethod
    def from_dict(cls, payload: dict) -> "InferenceConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
