"""Configuration schema for model inference stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from domain.models.model_registry import MODEL_REGISTRY


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str = "inference"
    output_dir: Path = Path("data/experiments")
    checkpoint_path: Path
    model_type: str
    input_data_path: Path
    tabnet_decision_dim: int = 16
    tabnet_attention_dim: int = 16
    tabnet_steps: int = 3
    tabnet_relaxation_factor: float = 1.5
    tabnet_sparsity_weight: float = 1e-4

    @field_validator("model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        if not MODEL_REGISTRY.has(value):
            valid_model_types = ", ".join(MODEL_REGISTRY.list_model_types())
            raise ValueError(f"model_type must be one of: {valid_model_types}")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "InferenceConfig":
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
