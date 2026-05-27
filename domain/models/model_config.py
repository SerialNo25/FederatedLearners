"""Shared model configuration schema used by both local and federated training."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from domain.models.model_registry import MODEL_REGISTRY


class TabNetModelConfig(BaseModel):
    model_type: Literal["tabnet"]
    decision_dim: int = 16
    attention_dim: int = 16
    steps: int = 3
    relaxation_factor: float = 1.5
    sparsity_weight: float = 1e-4

    @field_validator("steps")
    @classmethod
    def _validate_tabnet_steps(cls, value: int) -> int:
        if value < 1:
            raise ValueError("steps must be >= 1")
        return value

    @field_validator("decision_dim", "attention_dim")
    @classmethod
    def _validate_positive_dims(cls, value: int) -> int:
        if value < 1:
            raise ValueError("decision_dim and attention_dim must be >= 1")
        return value


ModelConfig = Annotated[
    TabNetModelConfig,
    Field(discriminator="model_type"),
]


def validate_model_config(value: ModelConfig) -> ModelConfig:
    """Raises ValueError if the model_type is not registered."""
    if not MODEL_REGISTRY.has(value.model_type):
        valid = ", ".join(MODEL_REGISTRY.list_model_types())
        raise ValueError(f"model_type must be one of: {valid}")
    return value
