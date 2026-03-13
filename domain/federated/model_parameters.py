"""Helpers for accessing model parameters in federated workflows."""

from __future__ import annotations

from typing import Any


def get_model_parameters(model: Any) -> dict[str, list[float]]:
    """Return serializable model parameters for federated exchange."""
    if hasattr(model, "federated_parameters"):
        return model.federated_parameters()
    return model.parameters()

