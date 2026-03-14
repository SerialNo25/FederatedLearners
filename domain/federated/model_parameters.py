"""Helpers for accessing model parameters in federated workflows."""

from __future__ import annotations

from domain.models.federated_model_protocol import FederatedModelProtocol


def get_model_parameters(model: FederatedModelProtocol) -> dict[str, list[float]]:
    """Return serializable model parameters for federated exchange."""
    return model.federated_parameters()
