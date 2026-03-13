"""Protocol definitions for models used in federated workflows."""

from __future__ import annotations

from typing import Protocol


class FederatedModelProtocol(Protocol):
    """Minimal interface required by federated orchestration and evaluation."""

    def load_parameters(self, parameters: dict[str, list[float]]) -> None:
        ...

    def federated_parameters(self) -> dict[str, list[float]]:
        ...

    def predict_proba(self, features: list[list[float]]) -> list[float]:
        ...
