"""Adapters for reusing Flower strategies in local orchestrators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from flwr.common import (
        Code,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        GetPropertiesIns,
        GetPropertiesRes,
        Status,
        ndarrays_to_parameters,
    )
    from flwr.server.client_proxy import ClientProxy
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Flower is required for federated training. Install dependencies with `pip install flwr`."
    ) from exc


@dataclass(frozen=True)
class LocalFitUpdate:
    """Minimal update payload required to construct Flower fit responses."""

    institution_id: str
    num_samples: int
    parameters: dict[str, list[float]]
    local_loss: float


class LocalClientProxy(ClientProxy):
    """Minimal Flower client proxy for local simulation aggregation APIs."""

    def __init__(self, cid: str) -> None:
        super().__init__(cid=cid)

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetPropertiesRes:
        raise NotImplementedError("Local-only proxy does not expose get_properties")

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetParametersRes:
        raise NotImplementedError("Local-only proxy does not expose get_parameters")

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        raise NotImplementedError("Local-only proxy does not expose fit")

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: float | None,
        group_id: int | None,
    ) -> EvaluateRes:
        raise NotImplementedError("Local-only proxy does not expose evaluate")

    def reconnect(self, ins: Any, timeout: float | None, group_id: int | None) -> Any:
        raise NotImplementedError("Local-only proxy does not expose reconnect")


def build_fit_results(
    updates: Sequence[LocalFitUpdate],
    parameter_names: Sequence[str],
) -> list[tuple[ClientProxy, FitRes]]:
    """Translate local update payloads to Flower-compatible fit results."""

    return [
        (
            LocalClientProxy(update.institution_id),
            FitRes(
                status=Status(code=Code.OK, message="local_fit_complete"),
                parameters=ndarrays_to_parameters(
                    [np.array(update.parameters[name], dtype=np.float64) for name in parameter_names]
                ),
                num_examples=update.num_samples,
                metrics={"local_loss": update.local_loss},
            ),
        )
        for update in updates
    ]
