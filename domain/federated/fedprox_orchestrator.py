"""Federated orchestration for three institutions using Flower FedProx aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from domain.dataset.dataset_loader import InstitutionDataset
from domain.models.basic_model import LogisticRegressionModel
from domain.training.trainer import TrainingConfig, train_local_model

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
        Parameters,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedProx
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Flower is required for federated training. Install dependencies with `pip install flwr`."
    ) from exc


@dataclass(frozen=True)
class InstitutionUpdate:
    institution_id: str
    num_samples: int
    weights: list[float]
    bias: float
    local_loss: float
    weight_delta_l2: float
    bias_delta_abs: float


class InstitutionNode:
    """Represents one institution client participating in federated rounds."""

    def __init__(self, dataset: InstitutionDataset, training_config: TrainingConfig) -> None:
        self.dataset = dataset
        self.training_config = training_config

    @property
    def institution_id(self) -> str:
        return self.dataset.institution_id

    def fit(self, global_parameters: tuple[list[float], float]) -> InstitutionUpdate:
        global_weights, global_bias = global_parameters
        local_model = LogisticRegressionModel.initialize(len(self.dataset.features[0]))
        local_model.load_parameters(*global_parameters)

        local_loss = train_local_model(
            model=local_model,
            features=self.dataset.features,
            labels=self.dataset.labels,
            config=self.training_config,
            global_parameters=global_parameters,
        )
        weights, bias = local_model.parameters()
        weight_delta_l2 = float(
            np.linalg.norm(np.array(weights, dtype=np.float64) - np.array(global_weights, dtype=np.float64))
        )
        bias_delta_abs = abs(bias - global_bias)

        return InstitutionUpdate(
            institution_id=self.institution_id,
            num_samples=len(self.dataset.labels),
            weights=weights,
            bias=bias,
            local_loss=local_loss,
            weight_delta_l2=weight_delta_l2,
            bias_delta_abs=bias_delta_abs,
        )


class _LocalClientProxy(ClientProxy):
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


class ThreeInstitutionFedProxOrchestrator:
    """Coordinates three institution nodes with Flower's FedProx aggregation."""

    def __init__(
        self,
        institutions: list[InstitutionNode],
        initial_model: LogisticRegressionModel,
        proximal_mu: float,
    ) -> None:
        self._institutions = institutions
        self._global_model = initial_model
        self._strategy = FedProx(proximal_mu=proximal_mu)

    def run_round(self, round_index: int) -> list[InstitutionUpdate]:
        global_parameters = self._global_model.parameters()
        updates = [institution.fit(global_parameters) for institution in self._institutions]

        flower_results = [
            (
                _LocalClientProxy(update.institution_id),
                FitRes(
                    status=Status(code=Code.OK, message="local_fit_complete"),
                    parameters=ndarrays_to_parameters(
                        [
                            np.array(update.weights, dtype=np.float64),
                            np.array([update.bias], dtype=np.float64),
                        ]
                    ),
                    num_examples=update.num_samples,
                    metrics={"local_loss": update.local_loss},
                ),
            )
            for update in updates
        ]

        aggregated_result = self._strategy.aggregate_fit(
            server_round=round_index,
            results=flower_results,
            failures=[],
        )
        aggregated_parameters: Parameters | None = aggregated_result[0] if aggregated_result else None
        if aggregated_parameters is None:
            raise RuntimeError("FedProx aggregation produced no parameters")

        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        self._global_model.load_parameters(
            aggregated_ndarrays[0].tolist(),
            float(aggregated_ndarrays[1][0]),
        )

        return updates

    @property
    def global_model(self) -> LogisticRegressionModel:
        return self._global_model
