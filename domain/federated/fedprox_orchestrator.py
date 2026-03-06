"""Federated orchestration for three institutions using Flower FedProx aggregation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from domain.dataset.dataset_loader import InstitutionDataset
from domain.federated.flower_adapter import LocalFitUpdate, build_fit_results
from domain.training.trainer import TrainingConfig, train_local_model

try:
    from flwr.common import Parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedProx
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Flower is required for federated training. Install dependencies with `pip install flwr`."
    ) from exc


@dataclass(frozen=True)
class InstitutionUpdate:
    institution_id: str
    num_samples: int
    parameters: dict[str, list[float]]
    local_loss: float
    parameter_delta_l2: float


class InstitutionNode:
    """Represents one institution client participating in federated rounds."""

    def __init__(
        self,
        dataset: InstitutionDataset,
        training_config: TrainingConfig,
        model_factory: Callable[[int], Any],
    ) -> None:
        self.dataset = dataset
        self.training_config = training_config
        self.model_factory = model_factory

    @property
    def institution_id(self) -> str:
        return self.dataset.institution_id

    def fit(self, global_parameters: dict[str, list[float]]) -> InstitutionUpdate:
        local_model = self.model_factory(len(self.dataset.features[0]))
        local_model.load_parameters(global_parameters)

        local_loss = train_local_model(
            model=local_model,
            features=self.dataset.features,
            labels=self.dataset.labels,
            config=self.training_config,
            global_parameters=global_parameters,
        )
        local_parameters = local_model.parameters()
        parameter_delta_l2 = self._parameter_delta_l2(local_parameters, global_parameters)

        return InstitutionUpdate(
            institution_id=self.institution_id,
            num_samples=len(self.dataset.labels),
            parameters=local_parameters,
            local_loss=local_loss,
            parameter_delta_l2=parameter_delta_l2,
        )

    @staticmethod
    def _parameter_delta_l2(
        local_parameters: dict[str, list[float]],
        global_parameters: dict[str, list[float]],
    ) -> float:
        squares = 0.0
        for name, local_values in local_parameters.items():
            global_values = global_parameters[name]
            local_array = np.array(local_values, dtype=np.float64)
            global_array = np.array(global_values, dtype=np.float64)
            squares += float(np.sum((local_array - global_array) ** 2))
        return float(np.sqrt(squares))


class ThreeInstitutionFedProxOrchestrator:
    """Coordinates three institution nodes with Flower's FedProx aggregation."""

    def __init__(
        self,
        institutions: list[InstitutionNode],
        initial_model: Any,
        proximal_mu: float,
    ) -> None:
        self._institutions = institutions
        self._global_model = initial_model
        self._strategy = FedProx(proximal_mu=proximal_mu)

    def run_round(self, round_index: int) -> list[InstitutionUpdate]:
        global_parameters = self._global_model.parameters()
        parameter_names = list(global_parameters.keys())
        updates = [institution.fit(global_parameters) for institution in self._institutions]

        flower_results = build_fit_results(
            updates=[
                LocalFitUpdate(
                    institution_id=update.institution_id,
                    num_samples=update.num_samples,
                    parameters=update.parameters,
                    local_loss=update.local_loss,
                )
                for update in updates
            ],
            parameter_names=parameter_names,
        )

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
            {
                name: values.tolist()
                for name, values in zip(parameter_names, aggregated_ndarrays, strict=True)
            }
        )

        return updates

    @property
    def global_model(self) -> Any:
        return self._global_model
