"""Federated orchestration for n institutions using FedProx local training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.dataset.dataset_loader import InstitutionDataset
from domain.federated.model_parameters import get_model_parameters
from domain.models.federated_model_protocol import FederatedModelProtocol
from domain.models.model_registry import ModelFactoryProtocol
from domain.training.trainer import TrainingConfig, train_local_model


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
        model_factory: ModelFactoryProtocol,
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
        local_parameters = get_model_parameters(local_model)
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


class FedProxOrchestrator:
    """Coordinates institution nodes with FedProx-style local optimization and weighted averaging."""

    def __init__(
        self,
        institutions: list[InstitutionNode],
        initial_model: FederatedModelProtocol,
    ) -> None:
        self._institutions = institutions
        self._global_model = initial_model

    def run_round(self) -> list[InstitutionUpdate]:
        global_parameters = get_model_parameters(self._global_model)
        parameter_names = list(global_parameters.keys())
        updates = [institution.fit(global_parameters) for institution in self._institutions]

        aggregated_parameters = self._aggregate_weighted_parameters(updates, parameter_names)
        self._global_model.load_parameters(
            {
                name: aggregated_parameters[name].tolist() for name in parameter_names
            }
        )

        return updates

    @property
    def global_model(self) -> FederatedModelProtocol:
        return self._global_model

    @staticmethod
    def _aggregate_weighted_parameters(
        updates: list[InstitutionUpdate],
        parameter_names: list[str],
    ) -> dict[str, np.ndarray]:
        total_samples = sum(update.num_samples for update in updates)
        if total_samples <= 0:
            raise RuntimeError("FedProx aggregation requires at least one local sample")

        aggregated = {
            name: np.zeros(len(updates[0].parameters[name]), dtype=np.float64)
            for name in parameter_names
        }
        for update in updates:
            sample_weight = update.num_samples / total_samples
            for name in parameter_names:
                aggregated[name] += np.array(update.parameters[name], dtype=np.float64) * sample_weight
        return aggregated
