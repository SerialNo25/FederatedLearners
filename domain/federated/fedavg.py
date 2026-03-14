"""Federated optimization routines for multi-institution training."""

from __future__ import annotations

from dataclasses import dataclass

from domain.dataset.dataset_loader import InstitutionDataset
from domain.federated.model_parameters import get_model_parameters
from domain.models.federated_model_protocol import FederatedModelProtocol
from domain.training.trainer import TrainingConfig, train_local_model


@dataclass(frozen=True)
class InstitutionUpdate:
    institution_id: str
    num_samples: int
    parameters: dict[str, list[float]]
    local_loss: float


def _scale_parameters(parameters: dict[str, list[float]], scalar: float) -> dict[str, list[float]]:
    return {name: [value * scalar for value in values] for name, values in parameters.items()}


def _add_parameters(
    lhs: dict[str, list[float]], rhs: dict[str, list[float]]
) -> dict[str, list[float]]:
    return {
        name: [left + right for left, right in zip(lhs[name], rhs[name])] for name in lhs
    }


def aggregate_weighted(updates: list[InstitutionUpdate]) -> dict[str, list[float]]:
    total_samples = sum(update.num_samples for update in updates)
    weighted_sum: dict[str, list[float]] = {}

    for update in updates:
        scaled = _scale_parameters(update.parameters, update.num_samples)
        if not weighted_sum:
            weighted_sum = scaled
            continue
        weighted_sum = _add_parameters(weighted_sum, scaled)

    return {name: [value / total_samples for value in values] for name, values in weighted_sum.items()}


def run_federated_round(
    global_model: FederatedModelProtocol,
    institution_datasets: list[InstitutionDataset],
    training_config: TrainingConfig,
    local_model_factory,
) -> list[InstitutionUpdate]:
    updates: list[InstitutionUpdate] = []
    global_params = get_model_parameters(global_model)

    for dataset in institution_datasets:
        local_model = local_model_factory(len(dataset.features[0]))
        local_model.load_parameters(global_params)
        local_loss = train_local_model(
            model=local_model,
            features=dataset.features,
            labels=dataset.labels,
            config=training_config,
            global_parameters=global_params,
        )
        updates.append(
            InstitutionUpdate(
                institution_id=dataset.institution_id,
                num_samples=len(dataset.labels),
                parameters=get_model_parameters(local_model),
                local_loss=local_loss,
            )
        )

    global_model.load_parameters(aggregate_weighted(updates))
    return updates
