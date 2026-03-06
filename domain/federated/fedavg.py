"""Federated optimization routines for three-institution training."""

from __future__ import annotations

from dataclasses import dataclass

from domain.data.dataset_loader import InstitutionDataset
from domain.models.basic_model import LogisticRegressionModel
from domain.training.trainer import TrainingConfig, train_local_model


@dataclass(frozen=True)
class InstitutionUpdate:
    institution_id: str
    num_samples: int
    weights: list[float]
    bias: float
    local_loss: float


def aggregate_weighted(updates: list[InstitutionUpdate]) -> tuple[list[float], float]:
    total_samples = sum(update.num_samples for update in updates)
    n_features = len(updates[0].weights)
    aggregated_weights = [0.0] * n_features
    aggregated_bias = 0.0

    for update in updates:
        for feature_idx, value in enumerate(update.weights):
            aggregated_weights[feature_idx] += value * update.num_samples
        aggregated_bias += update.bias * update.num_samples

    aggregated_weights = [value / total_samples for value in aggregated_weights]
    aggregated_bias /= total_samples
    return aggregated_weights, aggregated_bias


def run_federated_round(
    global_model: LogisticRegressionModel,
    institution_datasets: list[InstitutionDataset],
    training_config: TrainingConfig,
) -> list[InstitutionUpdate]:
    updates: list[InstitutionUpdate] = []
    global_params = global_model.parameters()

    for dataset in institution_datasets:
        local_model = LogisticRegressionModel.initialize(len(dataset.features[0]))
        local_model.load_parameters(*global_params)
        local_loss = train_local_model(
            model=local_model,
            features=dataset.features,
            labels=dataset.labels,
            config=training_config,
            global_parameters=global_params,
        )
        weights, bias = local_model.parameters()
        updates.append(
            InstitutionUpdate(
                institution_id=dataset.institution_id,
                num_samples=len(dataset.labels),
                weights=weights,
                bias=bias,
                local_loss=local_loss,
            )
        )

    global_model.load_parameters(*aggregate_weighted(updates))
    return updates
