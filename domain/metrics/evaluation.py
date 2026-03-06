"""Evaluation utilities for institution-level global model quality."""

from __future__ import annotations

from dataclasses import dataclass

from domain.data.dataset_loader import InstitutionDataset
from domain.models.basic_model import LogisticRegressionModel
from domain.training.trainer import binary_cross_entropy


@dataclass(frozen=True)
class InstitutionMetrics:
    institution_id: str
    loss: float
    accuracy: float


def evaluate_institution(
    model: LogisticRegressionModel,
    dataset: InstitutionDataset,
) -> InstitutionMetrics:
    probabilities = model.predict_proba(dataset.features)
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    matches = sum(int(prediction == label) for prediction, label in zip(predictions, dataset.labels))
    accuracy = matches / max(len(dataset.labels), 1)
    loss = binary_cross_entropy(dataset.labels, probabilities)
    return InstitutionMetrics(institution_id=dataset.institution_id, loss=loss, accuracy=accuracy)
