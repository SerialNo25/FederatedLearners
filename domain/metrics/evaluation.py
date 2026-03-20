"""Evaluation utilities for institution-level global model quality."""

from __future__ import annotations

from dataclasses import dataclass

from domain.dataset.dataset_loader import InstitutionDataset
from domain.training.trainer import binary_cross_entropy


@dataclass(frozen=True)
class InstitutionMetrics:
    institution_id: str
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float


def _average_precision(labels: list[int], probabilities: list[float]) -> float:
    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
    total_positives = sum(labels)
    if total_positives == 0:
        return 0.0

    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0

    for _, label in ranked:
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / total_positives
        area += precision * max(recall - previous_recall, 0.0)
        previous_recall = recall

    return area


def evaluate_institution(
    model,
    dataset: InstitutionDataset,
    pos_weight: float = 1.0,
    threshold: float = 0.5,
) -> InstitutionMetrics:
    probabilities = model.predict_proba(dataset.features)
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]

    matches = sum(int(p == l) for p, l in zip(predictions, dataset.labels))
    accuracy = matches / max(len(dataset.labels), 1)
    loss = binary_cross_entropy(dataset.labels, probabilities, pos_weight=pos_weight)

    tp = sum(1 for p, l in zip(predictions, dataset.labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, dataset.labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, dataset.labels) if p == 0 and l == 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    pr_auc = _average_precision(dataset.labels, probabilities)

    return InstitutionMetrics(
        institution_id=dataset.institution_id,
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
    )
