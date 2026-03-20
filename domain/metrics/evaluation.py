"""Evaluation utilities for institution-level model quality."""

from __future__ import annotations

from dataclasses import dataclass

from domain.dataset.dataset_loader import InstitutionDataset
from domain.training.trainer import binary_cross_entropy


@dataclass(frozen=True)
class InstitutionMetrics:
    institution_id: str
    loss: float
    accuracy: float
    pr_auc: float
    best_f1: float
    best_threshold: float
    thresholds: list[float]
    precision_curve: list[float]
    recall_curve: list[float]
    f1_curve: list[float]
    labels: list[int]
    probabilities: list[float]


def evaluate_institution(
    model,
    dataset: InstitutionDataset,
) -> InstitutionMetrics:
    probabilities = model.predict_proba(dataset.features)
    threshold_curve = compute_threshold_curve(dataset.labels, probabilities)
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    matches = sum(int(prediction == label) for prediction, label in zip(predictions, dataset.labels))
    accuracy = matches / max(len(dataset.labels), 1)
    loss = binary_cross_entropy(dataset.labels, probabilities)
    return InstitutionMetrics(
        institution_id=dataset.institution_id,
        loss=loss,
        accuracy=accuracy,
        pr_auc=threshold_curve.pr_auc,
        best_f1=threshold_curve.best_f1,
        best_threshold=threshold_curve.best_threshold,
        thresholds=threshold_curve.thresholds,
        precision_curve=threshold_curve.precision,
        recall_curve=threshold_curve.recall,
        f1_curve=threshold_curve.f1_scores,
        labels=list(dataset.labels),
        probabilities=probabilities,
    )


@dataclass(frozen=True)
class ThresholdCurve:
    thresholds: list[float]
    precision: list[float]
    recall: list[float]
    f1_scores: list[float]
    pr_auc: float
    best_f1: float
    best_threshold: float


def compute_threshold_curve(labels: list[int], probabilities: list[float]) -> ThresholdCurve:
    if not labels or not probabilities:
        return ThresholdCurve(
            thresholds=[0.5],
            precision=[0.0],
            recall=[0.0],
            f1_scores=[0.0],
            pr_auc=0.0,
            best_f1=0.0,
            best_threshold=0.5,
        )

    candidate_thresholds = sorted({0.0, 0.5, 1.0, *[round(probability, 6) for probability in probabilities]})
    thresholds = sorted(candidate_thresholds)

    precision: list[float] = []
    recall: list[float] = []
    f1_scores: list[float] = []
    for threshold in thresholds:
        predicted_positive = [probability >= threshold for probability in probabilities]
        tp = sum(int(prediction and label == 1) for prediction, label in zip(predicted_positive, labels))
        fp = sum(int(prediction and label == 0) for prediction, label in zip(predicted_positive, labels))
        fn = sum(int((not prediction) and label == 1) for prediction, label in zip(predicted_positive, labels))

        threshold_precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        threshold_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if threshold_precision + threshold_recall > 0:
            threshold_f1 = (
                2.0 * threshold_precision * threshold_recall / (threshold_precision + threshold_recall)
            )
        else:
            threshold_f1 = 0.0

        precision.append(threshold_precision)
        recall.append(threshold_recall)
        f1_scores.append(threshold_f1)

    best_index = max(range(len(thresholds)), key=lambda index: (f1_scores[index], -abs(thresholds[index] - 0.5)))

    pr_auc = 0.0
    recall_precision_pairs = sorted(zip(recall, precision), key=lambda item: item[0])
    previous_recall, previous_precision = 0.0, 1.0
    for current_recall, current_precision in recall_precision_pairs:
        pr_auc += (current_recall - previous_recall) * (current_precision + previous_precision) / 2.0
        previous_recall, previous_precision = current_recall, current_precision
    if previous_recall < 1.0:
        pr_auc += (1.0 - previous_recall) * previous_precision / 2.0

    return ThresholdCurve(
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        f1_scores=f1_scores,
        pr_auc=max(pr_auc, 0.0),
        best_f1=f1_scores[best_index],
        best_threshold=thresholds[best_index],
    )
