"""Evaluation utilities for institution-level model quality."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np

from domain.dataset.dataset_loader import InstitutionDataset
from domain.training.trainer import binary_cross_entropy

if importlib.util.find_spec("sklearn") is not None:
    from sklearn.metrics import auc as sklearn_auc
    from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve
else:
    sklearn_auc = None
    sklearn_precision_recall_curve = None


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


@dataclass(frozen=True)
class ThresholdCurve:
    thresholds: list[float]
    precision: list[float]
    recall: list[float]
    f1_scores: list[float]
    pr_auc: float
    best_f1: float
    best_threshold: float


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

    if sklearn_precision_recall_curve is not None and sklearn_auc is not None:
        return _compute_threshold_curve_sklearn(labels, probabilities)
    return _compute_threshold_curve_fallback(labels, probabilities)


def _compute_threshold_curve_sklearn(labels: list[int], probabilities: list[float]) -> ThresholdCurve:
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    precision, recall, thresholds = sklearn_precision_recall_curve(label_array, probability_array)

    f1_scores = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    plot_thresholds = np.concatenate(([0.0], thresholds.astype(np.float64, copy=False)))
    best_f1_scores = f1_scores[1:] if len(f1_scores) > 1 else f1_scores
    best_thresholds = plot_thresholds[1:] if len(plot_thresholds) > 1 else plot_thresholds

    if len(best_thresholds) == 0:
        best_f1 = 0.0
        best_threshold = 0.5
    else:
        max_f1 = float(np.max(best_f1_scores))
        candidate_indices = np.flatnonzero(np.isclose(best_f1_scores, max_f1))
        best_index = min(
            candidate_indices.tolist(),
            key=lambda index: abs(float(best_thresholds[index]) - 0.5),
        )
        best_f1 = float(best_f1_scores[best_index])
        best_threshold = float(best_thresholds[best_index])

    pr_auc = float(sklearn_auc(recall[::-1], precision[::-1]))
    return ThresholdCurve(
        thresholds=plot_thresholds.tolist(),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1_scores=f1_scores.tolist(),
        pr_auc=pr_auc,
        best_f1=best_f1,
        best_threshold=best_threshold,
    )


def _compute_threshold_curve_fallback(labels: list[int], probabilities: list[float]) -> ThresholdCurve:
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

    best_index = max(
        range(len(thresholds)),
        key=lambda index: (f1_scores[index], -abs(thresholds[index] - 0.5)),
    )

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
