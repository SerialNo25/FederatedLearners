"""Evaluation utilities for institution-level global model quality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tqdm import tqdm

from domain.dataset.dataset_loader import InstitutionDataset
from domain.training.trainer import binary_cross_entropy

from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve

@dataclass(frozen=True)
class InstitutionMetrics:
    institution_id: str
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
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
    pos_weight: float = 1.0,
    threshold: float = 0.5,
) -> InstitutionMetrics:
    probabilities = model.predict_proba(dataset.features)
    threshold_curve = compute_threshold_curve(dataset.labels, probabilities)
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

    return InstitutionMetrics(
        institution_id=dataset.institution_id,
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
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

    return _compute_threshold_curve_sklearn(labels, probabilities)


def _compute_threshold_curve_sklearn(labels: list[int], probabilities: list[float]) -> ThresholdCurve:
    print("computing thresholds")
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