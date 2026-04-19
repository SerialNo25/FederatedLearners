"""Class weighting helpers for imbalanced binary classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryClassBalance:
    negatives: int
    positives: int
    pos_weight: float

    @property
    def positive_rate(self) -> float:
        total = self.negatives + self.positives
        return self.positives / max(total, 1)


def compute_binary_class_balance(labels: list[int]) -> BinaryClassBalance:
    negatives = sum(1 for label in labels if label == 0)
    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        raise ValueError("Cannot compute positive class weight with zero positive labels")
    if negatives == 0:
        raise ValueError("Cannot compute positive class weight with zero negative labels")
    return BinaryClassBalance(
        negatives=negatives,
        positives=positives,
        pos_weight=negatives / positives,
    )
