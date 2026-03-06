"""Utility helpers for consistent metric aggregation."""

from __future__ import annotations


def weighted_mean(values: list[float], weights: list[int]) -> float:
    """Return the weighted mean, guarding against empty/zero-weight inputs."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0

    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight

