"""Training utilities for local institution optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    local_epochs: int
    proximal_mu: float = 0.0


def binary_cross_entropy(y_true: list[int] | list[float], y_prob: list[float]) -> float:
    eps = 1e-7
    losses: list[float] = []
    for label, probability in zip(y_true, y_prob):
        probability = min(max(probability, eps), 1.0 - eps)
        losses.append(-(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)))
    return sum(losses) / max(len(losses), 1)


def train_local_model(
    model,
    features: list[list[float]],
    labels: list[int],
    config: TrainingConfig,
    global_parameters: dict[str, list[float]] | None = None,
) -> float:
    if hasattr(model, "fit"):
        return float(
            model.fit(
                features=features,
                labels=labels,
                learning_rate=config.learning_rate,
                epochs=config.local_epochs,
            )
        )

    initial_parameters = global_parameters if global_parameters is not None else model.parameters()
    initial_weights = initial_parameters["weights"]
    initial_bias = initial_parameters["bias"][0]

    n_samples = len(features)
    n_features = len(features[0]) if features else 0

    for _ in range(config.local_epochs):
        predictions = model.predict_proba(features)
        grad_weights = [0.0] * n_features
        grad_bias = 0.0

        for row, label, prediction in zip(features, labels, predictions):
            error = prediction - float(label)
            for feature_idx, value in enumerate(row):
                grad_weights[feature_idx] += value * error
            grad_bias += error

        grad_weights = [value / n_samples for value in grad_weights]
        grad_bias /= n_samples

        if global_parameters is not None and config.proximal_mu > 0.0:
            grad_weights = [
                grad + config.proximal_mu * (weight - init)
                for grad, weight, init in zip(grad_weights, model.weights, initial_weights)
            ]
            grad_bias += config.proximal_mu * (model.bias - initial_bias)

        model.weights = [
            weight - config.learning_rate * grad
            for weight, grad in zip(model.weights, grad_weights)
        ]
        model.bias -= config.learning_rate * grad_bias

    final_predictions = model.predict_proba(features)
    return binary_cross_entropy(labels, final_predictions)
