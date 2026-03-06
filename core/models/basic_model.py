"""Basic binary classifier used for federated experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LogisticRegressionModel:
    """Simple logistic regression model with list-based parameters."""

    weights: list[float]
    bias: float

    @classmethod
    def initialize(cls, n_features: int) -> "LogisticRegressionModel":
        return cls(weights=[0.0] * n_features, bias=0.0)

    def predict_proba(self, features: list[list[float]]) -> list[float]:
        probabilities: list[float] = []
        for row in features:
            logit = sum(value * weight for value, weight in zip(row, self.weights)) + self.bias
            logit = max(min(logit, 50.0), -50.0)
            probabilities.append(1.0 / (1.0 + math.exp(-logit)))
        return probabilities

    def parameters(self) -> tuple[list[float], float]:
        return self.weights[:], float(self.bias)

    def load_parameters(self, weights: list[float], bias: float) -> None:
        self.weights = [float(weight) for weight in weights]
        self.bias = float(bias)
