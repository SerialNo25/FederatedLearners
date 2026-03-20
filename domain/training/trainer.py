"""Training utilities for local institution optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    local_epochs: int
    proximal_mu: float = 0.0
    fraud_weight: float = 1.0
    batch_size: int = 256
    seed: int = 42


def binary_cross_entropy(
    y_true: list[int] | list[float],
    y_prob: list[float],
    pos_weight: float = 1.0,
) -> float:
    eps = 1e-7
    losses: list[float] = []
    for label, probability in zip(y_true, y_prob):
        probability = min(max(probability, eps), 1.0 - eps)
        sample_weight = pos_weight if label == 1 else 1.0
        losses.append(-sample_weight * (label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)))
    return sum(losses) / max(len(losses), 1)


def train_local_model(
    model,
    features: list[list[float]],
    labels: list[int],
    config: TrainingConfig,
    global_parameters: dict[str, list[float]] | None = None,
) -> float:
    if hasattr(model, "network"):
        return _train_torch_model(
            model=model,
            features=features,
            labels=labels,
            config=config,
            global_parameters=global_parameters,
        )

    return _train_manual_model(
        model=model,
        features=features,
        labels=labels,
        config=config,
        global_parameters=global_parameters,
    )


def _train_torch_model(
    model,
    features: list[list[float]],
    labels: list[int],
    config: TrainingConfig,
    global_parameters: dict[str, list[float]] | None,
) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    pos_weight = torch.tensor([config.fraud_weight], dtype=torch.float32, device=model.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    initial_state: dict[str, Tensor] = {}
    if global_parameters is not None and config.proximal_mu > 0.0:
        current_state = model.state_dict()
        for name, tensor in current_state.items():
            initial_state[name] = torch.tensor(
                global_parameters[name], dtype=tensor.dtype, device=tensor.device
            ).reshape(tensor.shape)

    torch.manual_seed(config.seed)

    inputs = torch.tensor(features, dtype=torch.float32, device=model.device)
    targets = torch.tensor(labels, dtype=torch.float32, device=model.device)
    n = inputs.shape[0]

    final_loss = 0.0
    for _ in tqdm(range(config.local_epochs), desc="Local epochs", leave=False):
        perm = torch.randperm(n, device=model.device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch_inputs = inputs[idx]
            batch_targets = targets[idx]

            optimizer.zero_grad()
            logits, sparsity_loss = model(batch_inputs)
            clf_loss = criterion(logits, batch_targets)
            total_loss = clf_loss + model.sparsity_weight * sparsity_loss

            if initial_state:
                prox_term = torch.tensor(0.0, device=model.device)
                for name, parameter in model.named_parameters():
                    prox_term = prox_term + torch.sum((parameter - initial_state[name]) ** 2)
                total_loss = total_loss + 0.5 * config.proximal_mu * prox_term

            total_loss.backward()
            optimizer.step()
            epoch_loss += float(total_loss.detach().cpu().item())
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)

    return final_loss


def _train_manual_model(
    model,
    features: list[list[float]],
    labels: list[int],
    config: TrainingConfig,
    global_parameters: dict[str, list[float]] | None,
) -> float:
    initial_parameters = global_parameters if global_parameters is not None else model.parameters()
    initial_weights = initial_parameters["weights"]
    initial_bias = initial_parameters["bias"][0]

    n_samples = len(features)
    n_features = len(features[0]) if features else 0

    if n_samples == 0:
        return 0.0

    for _ in tqdm(range(config.local_epochs), desc="Local epochs", leave=False):
        predictions = model.predict_proba(features)
        grad_weights = [0.0] * n_features
        grad_bias = 0.0

        for row, label, prediction in zip(features, labels, predictions):
            sample_weight = config.fraud_weight if label == 1 else 1.0
            error = sample_weight * (prediction - float(label))
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
    return binary_cross_entropy(labels, final_predictions, pos_weight=config.fraud_weight)
