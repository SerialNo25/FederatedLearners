"""Training utilities for local institution optimization."""

from __future__ import annotations

from collections.abc import Callable
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
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> float:
    return _train_torch_model(
        model=model,
        features=features,
        labels=labels,
        config=config,
        global_parameters=global_parameters,
        on_epoch_end=on_epoch_end,
    )


def _train_torch_model(
    model,
    features: list[list[float]],
    labels: list[int],
    config: TrainingConfig,
    global_parameters: dict[str, list[float]] | None,
    on_epoch_end: Callable[[int, float], None] | None,
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
    for epoch_index in tqdm(range(config.local_epochs), desc="Local epochs", leave=False):
        model.train()
        perm = torch.randperm(n, device=model.device)
        epoch_loss = 0.0
        n_batches = 0

        for start in tqdm(range(0, n, config.batch_size), desc="Batches", leave=False):
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
        if on_epoch_end is not None:
            on_epoch_end(epoch_index + 1, final_loss)

    return final_loss
