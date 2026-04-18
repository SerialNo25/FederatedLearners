"""Training utilities for local institution optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from time import perf_counter

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    local_epochs: int
    proximal_mu: float = 0.0
    fraud_weight: float = 1.0
    batch_size: int = 4096
    seed: int = 42


@dataclass(frozen=True)
class TrainingEpochProfile:
    epoch: int
    samples: int
    batches: int
    batch_size: int
    data_setup_seconds: float
    shuffle_seconds: float
    batch_seconds: float
    forward_backward_seconds: float
    optimizer_step_seconds: float
    loss_read_seconds: float
    callback_seconds: float
    epoch_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "epoch": self.epoch,
            "samples": self.samples,
            "batches": self.batches,
            "batch_size": self.batch_size,
            "data_setup_seconds": self.data_setup_seconds,
            "shuffle_seconds": self.shuffle_seconds,
            "batch_seconds": self.batch_seconds,
            "forward_backward_seconds": self.forward_backward_seconds,
            "optimizer_step_seconds": self.optimizer_step_seconds,
            "loss_read_seconds": self.loss_read_seconds,
            "callback_seconds": self.callback_seconds,
            "epoch_seconds": self.epoch_seconds,
            "samples_per_second": self.samples / max(self.batch_seconds, 1e-12),
            "batches_per_second": self.batches / max(self.batch_seconds, 1e-12),
            "milliseconds_per_batch": 1000.0 * self.batch_seconds / max(self.batches, 1),
        }


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
    on_epoch_profile: Callable[[TrainingEpochProfile], None] | None = None,
) -> float:
    return _train_torch_model(
        model=model,
        features=features,
        labels=labels,
        config=config,
        global_parameters=global_parameters,
        on_epoch_end=on_epoch_end,
        on_epoch_profile=on_epoch_profile,
    )


def _train_torch_model(
    model,
    features: list[list[float]],
    labels: list[int],
    config: TrainingConfig,
    global_parameters: dict[str, list[float]] | None,
    on_epoch_end: Callable[[int, float], None] | None,
    on_epoch_profile: Callable[[TrainingEpochProfile], None] | None,
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

    data_setup_started = perf_counter()
    inputs = torch.tensor(features, dtype=torch.float32, device=model.device)
    targets = torch.tensor(labels, dtype=torch.float32, device=model.device)
    data_setup_seconds = perf_counter() - data_setup_started
    n = inputs.shape[0]

    final_loss = 0.0
    for epoch_index in tqdm(range(config.local_epochs), desc="Local epochs", leave=False):
        epoch_started = perf_counter()
        model.train()
        shuffle_started = perf_counter()
        perm = torch.randperm(n, device=model.device)
        shuffled_inputs = inputs[perm]
        shuffled_targets = targets[perm]
        shuffle_seconds = perf_counter() - shuffle_started
        epoch_loss = torch.zeros((), dtype=torch.float32, device=model.device)
        n_batches = 0
        batch_seconds = 0.0
        forward_backward_seconds = 0.0
        optimizer_step_seconds = 0.0

        for start in range(0, n, config.batch_size):
            batch_started = perf_counter()
            batch_inputs = shuffled_inputs[start : start + config.batch_size]
            batch_targets = shuffled_targets[start : start + config.batch_size]

            optimizer.zero_grad()
            forward_backward_started = perf_counter()
            logits, sparsity_loss = model(batch_inputs)
            clf_loss = criterion(logits, batch_targets)
            total_loss = clf_loss + model.sparsity_weight * sparsity_loss

            if initial_state:
                prox_term = torch.tensor(0.0, device=model.device)
                for name, parameter in model.named_parameters():
                    prox_term = prox_term + torch.sum((parameter - initial_state[name]) ** 2)
                total_loss = total_loss + 0.5 * config.proximal_mu * prox_term

            total_loss.backward()
            forward_backward_seconds += perf_counter() - forward_backward_started
            optimizer_step_started = perf_counter()
            optimizer.step()
            optimizer_step_seconds += perf_counter() - optimizer_step_started
            epoch_loss = epoch_loss + total_loss.detach()
            n_batches += 1
            batch_seconds += perf_counter() - batch_started

        loss_read_started = perf_counter()
        final_loss = float((epoch_loss / max(n_batches, 1)).detach().cpu().item())
        loss_read_seconds = perf_counter() - loss_read_started
        callback_seconds = 0.0
        if on_epoch_end is not None:
            callback_started = perf_counter()
            on_epoch_end(epoch_index + 1, final_loss)
            callback_seconds = perf_counter() - callback_started
        if on_epoch_profile is not None:
            on_epoch_profile(
                TrainingEpochProfile(
                    epoch=epoch_index + 1,
                    samples=n,
                    batches=n_batches,
                    batch_size=config.batch_size,
                    data_setup_seconds=data_setup_seconds if epoch_index == 0 else 0.0,
                    shuffle_seconds=shuffle_seconds,
                    batch_seconds=batch_seconds,
                    forward_backward_seconds=forward_backward_seconds,
                    optimizer_step_seconds=optimizer_step_seconds,
                    loss_read_seconds=loss_read_seconds,
                    callback_seconds=callback_seconds,
                    epoch_seconds=perf_counter() - epoch_started,
                )
            )

    return final_loss
