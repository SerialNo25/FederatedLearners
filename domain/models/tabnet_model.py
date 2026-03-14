"""TabNet-based binary classifier for federated fraud experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _FeatureTransformer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _AttentiveTransformer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: Tensor, prior: Tensor) -> Tensor:
        scores = self.bn(self.fc(x))
        return torch.softmax(scores * prior, dim=-1)


class _TabNetBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        decision_dim: int,
        attention_dim: int,
        n_steps: int,
        relaxation_factor: float,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.relaxation_factor = relaxation_factor

        self.initial = _FeatureTransformer(n_features, (decision_dim + attention_dim) * 2)
        self.feature_transformers = nn.ModuleList(
            _FeatureTransformer(n_features, (decision_dim + attention_dim) * 2)
            for _ in range(n_steps)
        )
        self.attentive_transformers = nn.ModuleList(
            _AttentiveTransformer(attention_dim, n_features) for _ in range(n_steps)
        )
        self.projection = nn.Linear(decision_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        prior = torch.ones_like(x)
        aggregated_decision = torch.zeros((x.shape[0], self.projection.in_features), device=x.device)
        sparsity_penalty = torch.tensor(0.0, device=x.device)

        initial_out = self.initial(x)
        _, attention = torch.chunk(initial_out, 2, dim=1)

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](attention, prior)
            masked_x = x * mask

            transformed = self.feature_transformers[step](masked_x)
            decision, attention = torch.chunk(transformed, 2, dim=1)
            decision = torch.relu(decision)
            aggregated_decision = aggregated_decision + decision

            prior = (self.relaxation_factor - mask) * prior
            sparsity_penalty = sparsity_penalty + torch.mean(
                torch.sum(mask * torch.log(mask + 1e-10), dim=1)
            )

        logits = self.projection(aggregated_decision).squeeze(-1)
        return logits, sparsity_penalty / self.n_steps


class TabNetModel(nn.Module):
    """TabNet-style binary classifier with federated-parameter helpers."""

    def __init__(self, network: _TabNetBackbone, sparsity_weight: float = 1e-4, device: str = "cpu") -> None:
        super().__init__()
        self.network = network
        self.sparsity_weight = sparsity_weight
        self.device = device

    @classmethod
    def initialize(
        cls,
        n_features: int,
        decision_dim: int = 16,
        attention_dim: int = 16,
        n_steps: int = 3,
        relaxation_factor: float = 1.5,
        sparsity_weight: float = 1e-4,
        device: str = "cpu",
    ) -> "TabNetModel":
        torch.manual_seed(42)
        network = _TabNetBackbone(
            n_features=n_features,
            decision_dim=decision_dim,
            attention_dim=attention_dim,
            n_steps=n_steps,
            relaxation_factor=relaxation_factor,
        ).to(device)
        return cls(network=network, sparsity_weight=sparsity_weight, device=device)

    def predict_proba(self, features: list[list[float]]) -> list[float]:
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(features, dtype=torch.float32, device=self.device)
            logits, _ = self(inputs)
            probabilities = torch.sigmoid(logits)
            return probabilities.detach().cpu().tolist()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.network(x)

    def federated_parameters(self) -> dict[str, list[float]]:
        state = self.state_dict()
        return {name: value.detach().cpu().flatten().tolist() for name, value in state.items()}

    def load_parameters(self, parameters: dict[str, list[float]]) -> None:
        current_state = self.state_dict()
        restored: dict[str, Tensor] = {}

        for name, tensor in current_state.items():
            restored[name] = torch.tensor(
                parameters[name], dtype=tensor.dtype, device=tensor.device
            ).reshape(tensor.shape)
        self.load_state_dict(restored)
