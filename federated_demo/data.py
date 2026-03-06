from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class InstitutionData:
    train_loader: DataLoader
    test_loader: DataLoader


def _make_pool(institution_id: int, num_samples: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic non-IID data pool for a specific institution."""
    rng = np.random.default_rng(seed=2024 + institution_id)

    # Institution-specific shift to simulate different local distributions.
    shift_x = (institution_id - 1) * 0.8
    shift_y = (1 - institution_id) * 0.6

    x = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 2)).astype(np.float32)
    x[:, 0] += shift_x
    x[:, 1] += shift_y

    # Shared nonlinear decision boundary (same concept, shifted local populations).
    logits = x[:, 0] * 0.7 - x[:, 1] * 0.4 + 0.2 * np.sin(x[:, 0])
    y = (logits > 0).astype(np.int64)

    return x, y


def load_institution_data(institution_id: int, batch_size: int = 32) -> InstitutionData:
    x, y = _make_pool(institution_id)
    split_idx = int(0.8 * len(x))

    x_train = torch.from_numpy(x[:split_idx])
    y_train = torch.from_numpy(y[:split_idx])
    x_test = torch.from_numpy(x[split_idx:])
    y_test = torch.from_numpy(y[split_idx:])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return InstitutionData(train_loader=train_loader, test_loader=test_loader)
