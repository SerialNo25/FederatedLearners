"""Persistence helpers for federated model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from domain.federated.model_parameters import get_model_parameters


class ModelArtifactWriter:
    """Writes serialized model checkpoints for federated training outputs."""

    @staticmethod
    def write_model_checkpoint(checkpoint_path: Path, model_type: str, model: Any) -> None:
        torch.save(
            {
                "model_type": model_type,
                "parameters": get_model_parameters(model),
            },
            checkpoint_path,
        )
