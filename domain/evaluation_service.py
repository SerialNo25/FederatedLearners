"""Reusable services for loading checkpoints and evaluating persisted models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from domain.dataset.dataset_loader import InstitutionDataset
from domain.metrics.evaluation import InstitutionMetrics, evaluate_institution
from domain.models.model_registry import MODEL_REGISTRY


@dataclass(frozen=True)
class LoadedCheckpoint:
    model_type: str
    parameters: dict[str, list[float]]
    model_config: dict[str, Any]


class EvaluationCheckpointLoader:
    """Loads model checkpoints produced by training stages."""

    def load(self, checkpoint_path: Path) -> LoadedCheckpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "parameters" not in checkpoint:
            raise ValueError(
                "model_path must point to a torch checkpoint containing 'parameters' and 'model_type'"
            )

        model_type = checkpoint.get("model_type")
        if not model_type:
            raise ValueError("checkpoint is missing required 'model_type'")

        model_config = checkpoint.get("model_config") or {"model_type": model_type}
        if "model_type" not in model_config:
            model_config = {**model_config, "model_type": model_type}

        return LoadedCheckpoint(
            model_type=model_type,
            parameters=checkpoint["parameters"],
            model_config=model_config,
        )


class ModelEvaluationService:
    """Builds a model from checkpoint data and evaluates it on a dataset."""

    def evaluate(
        self,
        *,
        checkpoint: LoadedCheckpoint,
        dataset: InstitutionDataset,
        classification_threshold: float,
    ) -> InstitutionMetrics:
        if not dataset.features:
            raise ValueError("dataset must contain at least one feature row")

        model_factory = MODEL_REGISTRY.get_factory(
            checkpoint.model_type,
            checkpoint.model_config,
        )
        model = model_factory(len(dataset.features[0]))
        model.load_parameters(checkpoint.parameters)
        return evaluate_institution(
            model,
            dataset,
            threshold=classification_threshold,
        )
