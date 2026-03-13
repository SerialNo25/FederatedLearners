"""Reusable inference service and input/output adapters."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from domain.models.model_registry import MODEL_REGISTRY
from domain.training.trainer import binary_cross_entropy


@dataclass(frozen=True)
class InferenceBatch:
    features: list[list[float]]
    labels: list[int] | None


class InferenceDataLoader:
    """Loads and validates CSV rows for inference."""

    def load_csv(
        self,
        *,
        input_data_path: Path,
        feature_columns: list[str],
        label_column: str | None,
    ) -> InferenceBatch:
        if not input_data_path.exists():
            raise FileNotFoundError(f"Inference input data not found: {input_data_path}")

        with input_data_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            required_columns = list(feature_columns)
            if label_column is not None:
                required_columns.append(label_column)

            missing = [column for column in required_columns if column not in header]
            if missing:
                raise ValueError(
                    f"{input_data_path} is missing required inference columns: {missing}. Header columns: {header}"
                )

            features: list[list[float]] = []
            labels: list[int] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    feature_row = [float(row[column]) for column in feature_columns]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{input_data_path}:{row_number} contains non-numeric feature values"
                    ) from exc

                features.append(feature_row)

                if label_column is not None:
                    try:
                        label = int(float(row[label_column]))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{input_data_path}:{row_number} contains non-numeric label values"
                        ) from exc
                    if label not in (0, 1):
                        raise ValueError(
                            f"{input_data_path}:{row_number} contains invalid class label {label}"
                        )
                    labels.append(label)

        if not features:
            raise ValueError("input_data_path must contain at least one data row")

        return InferenceBatch(features=features, labels=labels if label_column is not None else None)


class CheckpointParameterLoader:
    """Loads model parameters from checkpoint payloads."""

    def load(self, *, checkpoint_path: Path, expected_model_type: str) -> dict[str, list[float]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "parameters" in checkpoint:
            checkpoint_model_type = checkpoint.get("model_type")
            if checkpoint_model_type and checkpoint_model_type != expected_model_type:
                raise ValueError(
                    "Checkpoint model_type mismatch: "
                    f"expected {expected_model_type}, got {checkpoint_model_type}"
                )
            return checkpoint["parameters"]

        if isinstance(checkpoint, dict):
            return checkpoint

        # Backward compatibility for legacy JSON checkpoints.
        try:
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "checkpoint_path must contain either a torch-saved checkpoint dict "
                "or a JSON parameter dictionary"
            ) from exc


class InferenceService:
    """Executes model inference and optional label-aware metrics."""

    def run(
        self,
        *,
        model_type: str,
        model_config: dict[str, Any],
        input_batch: InferenceBatch,
        checkpoint_parameters: dict[str, list[float]],
        num_features: int,
    ) -> tuple[list[float], dict[str, float | int | str | None]]:
        model_factory = MODEL_REGISTRY.get_factory(model_type, model_config)
        model = model_factory(num_features)
        model.load_parameters(checkpoint_parameters)
        predictions = model.predict_proba(input_batch.features)

        metrics: dict[str, float | int | str | None] = {
            "num_samples": len(input_batch.features),
            "mean_prediction": sum(predictions) / len(predictions),
            "device": getattr(model, "device", None),
        }
        if input_batch.labels is not None:
            predicted_labels = [1 if value >= 0.5 else 0 for value in predictions]
            matches = sum(
                int(predicted == actual)
                for predicted, actual in zip(predicted_labels, input_batch.labels)
            )
            metrics["loss"] = binary_cross_entropy(input_batch.labels, predictions)
            metrics["accuracy"] = matches / len(input_batch.labels)

        return predictions, metrics

