"""Stage orchestration for checkpoint-based inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from domain.logging.experiment_logger import StageExperimentLogger
from domain.models.device_selector import DeviceSelector
from domain.models.model_registry import MODEL_REGISTRY
from domain.training.trainer import binary_cross_entropy
from stages.inference.config import InferenceConfig


class InferenceStage:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def execute(self) -> Path:
        experiment_dir = self.config.output_dir / self.config.experiment_name
        logger = StageExperimentLogger(
            experiment_dir=str(experiment_dir),
            stage_name="inference",
        )

        features, labels = self._load_input_rows()

        model_config = self.config.to_dict()
        if self.config.model_type == "tabnet" and not self.config.tabnet_device:
            selector = DeviceSelector()
            model_config["tabnet_device"] = selector.select_best_device()
            logger.info(
                "tabnet_device_selection selected=%s available=%s"
                % (model_config["tabnet_device"], ",".join(selector.available_devices()))
            )

        model_factory = MODEL_REGISTRY.get_factory(self.config.model_type, model_config)
        model = model_factory(len(self.config.feature_columns))

        checkpoint = json.loads(self.config.checkpoint_path.read_text(encoding="utf-8"))
        model.load_parameters(checkpoint)
        predictions = model.predict_proba(features)

        metrics: dict[str, float | int | None] = {
            "num_samples": len(features),
            "mean_prediction": sum(predictions) / len(predictions),
        }
        if labels is not None:
            predicted_labels = [1 if value >= 0.5 else 0 for value in predictions]
            matches = sum(int(predicted == actual) for predicted, actual in zip(predicted_labels, labels))
            metrics["loss"] = binary_cross_entropy(labels, predictions)
            metrics["accuracy"] = matches / len(labels)

        outputs = {
            "checkpoint_path": str(self.config.checkpoint_path),
            "input_data_path": str(self.config.input_data_path),
            "model_type": self.config.model_type,
            "feature_columns": self.config.feature_columns,
            "label_column": self.config.label_column,
            "predictions": predictions,
            "metrics": metrics,
        }
        (experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (experiment_dir / "predictions.json").write_text(
            json.dumps(outputs, indent=2), encoding="utf-8"
        )
        logger.write_metrics(
            step="inference",
            values={
                "epoch": 1,
                "train_loss": None,
                "val_loss": metrics.get("loss"),
                "metrics": metrics,
                "learning_rate": None,
            },
        )
        logger.info(
            f"inference_complete num_samples={metrics['num_samples']} labels_available={labels is not None}"
        )
        return experiment_dir

    def _load_input_rows(self) -> tuple[list[list[float]], list[int] | None]:
        path = self.config.input_data_path
        if not path.exists():
            raise FileNotFoundError(f"Inference input data not found: {path}")

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            required_columns = list(self.config.feature_columns)
            if self.config.label_column is not None:
                required_columns.append(self.config.label_column)

            missing = [column for column in required_columns if column not in header]
            if missing:
                raise ValueError(
                    f"{path} is missing required inference columns: {missing}. Header columns: {header}"
                )

            features: list[list[float]] = []
            labels: list[int] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    feature_row = [float(row[column]) for column in self.config.feature_columns]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{row_number} contains non-numeric feature values"
                    ) from exc

                features.append(feature_row)

                if self.config.label_column is not None:
                    try:
                        label = int(float(row[self.config.label_column]))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{path}:{row_number} contains non-numeric label values"
                        ) from exc
                    if label not in (0, 1):
                        raise ValueError(
                            f"{path}:{row_number} contains invalid class label {label}"
                        )
                    labels.append(label)

        if not features:
            raise ValueError("input_data_path must contain at least one data row")

        return features, labels if self.config.label_column is not None else None
