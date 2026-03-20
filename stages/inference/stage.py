"""Stage orchestration for checkpoint-based inference."""

from __future__ import annotations

import json
from pathlib import Path

from domain.dataset.schema import FEATURE_COLUMNS, TARGET_COLUMN
from domain.inference.inference_service import (
    CheckpointParameterLoader,
    InferenceDataLoader,
    InferenceService,
)
from domain.logging.experiment_logger import StageExperimentLogger
from stages.inference.config import InferenceConfig
from stages.stage import Stage


class InferenceStage(Stage):
    def __init__(
        self,
        config: InferenceConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
        inference_service: InferenceService,
        data_loader: InferenceDataLoader,
        checkpoint_loader: CheckpointParameterLoader,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir
        self.inference_service = inference_service
        self.data_loader = data_loader
        self.checkpoint_loader = checkpoint_loader

    def execute(self) -> Path:
        input_batch = self.data_loader.load_csv(input_data_path=self.config.input_data_path)

        model_config = self.config.to_dict()

        checkpoint_parameters = self.checkpoint_loader.load(
            checkpoint_path=self.config.checkpoint_path,
            expected_model_type=self.config.model_type,
        )
        predictions, metrics = self.inference_service.run(
            model_type=self.config.model_type,
            model_config=model_config,
            input_batch=input_batch,
            checkpoint_parameters=checkpoint_parameters,
            num_features=len(FEATURE_COLUMNS),
        )

        if metrics.get("device"):
            self.experiment_logger.info(f"tabnet_device_selection selected={metrics['device']}")

        outputs = {
            "checkpoint_path": str(self.config.checkpoint_path),
            "input_data_path": str(self.config.input_data_path),
            "model_type": self.config.model_type,
            "feature_columns": FEATURE_COLUMNS,
            "label_column": TARGET_COLUMN if input_batch.labels is not None else None,
            "predictions": predictions,
            "metrics": metrics,
        }
        (self.experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (self.experiment_dir / "predictions.json").write_text(
            json.dumps(outputs, indent=2), encoding="utf-8"
        )
        self.experiment_logger.write_metrics(
            step="inference",
            values={
                "epoch": 1,
                "train_loss": None,
                "val_loss": metrics.get("loss"),
                "metrics": metrics,
                "learning_rate": None,
            },
        )
        self.experiment_logger.info(
            f"inference_complete num_samples={metrics['num_samples']} labels_available={input_batch.labels is not None}"
        )
        return self.experiment_dir
