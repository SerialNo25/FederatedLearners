"""Stage orchestration for checkpoint-based inference."""

from __future__ import annotations

import json
from pathlib import Path

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
        inference_service: InferenceService,
        data_loader: InferenceDataLoader,
        checkpoint_loader: CheckpointParameterLoader,
    ) -> None:
        self.config = config
        self.inference_service = inference_service
        self.data_loader = data_loader
        self.checkpoint_loader = checkpoint_loader

    def execute(self) -> Path:
        experiment_dir = self.config.output_dir / self.config.experiment_name
        logger = StageExperimentLogger(
            experiment_dir=str(experiment_dir),
            stage_name="inference",
        )

        input_batch = self.data_loader.load_csv(
            input_data_path=self.config.input_data_path,
            feature_columns=self.config.feature_columns,
            label_column=self.config.label_column,
        )

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
            num_features=len(self.config.feature_columns),
        )

        if metrics.get("device"):
            logger.info(f"tabnet_device_selection selected={metrics['device']}")

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
            f"inference_complete num_samples={metrics['num_samples']} labels_available={input_batch.labels is not None}"
        )
        return experiment_dir
