"""Stage orchestration for single-institution local training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from domain.dataset.dataset_loader import load_institution_dataset
from domain.federated.model_artifact_writer import ModelArtifactWriter
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.evaluation import evaluate_institution
from domain.models.model_registry import ModelFactoryProtocol
from domain.training.trainer import TrainingConfig, train_local_model
from stages.local_training.config import LocalTrainingConfig
from stages.stage import Stage


class LocalTrainingStage(Stage):
    def __init__(
        self,
        config: LocalTrainingConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
        model_factory: ModelFactoryProtocol,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir
        self.model_factory = model_factory

    def execute(self) -> Path:
        selected_institution = self.config.selected_institution
        dataset = load_institution_dataset(
            institution_id=selected_institution.institution_id,
            csv_path=selected_institution.dataset_path,
        )
        if not dataset.features or not dataset.labels:
            raise RuntimeError(
                f"Institution dataset '{dataset.institution_id}' is empty; at least one row is required"
            )

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(f"local_institution={dataset.institution_id}")

        model = self.model_factory(len(dataset.features[0]))
        model_device = getattr(model, "device", None)
        if model_device is not None:
            self.experiment_logger.info(f"tabnet_device_selection selected={model_device}")

        final_train_loss = train_local_model(
            model=model,
            features=dataset.features,
            labels=dataset.labels,
            config=TrainingConfig(
                learning_rate=self.config.learning_rate,
                local_epochs=self.config.local_epochs,
                proximal_mu=0.0,
            ),
        )
        evaluation = evaluate_institution(model, dataset)

        self.experiment_logger.write_metrics(
            step="local_training",
            values={
                "epoch": self.config.local_epochs,
                "train_loss": final_train_loss,
                "val_loss": evaluation.loss,
                "metrics": {
                    "institution_id": evaluation.institution_id,
                    "loss": evaluation.loss,
                    "accuracy": evaluation.accuracy,
                },
                "learning_rate": self.config.learning_rate,
            },
        )
        self.experiment_logger.info(
            f"local_training_complete institution={evaluation.institution_id} "
            f"loss={evaluation.loss:.6f} accuracy={evaluation.accuracy:.6f}"
        )

        (self.experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        ModelArtifactWriter.write_model_checkpoint(
            checkpoint_path=self.experiment_dir / "model.pt",
            model_type=self.config.model.model_type,
            model=model,
        )
        return self.experiment_dir
