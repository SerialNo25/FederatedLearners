"""Stage orchestration for single-institution local training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from domain.dataset.dataset_loader import load_institution_dataset, split_dataset
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
        dataset = load_institution_dataset(
            institution_id=self.config.institution_id,
            csv_path=self.config.dataset_path,
        )
        if not dataset.features or not dataset.labels:
            raise RuntimeError(
                f"Institution dataset '{dataset.institution_id}' is empty; at least one row is required"
            )

        train_dataset, val_dataset = split_dataset(dataset, seed=self.config.seed, val_fraction=0.2)

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(f"local_institution={dataset.institution_id}")
        self.experiment_logger.info(
            f"dataset_split train={len(train_dataset.features)} val={len(val_dataset.features)}"
        )
        self.experiment_logger.info(
            f"fraud_weight={self.config.fraud_weight} classification_threshold={self.config.classification_threshold}"
        )

        torch.manual_seed(self.config.seed)
        model = self.model_factory(len(dataset.features[0]))
        model_device = getattr(model, "device", None)
        if model_device is not None:
            self.experiment_logger.info(f"tabnet_device_selection selected={model_device}")

        final_train_loss = train_local_model(
            model=model,
            features=train_dataset.features,
            labels=train_dataset.labels,
            config=TrainingConfig(
                learning_rate=self.config.learning_rate,
                local_epochs=self.config.local_epochs,
                proximal_mu=0.0,
                fraud_weight=self.config.fraud_weight,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
            ),
        )
        evaluation = evaluate_institution(
            model,
            val_dataset,
            pos_weight=self.config.fraud_weight,
            threshold=self.config.classification_threshold,
        )

        self.experiment_logger.write_metrics(
            step="local_training",
            values={
                "epoch": self.config.local_epochs,
                "train_loss": final_train_loss,
                "val_loss": evaluation.loss,
                "learning_rate": self.config.learning_rate,
                "classification_threshold": self.config.classification_threshold,
                "metrics": {
                    "institution_id": evaluation.institution_id,
                    "val_loss": evaluation.loss,
                    "val_accuracy": evaluation.accuracy,
                    "val_precision": evaluation.precision,
                    "val_recall": evaluation.recall,
                    "val_f1": evaluation.f1,
                }
            },
        )
        self.experiment_logger.info(
            f"local_training_complete institution={evaluation.institution_id} "
            f"val_loss={evaluation.loss:.6f} val_accuracy={evaluation.accuracy:.6f} "
            f"val_precision={evaluation.precision:.6f} val_recall={evaluation.recall:.6f} val_f1={evaluation.f1:.6f}"
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
