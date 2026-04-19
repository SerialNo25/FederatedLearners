"""Stage orchestration for evaluating a persisted model checkpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from domain.dataset.dataset_loader import load_institution_dataset
from domain.evaluation_service import EvaluationCheckpointLoader, ModelEvaluationService
from domain.logging.experiment_logger import StageExperimentLogger
from stages.evaluation.config import EvaluationConfig
from stages.stage import Stage


class EvaluationStage(Stage):
    def __init__(
        self,
        config: EvaluationConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
        checkpoint_loader: EvaluationCheckpointLoader,
        evaluation_service: ModelEvaluationService,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir
        self.checkpoint_loader = checkpoint_loader
        self.evaluation_service = evaluation_service

    def execute(self) -> Path:
        self._write_run_state("running")
        dataset = load_institution_dataset(
            institution_id="evaluation_dataset",
            csv_path=self.config.dataset_path,
        )
        checkpoint = self.checkpoint_loader.load(self.config.model_path)
        metrics = self.evaluation_service.evaluate(
            checkpoint=checkpoint,
            dataset=dataset,
            classification_threshold=self.config.classification_threshold,
        )

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(
            "evaluation_complete "
            f"model_type={checkpoint.model_type} num_samples={len(dataset.features)} "
            f"loss={metrics.loss:.6f} accuracy={metrics.accuracy:.6f} "
            f"precision={metrics.precision:.6f} recall={metrics.recall:.6f} "
            f"f1={metrics.f1:.6f} pr_auc={metrics.pr_auc:.6f} "
            f"roc_auc={metrics.roc_auc:.6f} fpr_at_95_recall={metrics.fpr_at_95_recall:.6f}"
        )

        results = {
            "model_path": str(self.config.model_path),
            "dataset_path": str(self.config.dataset_path),
            "model_type": checkpoint.model_type,
            "model_config": checkpoint.model_config,
            "metrics": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "pr_auc": metrics.pr_auc,
                "roc_auc": metrics.roc_auc,
                "fpr_at_95_recall": metrics.fpr_at_95_recall,
            },
        }

        (self.experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (self.experiment_dir / "evaluation.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        self.experiment_logger.write_metrics(
            step="evaluation",
            values={
                "epoch": 1,
                "train_loss": None,
                "val_loss": metrics.loss,
                "learning_rate": None,
                "classification_threshold": self.config.classification_threshold,
                "metrics": results["metrics"],
            },
        )
        self._write_run_state("completed")
        return self.experiment_dir

    def _write_run_state(self, status: str) -> None:
        (self.experiment_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "stage": "evaluation",
                    "status": status,
                    "experiment_name": self.config.experiment_name,
                    "run_id": self.experiment_dir.name,
                    "run_dir": str(self.experiment_dir),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
