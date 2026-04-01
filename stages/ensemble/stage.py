"""Stage orchestration for ensemble evaluation of two model checkpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from domain.dataset.dataset_loader import load_institution_dataset
from domain.evaluation_service import EvaluationCheckpointLoader
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.evaluation import evaluate_from_probabilities
from domain.models.model_registry import MODEL_REGISTRY
from stages.ensemble.config import EnsembleConfig
from stages.stage import Stage


class EnsembleStage(Stage):
    def __init__(
        self,
        config: EnsembleConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
        checkpoint_loader: EvaluationCheckpointLoader,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir
        self.checkpoint_loader = checkpoint_loader

    def _load_model(self, checkpoint_path: Path, n_features: int):
        checkpoint = self.checkpoint_loader.load(checkpoint_path)
        model_factory = MODEL_REGISTRY.get_factory(
            checkpoint.model_type,
            checkpoint.model_config,
        )
        model = model_factory(n_features)
        model.load_parameters(checkpoint.parameters)
        return model, checkpoint

    def execute(self) -> Path:
        dataset = load_institution_dataset(
            institution_id="ensemble_dataset",
            csv_path=self.config.dataset_path,
        )

        n_features = len(dataset.features[0])
        local_model, local_ckpt = self._load_model(self.config.local_model_path, n_features)
        fed_model, fed_ckpt = self._load_model(self.config.federated_model_path, n_features)

        local_probs = local_model.predict_proba(dataset.features)
        fed_probs = fed_model.predict_proba(dataset.features)

        w = self.config.ensemble_weight
        ensemble_probs = [
            w * lp + (1 - w) * fp
            for lp, fp in zip(local_probs, fed_probs)
        ]

        metrics = evaluate_from_probabilities(
            institution_id="ensemble_dataset",
            labels=dataset.labels,
            probabilities=ensemble_probs,
            threshold=self.config.classification_threshold,
        )

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(
            f"ensemble_complete "
            f"local_model={local_ckpt.model_type} federated_model={fed_ckpt.model_type} "
            f"ensemble_weight={w} num_samples={len(dataset.features)} "
            f"loss={metrics.loss:.6f} accuracy={metrics.accuracy:.6f} "
            f"precision={metrics.precision:.6f} recall={metrics.recall:.6f} "
            f"f1={metrics.f1:.6f} pr_auc={metrics.pr_auc:.6f} "
            f"roc_auc={metrics.roc_auc:.6f} fpr_at_95_recall={metrics.fpr_at_95_recall:.6f}"
        )

        results = {
            "local_model_path": str(self.config.local_model_path),
            "federated_model_path": str(self.config.federated_model_path),
            "dataset_path": str(self.config.dataset_path),
            "ensemble_weight": w,
            "local_model_type": local_ckpt.model_type,
            "federated_model_type": fed_ckpt.model_type,
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
            step="ensemble",
            values={
                "epoch": 1,
                "train_loss": None,
                "val_loss": metrics.loss,
                "learning_rate": None,
                "ensemble_weight": w,
                "classification_threshold": self.config.classification_threshold,
                "metrics": results["metrics"],
            },
        )
        return self.experiment_dir
