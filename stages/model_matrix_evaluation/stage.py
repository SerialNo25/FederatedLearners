"""Stage orchestration for evaluating configured model runs across datasets."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from domain.dataset.dataset_loader import InstitutionDataset, load_institution_dataset
from domain.evaluation_service import EvaluationCheckpointLoader, ModelEvaluationService
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.evaluation import InstitutionMetrics, evaluate_from_probabilities
from domain.models.model_registry import MODEL_REGISTRY
from stages.model_matrix_evaluation.config import (
    CheckpointRunRef,
    EnsembleRunRef,
    ModelMatrixConfig,
)
from stages.stage import Stage


class ModelMatrixEvaluationStage(Stage):
    def __init__(
        self,
        config: ModelMatrixConfig,
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
        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")

        checkpoint_models = self._checkpoint_models_by_id()
        datasets = [
            load_institution_dataset(
                institution_id=dataset_ref.dataset_id,
                csv_path=dataset_ref.path,
            )
            for dataset_ref in self.config.datasets
        ]

        results: list[dict[str, object]] = []
        for dataset in datasets:
            for model_ref in checkpoint_models.values():
                metrics = self._evaluate_checkpoint_model(model_ref, dataset)
                results.append(
                    self._result_record(
                        model_id=model_ref.model_id,
                        model_group=self._checkpoint_group(model_ref.model_id),
                        dataset=dataset,
                        metrics=metrics,
                        model_path=model_ref.checkpoint_path,
                    )
                )

            for ensemble_ref in self.config.exclusive_ensembles:
                metrics = self._evaluate_ensemble(ensemble_ref, checkpoint_models, dataset)
                results.append(
                    self._result_record(
                        model_id=ensemble_ref.model_id,
                        model_group="exclusive_ensemble",
                        dataset=dataset,
                        metrics=metrics,
                        ensemble=ensemble_ref,
                    )
                )

            for ensemble_ref in self.config.inclusive_ensembles:
                metrics = self._evaluate_ensemble(ensemble_ref, checkpoint_models, dataset)
                results.append(
                    self._result_record(
                        model_id=ensemble_ref.model_id,
                        model_group="inclusive_ensemble",
                        dataset=dataset,
                        metrics=metrics,
                        ensemble=ensemble_ref,
                    )
                )

        (self.experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2),
            encoding="utf-8",
        )
        (self.experiment_dir / "evaluation_matrix.json").write_text(
            json.dumps({"results": results}, indent=2),
            encoding="utf-8",
        )
        self._write_csv(results)

        for result in results:
            self.experiment_logger.write_metrics(
                step="model_matrix_evaluation",
                values={
                    "epoch": 1,
                    "train_loss": None,
                    "val_loss": result["metrics"]["loss"],
                    "learning_rate": None,
                    "model_id": result["model_id"],
                    "model_group": result["model_group"],
                    "dataset_id": result["dataset_id"],
                    "classification_threshold": self.config.classification_threshold,
                    "metrics": result["metrics"],
                },
            )

        self.experiment_logger.info(
            f"model_matrix_evaluation_complete evaluations={len(results)}"
        )
        self._write_run_state("completed")
        return self.experiment_dir

    def _checkpoint_models_by_id(self) -> dict[str, CheckpointRunRef]:
        models = [
            *self.config.local_models,
            self.config.global_federated_model,
            *self.config.exclusive_federated_models,
        ]
        return {model.model_id: model for model in models}

    def _checkpoint_group(self, model_id: str) -> str:
        if model_id == self.config.global_federated_model.model_id:
            return "global_federated"
        exclusive_ids = {model.model_id for model in self.config.exclusive_federated_models}
        if model_id in exclusive_ids:
            return "exclusive_federated"
        return "local"

    def _evaluate_checkpoint_model(
        self,
        model_ref: CheckpointRunRef,
        dataset: InstitutionDataset,
    ) -> InstitutionMetrics:
        checkpoint = self.checkpoint_loader.load(model_ref.checkpoint_path)
        return self.evaluation_service.evaluate(
            checkpoint=checkpoint,
            dataset=dataset,
            classification_threshold=self.config.classification_threshold,
        )

    def _evaluate_ensemble(
        self,
        ensemble_ref: EnsembleRunRef,
        checkpoint_models: dict[str, CheckpointRunRef],
        dataset: InstitutionDataset,
    ) -> InstitutionMetrics:
        n_features = len(dataset.features[0])
        local_model = self._load_model(
            checkpoint_models[ensemble_ref.local_model_id].checkpoint_path,
            n_features,
        )
        federated_model = self._load_model(
            checkpoint_models[ensemble_ref.federated_model_id].checkpoint_path,
            n_features,
        )
        local_probs = local_model.predict_proba(dataset.features)
        federated_probs = federated_model.predict_proba(dataset.features)
        weight = (
            ensemble_ref.ensemble_weight
            if ensemble_ref.ensemble_weight is not None
            else self.config.ensemble_weight
        )
        probabilities = [
            weight * local_prob + (1.0 - weight) * federated_prob
            for local_prob, federated_prob in zip(local_probs, federated_probs)
        ]
        return evaluate_from_probabilities(
            institution_id=dataset.institution_id,
            labels=dataset.labels,
            probabilities=probabilities,
            threshold=self.config.classification_threshold,
        )

    def _load_model(self, checkpoint_path: Path, n_features: int):
        checkpoint = self.checkpoint_loader.load(checkpoint_path)
        model_factory = MODEL_REGISTRY.get_factory(
            checkpoint.model_type,
            checkpoint.model_config,
        )
        model = model_factory(n_features)
        model.load_parameters(checkpoint.parameters)
        return model

    def _result_record(
        self,
        *,
        model_id: str,
        model_group: str,
        dataset: InstitutionDataset,
        metrics: InstitutionMetrics,
        model_path: Path | None = None,
        ensemble: EnsembleRunRef | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "model_id": model_id,
            "model_group": model_group,
            "dataset_id": dataset.institution_id,
            "num_samples": len(dataset.features),
            "classification_threshold": self.config.classification_threshold,
            "metrics": self._metrics_dict(metrics),
        }
        if model_path is not None:
            record["model_path"] = str(model_path)
        if ensemble is not None:
            record["local_model_id"] = ensemble.local_model_id
            record["federated_model_id"] = ensemble.federated_model_id
            record["ensemble_weight"] = (
                ensemble.ensemble_weight
                if ensemble.ensemble_weight is not None
                else self.config.ensemble_weight
            )
        return record

    @staticmethod
    def _metrics_dict(metrics: InstitutionMetrics) -> dict[str, float]:
        return {
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "pr_auc": metrics.pr_auc,
            "roc_auc": metrics.roc_auc,
            "fpr_at_95_recall": metrics.fpr_at_95_recall,
        }

    def _write_csv(self, results: list[dict[str, object]]) -> None:
        fieldnames = [
            "model_id",
            "model_group",
            "dataset_id",
            "num_samples",
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "fpr_at_95_recall",
            "model_path",
            "local_model_id",
            "federated_model_id",
            "ensemble_weight",
        ]
        with (self.experiment_dir / "evaluation_matrix.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                metrics = result["metrics"]
                assert isinstance(metrics, dict)
                writer.writerow(
                    {
                        "model_id": result["model_id"],
                        "model_group": result["model_group"],
                        "dataset_id": result["dataset_id"],
                        "num_samples": result["num_samples"],
                        "loss": metrics["loss"],
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "pr_auc": metrics["pr_auc"],
                        "roc_auc": metrics["roc_auc"],
                        "fpr_at_95_recall": metrics["fpr_at_95_recall"],
                        "model_path": result.get("model_path", ""),
                        "local_model_id": result.get("local_model_id", ""),
                        "federated_model_id": result.get("federated_model_id", ""),
                        "ensemble_weight": result.get("ensemble_weight", ""),
                    }
                )

    def _write_run_state(self, status: str) -> None:
        (self.experiment_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "stage": "model_matrix_evaluation",
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
