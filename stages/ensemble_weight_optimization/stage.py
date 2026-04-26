"""Stage orchestration for Optuna-backed ensemble-weight optimization."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import optuna

from domain.dataset.dataset_loader import load_institution_dataset
from domain.evaluation_service import EvaluationCheckpointLoader
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.evaluation import InstitutionMetrics, evaluate_from_probabilities
from domain.models.model_registry import MODEL_REGISTRY
from stages.ensemble_weight_optimization.config import EnsembleWeightOptimizationConfig
from stages.stage import Stage


class EnsembleWeightOptimizationStage(Stage):
    def __init__(
        self,
        config: EnsembleWeightOptimizationConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
        checkpoint_loader: EvaluationCheckpointLoader,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir
        self.checkpoint_loader = checkpoint_loader

    def execute(self) -> Path:
        (self.experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        self._write_run_state("running")

        dataset = load_institution_dataset(
            institution_id=self.config.institution_id,
            csv_path=self.config.dataset_path,
        )
        if not dataset.features or not dataset.labels:
            raise RuntimeError(
                f"Institution dataset '{dataset.institution_id}' is empty; at least one row is required"
            )

        n_features = len(dataset.features[0])
        local_model, local_checkpoint = self._load_model(self.config.local_model.checkpoint_path, n_features)
        federated_model, federated_checkpoint = self._load_model(
            self.config.federated_model.checkpoint_path,
            n_features,
        )
        local_probs = local_model.predict_proba(dataset.features)
        federated_probs = federated_model.predict_proba(dataset.features)

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(
            f"source_models local={self.config.local_model.checkpoint_path} "
            f"federated={self.config.federated_model.checkpoint_path}"
        )
        self.experiment_logger.info(
            f"optuna n_trials={self.config.n_trials} objective={self.config.objective_metric} "
            f"direction={self.config.direction} storage_url={self.config.storage_url}"
        )

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study_name = self.config.study_name or self.config.experiment_name
        study = optuna.create_study(
            study_name=study_name,
            direction=self.config.direction,
            storage=self._prepare_storage_url(),
            load_if_exists=self.config.load_if_exists,
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            weight_space = self.config.search_space.ensemble_weight
            ensemble_weight = trial.suggest_float(
                "ensemble_weight",
                weight_space.low,
                weight_space.high,
                log=weight_space.log,
                step=weight_space.step,
            )
            metrics = self._evaluate_weight(
                dataset_id=dataset.institution_id,
                labels=dataset.labels,
                local_probs=local_probs,
                federated_probs=federated_probs,
                ensemble_weight=ensemble_weight,
            )
            objective_value = self._objective_value(metrics)

            trial.set_user_attr("metrics", self._metrics_dict(metrics))
            self.experiment_logger.write_metrics(
                step="ensemble_weight_optimization",
                values={
                    "trial": trial.number,
                    "epoch": 1,
                    "train_loss": None,
                    "val_loss": metrics.loss,
                    "learning_rate": None,
                    "classification_threshold": self.config.classification_threshold,
                    "ensemble_weight": ensemble_weight,
                    "metrics": self._metrics_dict(metrics),
                },
            )
            self.experiment_logger.info(
                f"trial_complete trial={trial.number} value={objective_value:.6f} "
                f"ensemble_weight={ensemble_weight:.6f} val_loss={metrics.loss:.6f} "
                f"val_pr_auc={metrics.pr_auc:.6f} val_roc_auc={metrics.roc_auc:.6f} "
                f"val_f1={metrics.f1:.6f}"
            )
            return objective_value

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
        )

        best_trial = study.best_trial
        best_metrics = best_trial.user_attrs.get("metrics", {})
        best_payload = {
            "institution_id": dataset.institution_id,
            "dataset_path": str(self.config.dataset_path),
            "local_model": {
                "model_id": self.config.local_model.model_id,
                "checkpoint_path": str(self.config.local_model.checkpoint_path),
                "model_type": local_checkpoint.model_type,
            },
            "federated_model": {
                "model_id": self.config.federated_model.model_id,
                "checkpoint_path": str(self.config.federated_model.checkpoint_path),
                "model_type": federated_checkpoint.model_type,
            },
            "classification_threshold": self.config.classification_threshold,
            "objective_metric": self.config.objective_metric,
            "direction": self.config.direction,
            "best_value": study.best_value,
            "best_trial_number": best_trial.number,
            "best_params": study.best_params,
            "best_trial_metrics": best_metrics,
        }
        (self.experiment_dir / "best_params.json").write_text(
            json.dumps(best_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        self._write_trials_csv(study)
        self.experiment_logger.info(
            f"optimization_complete best_value={study.best_value:.6f} "
            f"best_params={json.dumps(study.best_params, sort_keys=True)}"
        )
        self._write_run_state("completed")
        return self.experiment_dir

    def _load_model(self, checkpoint_path: Path, n_features: int):
        checkpoint = self.checkpoint_loader.load(checkpoint_path)
        model_factory = MODEL_REGISTRY.get_factory(
            checkpoint.model_type,
            checkpoint.model_config,
        )
        model = model_factory(n_features)
        model.load_parameters(checkpoint.parameters)
        return model, checkpoint

    def _evaluate_weight(
        self,
        *,
        dataset_id: str,
        labels: list[int],
        local_probs: list[float],
        federated_probs: list[float],
        ensemble_weight: float,
    ) -> InstitutionMetrics:
        ensemble_probs = [
            ensemble_weight * local_prob + (1.0 - ensemble_weight) * federated_prob
            for local_prob, federated_prob in zip(local_probs, federated_probs)
        ]
        return evaluate_from_probabilities(
            institution_id=dataset_id,
            labels=labels,
            probabilities=ensemble_probs,
            threshold=self.config.classification_threshold,
        )

    def _objective_value(self, metrics: InstitutionMetrics) -> float:
        if self.config.objective_metric == "val_pr_auc":
            return metrics.pr_auc
        if self.config.objective_metric == "val_roc_auc":
            return metrics.roc_auc
        if self.config.objective_metric == "val_f1":
            return metrics.f1
        return metrics.loss

    def _metrics_dict(self, metrics: InstitutionMetrics) -> dict[str, float | str]:
        return {
            "institution_id": metrics.institution_id,
            "val_loss": metrics.loss,
            "val_accuracy": metrics.accuracy,
            "val_precision": metrics.precision,
            "val_recall": metrics.recall,
            "val_f1": metrics.f1,
            "val_pr_auc": metrics.pr_auc,
            "val_roc_auc": metrics.roc_auc,
            "val_fpr_at_95_recall": metrics.fpr_at_95_recall,
        }

    def _prepare_storage_url(self) -> str | None:
        storage_url = self.config.storage_url
        if storage_url is None:
            return None

        sqlite_prefix = "sqlite:///"
        if storage_url.startswith(sqlite_prefix):
            sqlite_path = storage_url.removeprefix(sqlite_prefix)
            if sqlite_path and sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        return storage_url

    def _write_trials_csv(self, study: optuna.Study) -> None:
        path = self.experiment_dir / "optuna_trials.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["number", "state", "value", "ensemble_weight", "metrics"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trial in study.trials:
                writer.writerow(
                    {
                        "number": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "ensemble_weight": trial.params.get("ensemble_weight"),
                        "metrics": json.dumps(trial.user_attrs.get("metrics", {}), sort_keys=True),
                    }
                )

    def _write_run_state(self, status: str) -> None:
        (self.experiment_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "stage": "ensemble_weight_optimization",
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
