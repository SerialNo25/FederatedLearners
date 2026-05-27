"""Stage orchestration for Optuna-backed local hyperparameter optimization."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import torch

from domain.dataset.dataset_loader import load_institution_dataset, split_dataset
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.evaluation import InstitutionMetrics, evaluate_institution
from domain.models.model_registry import MODEL_REGISTRY
from domain.training.class_weighting import compute_binary_class_balance
from domain.training.trainer import TrainingConfig, train_local_model
from stages.hyperparameter_optimization.config import (
    FloatSearchSpace,
    HyperparameterOptimizationConfig,
    IntChoices,
    IntSearchSpace,
)
from stages.stage import Stage


class HyperparameterOptimizationStage(Stage):
    def __init__(
        self,
        config: HyperparameterOptimizationConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir

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

        train_dataset, val_dataset = split_dataset(
            dataset,
            seed=self.config.seed,
            val_fraction=self.config.validation_fraction,
        )
        class_balance = compute_binary_class_balance(train_dataset.labels)

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(f"local_institution={dataset.institution_id}")
        self.experiment_logger.info(
            f"validation_split train={len(train_dataset.features)} val={len(val_dataset.features)}"
        )
        self.experiment_logger.info(
            "train_class_balance "
            f"negatives={class_balance.negatives} positives={class_balance.positives} "
            f"positive_rate={class_balance.positive_rate:.6f} "
            f"recommended_pos_weight={class_balance.pos_weight:.6f}"
        )
        self._warn_if_fraud_weight_search_misses_balance(class_balance.pos_weight)
        self.experiment_logger.info(
            f"optuna n_trials={self.config.n_trials} objective={self.config.objective_metric} "
            f"direction={self.config.direction} storage_url={self.config.storage_url}"
        )

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
        study_name = self.config.study_name or self.config.experiment_name
        study = optuna.create_study(
            study_name=study_name,
            direction=self.config.direction,
            storage=self._prepare_storage_url(),
            load_if_exists=self.config.load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            training_params = self._sample_training_params(trial)
            model_params = self._sample_model_params(trial)
            model_factory = MODEL_REGISTRY.get_factory(
                model_params["model_type"],
                model_params,
            )

            torch.manual_seed(self.config.seed + trial.number)
            model = model_factory(len(dataset.features[0]))
            model_device = getattr(model, "device", None)
            if model_device is not None:
                self.experiment_logger.info(
                    f"trial={trial.number} tabnet_device_selection selected={model_device}"
                )

            last_epoch_metric: float | None = None

            def record_epoch_metrics(epoch: int, train_loss: float) -> None:
                nonlocal last_epoch_metric
                evaluation = evaluate_institution(
                    model,
                    val_dataset,
                    pos_weight=training_params["fraud_weight"],
                    threshold=training_params["classification_threshold"],
                )
                last_epoch_metric = self._objective_value(evaluation)
                trial.report(last_epoch_metric, step=epoch)
                self.experiment_logger.write_metrics(
                    step="hyperparameter_optimization",
                    values={
                        "trial": trial.number,
                        "epoch": epoch,
                        "total_epochs": training_params["local_epochs"],
                        "train_loss": train_loss,
                        "val_loss": evaluation.loss,
                        "learning_rate": training_params["learning_rate"],
                        "classification_threshold": training_params["classification_threshold"],
                        "metrics": self._metrics_dict(evaluation),
                        "params": {**training_params, **self._public_model_params(model_params)},
                    },
                )
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"trial={trial.number} pruned at epoch={epoch} "
                        f"{self.config.objective_metric}={last_epoch_metric:.6f}"
                    )

            try:
                final_train_loss = train_local_model(
                    model=model,
                    features=train_dataset.features,
                    labels=train_dataset.labels,
                    config=TrainingConfig(
                        learning_rate=training_params["learning_rate"],
                        local_epochs=training_params["local_epochs"],
                        proximal_mu=0.0,
                        fraud_weight=training_params["fraud_weight"],
                        batch_size=training_params["batch_size"],
                        seed=self.config.seed + trial.number,
                    ),
                    on_epoch_end=record_epoch_metrics,
                )
                evaluation = evaluate_institution(
                    model,
                    val_dataset,
                    pos_weight=training_params["fraud_weight"],
                    threshold=training_params["classification_threshold"],
                )
            except RuntimeError as exc:
                if not self._is_invalid_trial_error(exc):
                    raise
                message = str(exc)
                trial.set_user_attr("failure", message)
                self.experiment_logger.warning(
                    f"trial_pruned_invalid_configuration trial={trial.number} error={message}"
                )
                raise optuna.TrialPruned(message) from exc

            objective_value = self._objective_value(evaluation)
            trial.set_user_attr("train_loss", final_train_loss)
            trial.set_user_attr("metrics", self._metrics_dict(evaluation))
            trial.set_user_attr("model_params", self._public_model_params(model_params))

            self.experiment_logger.info(
                f"trial_complete trial={trial.number} value={objective_value:.6f} "
                f"train_loss={final_train_loss:.6f} val_loss={evaluation.loss:.6f} "
                f"val_pr_auc={evaluation.pr_auc:.6f} val_roc_auc={evaluation.roc_auc:.6f} "
                f"val_f1={evaluation.f1:.6f} params={json.dumps(trial.params, sort_keys=True)}"
            )
            return objective_value

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
        )

        self._write_trials_csv(study)
        self._write_best_params(study)
        self.experiment_logger.info(
            f"optimization_complete best_value={study.best_value:.6f} "
            f"best_params={json.dumps(study.best_params, sort_keys=True)}"
        )
        self._write_run_state("completed")
        return self.experiment_dir

    def _sample_training_params(self, trial: optuna.Trial) -> dict[str, Any]:
        search = self.config.search_space.training
        return {
            "learning_rate": self._suggest_float(
                trial,
                "learning_rate",
                search.learning_rate,
                self.config.learning_rate,
            ),
            "fraud_weight": self._suggest_float(
                trial,
                "fraud_weight",
                search.fraud_weight,
                self.config.fraud_weight,
            ),
            "local_epochs": self._suggest_int(
                trial,
                "local_epochs",
                search.local_epochs,
                self.config.local_epochs,
            ),
            "batch_size": self._suggest_int_choice(
                trial,
                "batch_size",
                search.batch_size,
                self.config.batch_size,
            ),
            "classification_threshold": self._suggest_float(
                trial,
                "classification_threshold",
                search.classification_threshold,
                self.config.classification_threshold,
            ),
        }

    def _sample_model_params(self, trial: optuna.Trial) -> dict[str, Any]:
        model_params = self.config.model.model_dump(mode="python")
        if model_params["model_type"] != "tabnet":
            return model_params

        search = self.config.search_space.tabnet
        model_params["decision_dim"] = self._suggest_int_choice(
            trial,
            "tabnet_decision_dim",
            search.decision_dim,
            int(model_params.get("decision_dim", 16)),
        )
        model_params["attention_dim"] = self._suggest_int_choice(
            trial,
            "tabnet_attention_dim",
            search.attention_dim,
            int(model_params.get("attention_dim", 16)),
        )
        model_params["steps"] = self._suggest_int(
            trial,
            "tabnet_steps",
            search.steps,
            int(model_params.get("steps", 3)),
        )
        model_params["relaxation_factor"] = self._suggest_float(
            trial,
            "tabnet_relaxation_factor",
            search.relaxation_factor,
            float(model_params.get("relaxation_factor", 1.5)),
        )
        model_params["sparsity_weight"] = self._suggest_float(
            trial,
            "tabnet_sparsity_weight",
            search.sparsity_weight,
            float(model_params.get("sparsity_weight", 1e-4)),
        )
        return model_params

    def _suggest_float(
        self,
        trial: optuna.Trial,
        name: str,
        space: FloatSearchSpace | None,
        default: float,
    ) -> float:
        if space is None:
            return default
        return trial.suggest_float(
            name,
            space.low,
            space.high,
            log=space.log,
            step=space.step,
        )

    def _suggest_int(
        self,
        trial: optuna.Trial,
        name: str,
        space: IntSearchSpace | None,
        default: int,
    ) -> int:
        if space is None:
            return default
        return trial.suggest_int(
            name,
            space.low,
            space.high,
            step=space.step,
            log=space.log,
        )

    def _suggest_int_choice(
        self,
        trial: optuna.Trial,
        name: str,
        space: IntChoices | None,
        default: int,
    ) -> int:
        if space is None:
            return default
        return int(trial.suggest_categorical(name, space.choices))

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

    def _public_model_params(self, model_params: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in model_params.items()
            if key != "model_type"
        }

    def _is_invalid_trial_error(self, exc: RuntimeError) -> bool:
        message = str(exc)
        invalid_trial_patterns = [
            "mat1 and mat2 shapes cannot be multiplied",
            "input and weight.T shapes cannot be multiplied",
            "Expected more than 1 value per channel",
        ]
        return any(pattern in message for pattern in invalid_trial_patterns)

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

    def _warn_if_fraud_weight_search_misses_balance(self, recommended_pos_weight: float) -> None:
        search_space = self.config.search_space.training.fraud_weight
        if search_space is None:
            if self.config.fraud_weight < 0.5 * recommended_pos_weight:
                self.experiment_logger.warning(
                    "fraud_weight_below_recommended "
                    f"configured={self.config.fraud_weight:.6f} "
                    f"recommended_pos_weight={recommended_pos_weight:.6f}"
                )
            return

        if not search_space.low <= recommended_pos_weight <= search_space.high:
            self.experiment_logger.warning(
                "fraud_weight_search_excludes_recommended "
                f"low={search_space.low:.6f} high={search_space.high:.6f} "
                f"recommended_pos_weight={recommended_pos_weight:.6f}"
            )

    def _write_trials_csv(self, study: optuna.Study) -> None:
        path = self.experiment_dir / "optuna_trials.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "number",
                "state",
                "value",
                "params",
                "metrics",
                "model_params",
                "train_loss",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trial in study.trials:
                writer.writerow(
                    {
                        "number": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "params": json.dumps(trial.params, sort_keys=True),
                        "metrics": json.dumps(trial.user_attrs.get("metrics", {}), sort_keys=True),
                        "model_params": json.dumps(
                            trial.user_attrs.get("model_params", {}), sort_keys=True
                        ),
                        "train_loss": trial.user_attrs.get("train_loss"),
                    }
                )

    def _write_best_params(self, study: optuna.Study) -> None:
        best_payload = {
            "objective_metric": self.config.objective_metric,
            "direction": self.config.direction,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "best_trial_metrics": study.best_trial.user_attrs.get("metrics", {}),
            "best_trial_model_params": study.best_trial.user_attrs.get("model_params", {}),
        }
        (self.experiment_dir / "best_params.json").write_text(
            json.dumps(best_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_run_state(self, status: str) -> None:
        (self.experiment_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "stage": "hyperparameter_optimization",
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
