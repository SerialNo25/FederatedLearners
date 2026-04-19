"""Stage orchestration for Optuna-backed federated hyperparameter optimization."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from domain.logging.experiment_logger import StageExperimentLogger
from domain.models.model_registry import MODEL_REGISTRY
from stages.federated_hyperparameter_optimization.config import (
    FederatedHyperparameterOptimizationConfig,
)
from stages.federated_training.config import (
    FederatedLocalTrainingOverrides,
    FederatedTrainingConfig,
)
from stages.federated_training.stage import FederatedTrainingStage
from stages.hyperparameter_optimization.config import (
    FloatSearchSpace,
    IntChoices,
    IntSearchSpace,
)
from stages.stage import Stage


class FederatedHyperparameterOptimizationStage(Stage):
    def __init__(
        self,
        config: FederatedHyperparameterOptimizationConfig,
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

        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
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
            federated_params = self._sample_federated_params(trial)
            local_training_params = self._sample_local_training_params(trial)
            model_params = self._sample_model_params(trial)
            trial_dir = self._trial_dir(trial.number)
            trial_logger = StageExperimentLogger(
                experiment_dir=str(trial_dir),
                stage_name="federated_hyperparameter_optimization_trial",
            )
            federated_config = FederatedTrainingConfig(
                stage="federated_training",
                experiment_name=f"{self.config.experiment_name}_trial_{trial.number:03d}",
                output_dir=self.config.output_dir,
                num_rounds=federated_params["num_rounds"],
                proximal_mu=federated_params["proximal_mu"],
                local_training_overrides={
                    institution_id: FederatedLocalTrainingOverrides(**params)
                    for institution_id, params in local_training_params.items()
                },
                model=model_params,
                institutions=self.config.institutions,
            )
            model_factory = MODEL_REGISTRY.get_factory(
                federated_config.model.model_type,
                federated_config.model.model_dump(mode="python"),
            )
            stage = FederatedTrainingStage(
                config=federated_config,
                experiment_logger=trial_logger,
                experiment_dir=trial_dir,
                model_factory=model_factory,
            )

            try:
                stage.execute()
                final_metrics = self._read_final_metrics(trial_dir)
            except RuntimeError as exc:
                if not self._is_invalid_trial_error(exc):
                    raise
                message = str(exc)
                trial.set_user_attr("failure", message)
                self.experiment_logger.warning(
                    f"trial_pruned_invalid_configuration trial={trial.number} error={message}"
                )
                raise optuna.TrialPruned(message) from exc

            objective_value = self._objective_value(final_metrics)
            trial.set_user_attr("metrics", final_metrics)
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("model_params", self._public_model_params(model_params))

            self.experiment_logger.write_metrics(
                step="federated_hyperparameter_optimization",
                values={
                    "trial": trial.number,
                    "epoch": final_metrics.get("epoch"),
                    "train_loss": final_metrics.get("train_loss"),
                    "val_loss": final_metrics.get("val_loss"),
                    "learning_rate": {
                        institution_id: params["learning_rate"]
                        for institution_id, params in local_training_params.items()
                    },
                    "metrics": final_metrics.get("metrics", {}),
                    "params": {
                        **federated_params,
                        "local_training": local_training_params,
                        **self._public_model_params(model_params),
                    },
                    "objective_value": objective_value,
                    "trial_dir": str(trial_dir),
                },
            )
            self.experiment_logger.info(
                f"trial_complete trial={trial.number} value={objective_value:.6f} "
                f"params={json.dumps(trial.params, sort_keys=True)} trial_dir={trial_dir}"
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

    def _sample_federated_params(self, trial: optuna.Trial) -> dict[str, Any]:
        search = self.config.search_space.federated
        return {
            "proximal_mu": self._suggest_float(
                trial,
                "proximal_mu",
                search.proximal_mu,
                self.config.proximal_mu,
            ),
            "num_rounds": self._suggest_int(
                trial,
                "num_rounds",
                search.num_rounds,
                self.config.num_rounds,
            ),
        }

    def _sample_local_training_params(self, trial: optuna.Trial) -> dict[str, Any]:
        search = self.config.search_space.local_training
        params_by_institution: dict[str, dict[str, Any]] = {}
        for institution in self.config.institutions:
            defaults = self.config.local_training_overrides.get(
                institution.institution_id,
                FederatedLocalTrainingOverrides(),
            )
            prefix = institution.institution_id
            params_by_institution[institution.institution_id] = {
                "learning_rate": self._suggest_float(
                    trial,
                    f"{prefix}_learning_rate",
                    search.learning_rate,
                    defaults.learning_rate or institution.learning_rate,
                ),
                "fraud_weight": self._suggest_float(
                    trial,
                    f"{prefix}_fraud_weight",
                    search.fraud_weight,
                    defaults.fraud_weight or institution.fraud_weight,
                ),
                "local_epochs": self._suggest_int(
                    trial,
                    f"{prefix}_local_epochs",
                    search.local_epochs,
                    defaults.local_epochs or institution.local_epochs,
                ),
                "batch_size": self._suggest_int_choice(
                    trial,
                    f"{prefix}_batch_size",
                    search.batch_size,
                    defaults.batch_size or institution.batch_size,
                ),
                "classification_threshold": self._suggest_float(
                    trial,
                    f"{prefix}_classification_threshold",
                    search.classification_threshold,
                    defaults.classification_threshold
                    or institution.classification_threshold,
                ),
            }
        return params_by_institution

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

    def _objective_value(self, metrics: dict[str, Any]) -> float:
        if self.config.objective_metric == "val_pr_auc":
            return float(metrics["pr_auc"])
        return float(metrics["val_loss"])

    def _read_final_metrics(self, trial_dir: Path) -> dict[str, Any]:
        metrics_path = trial_dir / "metrics.jsonl"
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise RuntimeError(f"No federated metrics were written for trial_dir={trial_dir}")
        return records[-1]

    def _trial_dir(self, trial_number: int) -> Path:
        trial_dir = self.experiment_dir / f"trial_{trial_number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

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
                "trial_dir",
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
                        "trial_dir": trial.user_attrs.get("trial_dir"),
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
            "best_trial_dir": study.best_trial.user_attrs.get("trial_dir"),
        }
        (self.experiment_dir / "best_params.json").write_text(
            json.dumps(best_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_run_state(self, status: str) -> None:
        (self.experiment_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "stage": "federated_hyperparameter_optimization",
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
