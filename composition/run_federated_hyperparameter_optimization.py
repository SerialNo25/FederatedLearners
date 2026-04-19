"""Composition root for Optuna-backed federated hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from stages.federated_hyperparameter_optimization.config import (
    FederatedHyperparameterOptimizationConfig,
)
from stages.federated_hyperparameter_optimization.stage import (
    FederatedHyperparameterOptimizationStage,
)
from stages.local_training.config import LocalTrainingConfig


def run_federated_hyperparameter_optimization(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))

    model_dict = tomli.loads(Path(config_dict.pop("model_config")).read_text(encoding="utf-8"))

    institution_config_paths = config_dict.pop("institution_configs")
    institutions = []
    for ic_path in institution_config_paths:
        ic_dict = tomli.loads(Path(ic_path).read_text(encoding="utf-8"))
        ic_dict.pop("model_config", None)
        ic_dict["model"] = model_dict
        institutions.append(LocalTrainingConfig.from_dict(ic_dict))

    config_dict["model"] = model_dict
    config_dict["institutions"] = institutions
    config = FederatedHyperparameterOptimizationConfig.from_dict(config_dict)

    experiment_dir = allocate_experiment_run_dir(config.output_dir, config.experiment_name)
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="federated_hyperparameter_optimization",
    )

    stage = FederatedHyperparameterOptimizationStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
    )
    return stage.execute()
