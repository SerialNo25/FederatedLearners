"""Composition root for Optuna-backed local hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from stages.hyperparameter_optimization.config import HyperparameterOptimizationConfig
from stages.hyperparameter_optimization.stage import HyperparameterOptimizationStage


def run_hyperparameter_optimization(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))

    model_dict = tomli.loads(Path(config_dict.pop("model_config")).read_text(encoding="utf-8"))
    config_dict["model"] = model_dict

    config = HyperparameterOptimizationConfig.from_dict(config_dict)

    experiment_dir = allocate_experiment_run_dir(config.output_dir, config.experiment_name)
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="hyperparameter_optimization",
    )

    stage = HyperparameterOptimizationStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
    )
    return stage.execute()
