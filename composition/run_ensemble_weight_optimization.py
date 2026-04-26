"""Composition root for Optuna-backed ensemble-weight optimization."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.evaluation_service import EvaluationCheckpointLoader
from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from stages.ensemble_weight_optimization.config import EnsembleWeightOptimizationConfig
from stages.ensemble_weight_optimization.stage import EnsembleWeightOptimizationStage


def run_ensemble_weight_optimization(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = EnsembleWeightOptimizationConfig.from_dict(config_dict)

    experiment_dir = allocate_experiment_run_dir(config.output_dir, config.experiment_name)
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="ensemble_weight_optimization",
    )

    stage = EnsembleWeightOptimizationStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
        checkpoint_loader=EvaluationCheckpointLoader(),
    )
    return stage.execute()
