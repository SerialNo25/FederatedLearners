"""Composition root for model matrix evaluation stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.evaluation_service import EvaluationCheckpointLoader, ModelEvaluationService
from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from stages.model_matrix_evaluation.config import ModelMatrixConfig
from stages.model_matrix_evaluation.stage import ModelMatrixEvaluationStage


def run_model_matrix_evaluation(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = ModelMatrixConfig.from_dict(config_dict)

    experiment_dir = allocate_experiment_run_dir(config.output_dir, config.experiment_name)
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="model_matrix_evaluation",
    )

    stage = ModelMatrixEvaluationStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
        checkpoint_loader=EvaluationCheckpointLoader(),
        evaluation_service=ModelEvaluationService(),
    )
    return stage.execute()
