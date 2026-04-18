"""Composition root for model evaluation stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.evaluation_service import EvaluationCheckpointLoader, ModelEvaluationService
from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from stages.evaluation.config import EvaluationConfig
from stages.evaluation.stage import EvaluationStage


def run_evaluation(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = EvaluationConfig.from_dict(config_dict)

    experiment_dir = allocate_experiment_run_dir(config.output_dir, config.experiment_name)
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="evaluation",
    )

    stage = EvaluationStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
        checkpoint_loader=EvaluationCheckpointLoader(),
        evaluation_service=ModelEvaluationService(),
    )
    return stage.execute()
