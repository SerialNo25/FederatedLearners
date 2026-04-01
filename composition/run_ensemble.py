"""Composition root for ensemble evaluation stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.evaluation_service import EvaluationCheckpointLoader
from domain.logging.experiment_logger import StageExperimentLogger
from stages.ensemble.config import EnsembleConfig
from stages.ensemble.stage import EnsembleStage


def run_ensemble(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = EnsembleConfig.from_dict(config_dict)

    experiment_dir = config.output_dir / config.experiment_name
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="ensemble",
    )

    stage = EnsembleStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
        checkpoint_loader=EvaluationCheckpointLoader(),
    )
    return stage.execute()
