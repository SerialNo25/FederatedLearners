"""Composition root for inference stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.inference.inference_service import (
    CheckpointParameterLoader,
    InferenceDataLoader,
    InferenceService,
)
from domain.logging.experiment_logger import StageExperimentLogger
from stages.inference.config import InferenceConfig
from stages.inference.stage import InferenceStage


def run_inference(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = InferenceConfig.from_dict(config_dict)

    experiment_dir = config.output_dir / config.experiment_name
    logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="inference",
    )

    stage = InferenceStage(
        config=config,
        experiment_logger=logger,
        experiment_dir=experiment_dir,
        inference_service=InferenceService(),
        data_loader=InferenceDataLoader(),
        checkpoint_loader=CheckpointParameterLoader(),
    )
    return stage.execute()
