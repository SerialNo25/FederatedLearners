"""Composition root for federated training stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.logging.experiment_logger import StageExperimentLogger
from domain.models.device_selector import DeviceSelector
from domain.models.model_registry import MODEL_REGISTRY
from stages.federated_training.config import FederatedTrainingConfig
from stages.federated_training.stage import FederatedTrainingStage


def run_federated_training(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = FederatedTrainingConfig.from_dict(config_dict)

    experiment_dir = config.output_dir / config.experiment_name
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="federated_training",
    )

    model_config = config.to_dict()
    if config.model_type == "tabnet":
        selector = DeviceSelector()
        selected_device = selector.select_best_device()
        model_config["tabnet_device"] = selected_device
        experiment_logger.info(
            "tabnet_device_selection selected=%s available=%s"
            % (selected_device, ",".join(selector.available_devices()))
        )

    model_factory = MODEL_REGISTRY.get_factory(config.model_type, model_config)
    stage = FederatedTrainingStage(
        config=config,
        experiment_logger=experiment_logger,
        model_factory=model_factory,
    )
    return stage.execute()
