"""Composition root for federated training stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from domain.models.model_registry import MODEL_REGISTRY
from stages.federated_training.config import FederatedTrainingConfig
from stages.federated_training.stage import FederatedTrainingStage
from stages.local_training.config import LocalTrainingConfig


def run_federated_training(config_path: str | Path) -> Path:
    path = Path(config_path)
    fed_dict = tomli.loads(path.read_text(encoding="utf-8"))

    model_dict = tomli.loads(Path(fed_dict.pop("model_config")).read_text(encoding="utf-8"))

    institution_config_paths = fed_dict.pop("institution_configs")
    institutions = []
    for ic_path in institution_config_paths:
        ic_dict = tomli.loads(Path(ic_path).read_text(encoding="utf-8"))
        ic_dict.pop("model_config", None)
        ic_dict["model"] = model_dict
        institutions.append(LocalTrainingConfig.from_dict(ic_dict))

    config = FederatedTrainingConfig(
        stage=fed_dict.get("stage", "federated_training"),
        experiment_name=fed_dict.get("experiment_name", "federated_global"),
        output_dir=fed_dict.get("output_dir", "data/experiments"),
        num_rounds=fed_dict["num_rounds"],
        proximal_mu=fed_dict.get("proximal_mu", 0.0),
        local_training_overrides=fed_dict.get("local_training_overrides", {}),
        model=model_dict,
        institutions=institutions,
    )

    experiment_dir = allocate_experiment_run_dir(config.output_dir, config.experiment_name)
    experiment_logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="federated_training",
    )

    model_factory = MODEL_REGISTRY.get_factory(
        config.model.model_type,
        config.model.model_dump(mode="python"),
    )
    stage = FederatedTrainingStage(
        config=config,
        experiment_logger=experiment_logger,
        experiment_dir=experiment_dir,
        model_factory=model_factory,
    )
    return stage.execute()
