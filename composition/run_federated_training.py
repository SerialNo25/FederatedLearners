"""Composition root for federated training stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from stages.federated_training.config import FederatedTrainingConfig
from stages.federated_training.stage import FederatedTrainingStage


def run_federated_training(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = FederatedTrainingConfig.from_dict(config_dict)
    stage = FederatedTrainingStage(config)
    return stage.execute()
