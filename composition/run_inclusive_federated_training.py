"""Composition root for inclusive federated training stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from stages.inclusive_federated_training.config import InclusiveFederatedTrainingConfig
from stages.inclusive_federated_training.stage import InclusiveFederatedTrainingStage


def run_inclusive_federated_training(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = InclusiveFederatedTrainingConfig.from_dict(config_dict)
    stage = InclusiveFederatedTrainingStage(config)
    return stage.execute()
