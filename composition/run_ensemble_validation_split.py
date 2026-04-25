"""Composition root for ensemble validation split stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from stages.ensemble_validation_split.config import EnsembleValidationSplitConfig
from stages.ensemble_validation_split.stage import EnsembleValidationSplitStage


def run_ensemble_validation_split(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = EnsembleValidationSplitConfig.from_dict(tomli.loads(path.read_text(encoding="utf-8")))
    return EnsembleValidationSplitStage(config).execute()
