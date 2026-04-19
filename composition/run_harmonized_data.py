"""Composition root for raw dataset harmonization stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from domain.harmonization.raw_data_harmonizer import RawDataHarmonizationService
from stages.harmonized_data.config import HarmonizedDataConfig
from stages.harmonized_data.stage import HarmonizedDataStage


def run_harmonized_data(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = HarmonizedDataConfig.from_dict(tomli.loads(path.read_text(encoding="utf-8")))
    stage = HarmonizedDataStage(
        config=config,
        harmonizer=RawDataHarmonizationService(
            seed=config.seed,
            sparkov_target_size=config.sparkov_target_size,
        ),
    )
    return stage.execute()
