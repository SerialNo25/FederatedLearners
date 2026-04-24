"""Composition root for dataset mixer stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from stages.dataset_mixer.config import DatasetMixerConfig
from stages.dataset_mixer.stage import DatasetMixerStage


def run_dataset_mixer(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = DatasetMixerConfig.from_dict(tomli.loads(path.read_text(encoding="utf-8")))
    return DatasetMixerStage(config).execute()
