"""Composition root for dataset split stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from stages.dataset_split.config import DatasetSplitConfig
from stages.dataset_split.stage import DatasetSplitStage


def run_dataset_split(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = DatasetSplitConfig.from_dict(tomli.loads(path.read_text(encoding="utf-8")))
    return DatasetSplitStage(config).execute()
