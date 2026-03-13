"""Composition root for inference stage."""

from __future__ import annotations

from pathlib import Path

import tomli

from stages.inference.config import InferenceConfig
from stages.inference.stage import InferenceStage


def run_inference(config_path: str | Path) -> Path:
    path = Path(config_path)
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    config = InferenceConfig.from_dict(config_dict)
    stage = InferenceStage(config)
    return stage.execute()
