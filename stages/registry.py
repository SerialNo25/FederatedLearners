"""Stage registry for CLI stage dispatch."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from composition.run_inference import run_inference
from composition.run_federated_training import run_federated_training

StageRunner = Callable[[str | Path], Path]


class StageRegistry:
    """Registers available experiment stages and their composition roots."""

    def __init__(self) -> None:
        self._runners: dict[str, StageRunner] = {}

    def register(self, stage_name: str, runner: StageRunner) -> None:
        self._runners[stage_name] = runner

    def get(self, stage_name: str) -> StageRunner:
        if stage_name not in self._runners:
            available = ", ".join(sorted(self._runners))
            raise KeyError(f"Unknown stage '{stage_name}'. Available stages: {available}")
        return self._runners[stage_name]

    def list_stages(self) -> list[str]:
        return sorted(self._runners)


def build_default_stage_registry() -> StageRegistry:
    registry = StageRegistry()
    registry.register("inference", run_inference)
    registry.register("federated_training", run_federated_training)
    return registry
