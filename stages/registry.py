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
        self._presets: dict[str, dict[str, Path]] = {}

    def register(self, stage_name: str, runner: StageRunner) -> None:
        self._runners[stage_name] = runner

    def register_preset(self, stage_name: str, preset_name: str, config_path: str | Path) -> None:
        if stage_name not in self._runners:
            raise KeyError(f"Cannot register preset for unknown stage '{stage_name}'")
        stage_presets = self._presets.setdefault(stage_name, {})
        stage_presets[preset_name] = Path(config_path)

    def resolve_config_path(
        self,
        stage_name: str,
        *,
        config_path: str | Path | None,
        preset_name: str | None,
    ) -> Path:
        if config_path is not None and preset_name is not None:
            raise ValueError("Use either --config or --preset, not both")

        if config_path is not None:
            return Path(config_path)

        if preset_name is None:
            available = ", ".join(self.list_presets(stage_name)) or "none"
            raise ValueError(
                f"Either --config or --preset is required for stage '{stage_name}'. "
                f"Available presets: {available}"
            )

        presets_for_stage = self._presets.get(stage_name, {})
        if preset_name not in presets_for_stage:
            available = ", ".join(sorted(presets_for_stage)) or "none"
            raise ValueError(
                f"Unknown preset '{preset_name}' for stage '{stage_name}'. "
                f"Available presets: {available}"
            )

        return presets_for_stage[preset_name]

    def list_presets(self, stage_name: str) -> list[str]:
        return sorted(self._presets.get(stage_name, {}))

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

    registry.register_preset("inference", "default", "configs/inference.toml")
    registry.register_preset("federated_training", "default", "configs/federated.toml")
    registry.register_preset("federated_training", "banks_1_2", "configs/federated_banks_1_2.toml")
    return registry
