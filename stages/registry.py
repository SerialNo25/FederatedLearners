"""Stage registry for CLI stage dispatch."""

from __future__ import annotations

from typing import Protocol
from pathlib import Path

from composition.run_ensemble import run_ensemble
from composition.run_evaluation import run_evaluation
from composition.run_dataset_split import run_dataset_split
from composition.run_federated_training import run_federated_training
from composition.run_harmonized_data import run_harmonized_data
from composition.run_local_training import run_local_training

class StageRunner(Protocol):
    def __call__(self, config_path: str | Path) -> Path: ...


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
    registry.register("ensemble", run_ensemble)
    registry.register("evaluation", run_evaluation)
    registry.register("harmonized_data", run_harmonized_data)
    registry.register("dataset_split", run_dataset_split)
    registry.register("federated_training", run_federated_training)
    registry.register("local_training", run_local_training)

    registry.register_preset("ensemble", "L1_Fexcl1", "configs/ensemble_L1_Fexcl1.toml")
    registry.register_preset("ensemble", "L2_Fexcl2", "configs/ensemble_L2_Fexcl2.toml")
    registry.register_preset("ensemble", "L3_Fexcl3", "configs/ensemble_L3_Fexcl3.toml")
    registry.register_preset("ensemble", "L1_Fincl", "configs/ensemble_L1_Fincl.toml")
    registry.register_preset("ensemble", "L2_Fincl", "configs/ensemble_L2_Fincl.toml")
    registry.register_preset("ensemble", "L3_Fincl", "configs/ensemble_L3_Fincl.toml")

    registry.register_preset("evaluation", "default", "configs/evaluation.toml")
    registry.register_preset("evaluation", "local_bank1", "configs/eval_local_bank1.toml")
    registry.register_preset("evaluation", "local_bank2", "configs/eval_local_bank2.toml")
    registry.register_preset("evaluation", "local_bank3", "configs/eval_local_bank3.toml")
    registry.register_preset("evaluation", "fincl_bank1", "configs/eval_fincl_bank1.toml")
    registry.register_preset("evaluation", "fincl_bank2", "configs/eval_fincl_bank2.toml")
    registry.register_preset("evaluation", "fincl_bank3", "configs/eval_fincl_bank3.toml")
    registry.register_preset("evaluation", "global_bank1", "configs/eval_global_bank1.toml")
    registry.register_preset("evaluation", "global_bank2", "configs/eval_global_bank2.toml")
    registry.register_preset("evaluation", "global_bank3", "configs/eval_global_bank3.toml")
    registry.register_preset("evaluation", "fexcl1_bank1", "configs/eval_fexcl1_bank1.toml")
    registry.register_preset("evaluation", "fexcl2_bank2", "configs/eval_fexcl2_bank2.toml")
    registry.register_preset("evaluation", "fexcl3_bank3", "configs/eval_fexcl3_bank3.toml")
    registry.register_preset("harmonized_data", "default", "configs/harmonized_data.toml")
    registry.register_preset("dataset_split", "default", "configs/dataset_split.toml")
    registry.register_preset("federated_training", "default", "configs/federated.toml")
    registry.register_preset("local_training", "default", "configs/local_training.toml")
    registry.register_preset("local_training", "bank_1", "configs/local_training_bank_1.toml")
    registry.register_preset("local_training", "bank_2", "configs/local_training_bank_2.toml")
    registry.register_preset("local_training", "bank_3", "configs/local_training_bank_3.toml")
    registry.register_preset("local_training", "global", "configs/local_training_global.toml")
    registry.register_preset("federated_training", "banks_1_2", "configs/federated_banks_1_2.toml")
    registry.register_preset("federated_training", "banks_1_3", "configs/federated_banks_1_3.toml")
    registry.register_preset("federated_training", "banks_2_3", "configs/federated_banks_2_3.toml")
    return registry
