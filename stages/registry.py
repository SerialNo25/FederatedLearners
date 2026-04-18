"""Dynamic stage runner resolution for CLI dispatch."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Protocol


class StageRunner(Protocol):
    def __call__(self, config_path: str | Path) -> Path: ...


_STAGE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def resolve_stage_runner(stage_name: str) -> StageRunner:
    """Resolve a composition-root runner by stage naming convention."""
    if not _STAGE_NAME_PATTERN.fullmatch(stage_name):
        raise ValueError(f"Invalid stage name '{stage_name}'")

    module_name = f"composition.run_{stage_name}"
    function_name = f"run_{stage_name}"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise KeyError(f"Unknown stage '{stage_name}'") from exc
        raise

    runner = getattr(module, function_name, None)
    if runner is None:
        raise KeyError(f"Stage '{stage_name}' does not expose {function_name}()")
    return runner
