"""CLI entrypoint for experiment stage execution."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import tomli
from pydantic import BaseModel, ConfigDict, field_validator

from stages.registry import resolve_stage_runner

_STAGE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class StageInvocationConfig(BaseModel):
    """Validated CLI routing fields from a stage config file."""

    model_config = ConfigDict(frozen=True)

    stage: str

    @field_validator("stage")
    @classmethod
    def _validate_stage_name(cls, value: str) -> str:
        if not _STAGE_NAME_PATTERN.fullmatch(value):
            raise ValueError("stage must be a lowercase composition-root stage name")
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated fraud experiment pipeline")
    parser.add_argument(
        "stage",
        nargs="?",
        help="Optional stage name. When provided, it must match the config's stage field.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML stage configuration file",
    )
    return parser.parse_args()


def load_stage_invocation(config_path: Path) -> StageInvocationConfig:
    return StageInvocationConfig.model_validate(
        tomli.loads(config_path.read_text(encoding="utf-8"))
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    invocation = load_stage_invocation(config_path)
    if args.stage is not None and args.stage != invocation.stage:
        raise ValueError(
            f"CLI stage '{args.stage}' does not match config stage '{invocation.stage}'"
        )

    runner = resolve_stage_runner(invocation.stage)
    output_dir = runner(config_path)
    print(f"Stage '{invocation.stage}' completed: {output_dir}")


if __name__ == "__main__":
    main()
