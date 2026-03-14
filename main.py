"""CLI entrypoint for experiment stage execution."""

from __future__ import annotations

import argparse

from stages.registry import build_default_stage_registry, StageRegistry


def parse_args(stageRegistry: StageRegistry) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated fraud experiment pipeline")
    parser.add_argument(
        "stage",
        choices=stageRegistry.list_stages(),
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML stage configuration file",
    )
    parser.add_argument(
        "--preset",
        help="Named stage configuration preset",
    )
    return parser.parse_args()


def main() -> None:
    registry = build_default_stage_registry()
    args = parse_args(registry)
    runner = registry.get(args.stage)
    config_path = registry.resolve_config_path(
        args.stage,
        config_path=args.config,
        preset_name=args.preset,
    )
    output_dir = runner(config_path)
    print(f"Stage '{args.stage}' completed: {output_dir}")


if __name__ == "__main__":
    main()
