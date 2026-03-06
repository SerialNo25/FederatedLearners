"""CLI entrypoint for experiment stage execution."""

from __future__ import annotations

import argparse

from stages.registry import build_default_stage_registry


def parse_args() -> argparse.Namespace:
    registry = build_default_stage_registry()

    parser = argparse.ArgumentParser(description="Federated fraud experiment pipeline")
    parser.add_argument(
        "stage",
        choices=registry.list_stages(),
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML stage configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_default_stage_registry()
    runner = registry.get(args.stage)
    output_dir = runner(args.config)
    print(f"Stage '{args.stage}' completed: {output_dir}")


if __name__ == "__main__":
    main()
