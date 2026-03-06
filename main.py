"""CLI entrypoint for experiment stage execution."""

from __future__ import annotations

import argparse

from run.run_federated_training import run_federated_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated fraud experiment pipeline")
    parser.add_argument("stage", choices=["federated_training"])
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "federated_training":
        output_dir = run_federated_training(args.config)
        print(f"Federated training completed: {output_dir}")


if __name__ == "__main__":
    main()
