#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/federated.toml}"

python main.py federated_training --config "$CONFIG_PATH"
