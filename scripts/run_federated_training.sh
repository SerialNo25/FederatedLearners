#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

CONFIG_PATH="${1:-configs/federated.toml}"

python main.py federated_training --config "$CONFIG_PATH"
