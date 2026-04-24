#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

CONFIG_PATH="${1:-configs/dataset_mixer/bank_1_dominant.toml}"

python main.py --config "$CONFIG_PATH"
