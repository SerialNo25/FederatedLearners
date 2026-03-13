#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

CONFIG_PATH="${1:-configs/inference.toml}"
python main.py inference --config "$CONFIG_PATH"
