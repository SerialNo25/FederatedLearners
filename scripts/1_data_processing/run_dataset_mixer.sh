#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/repo_root.sh"
REPO_ROOT="$(repo_root_from "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

CONFIG_PATH="${1:-configs/dataset_mixer/80_10_10/train/bank_1_dominant.toml}"

"$PYTHON_BIN" main.py --config "$CONFIG_PATH"
