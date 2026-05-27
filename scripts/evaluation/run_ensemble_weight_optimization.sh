#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/repo_root.sh"
REPO_ROOT="$(repo_root_from "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

CONFIG_PATH="${1:-configs/ensemble_weight_optimization/inclusive/bank_1.toml}"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" main.py ensemble_weight_optimization --config "$CONFIG_PATH"
