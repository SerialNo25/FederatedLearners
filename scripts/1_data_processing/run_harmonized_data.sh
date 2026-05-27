#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/repo_root.sh"
REPO_ROOT="$(repo_root_from "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

python main.py --config configs/pipeline/harmonized_data.toml
