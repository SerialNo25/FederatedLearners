#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

while IFS= read -r config_path; do
  echo "Running $config_path"
  python main.py --config "$config_path"
done < <(find configs/dataset_mixer -type f -name '*.toml' | sort)
