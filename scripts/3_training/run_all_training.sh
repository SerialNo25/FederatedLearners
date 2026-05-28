#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/repo_root.sh"
REPO_ROOT="$(repo_root_from "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

echo "=== Stage 1: Local Training ==="
python main.py --config configs/local_training/bank_1.toml
python main.py --config configs/local_training/bank_2.toml
python main.py --config configs/local_training/bank_3.toml

echo "=== Stage 2: Federated Training ==="
python main.py --config configs/federated/global.toml
python main.py --config configs/federated/banks_1_2.toml
python main.py --config configs/federated/banks_1_3.toml
python main.py --config configs/federated/banks_2_3.toml
