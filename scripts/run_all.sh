#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "=== Stage 1: Dataset Split ==="
python main.py dataset_split --preset default

echo "=== Stage 2: Local Training ==="
python main.py local_training --preset bank_1
python main.py local_training --preset bank_2
python main.py local_training --preset bank_3

echo "=== Stage 3: Federated Training ==="
python main.py federated_training --preset default
python main.py federated_training --preset banks_1_2
python main.py federated_training --preset banks_1_3
python main.py federated_training --preset banks_2_3

echo "=== Stage 4: Evaluation ==="
python main.py evaluation --preset default

echo "=== All stages completed ==="
