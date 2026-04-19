#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "=== Stage 1: Harmonized Train/Test Data ==="
python main.py --config configs/pipeline/harmonized_data.toml

echo "=== Stage 2: Local Training ==="
python main.py --config configs/local_training/bank_1.toml
python main.py --config configs/local_training/bank_2.toml
python main.py --config configs/local_training/bank_3.toml

echo "=== Stage 3: Federated Training ==="
python main.py --config configs/federated/global.toml
python main.py --config configs/federated/banks_1_2.toml
python main.py --config configs/federated/banks_1_3.toml
python main.py --config configs/federated/banks_2_3.toml

echo "=== Stage 4: Evaluation -- Local Models ==="
python main.py --config configs/evaluation/local/bank_1.toml
python main.py --config configs/evaluation/local/bank_2.toml
python main.py --config configs/evaluation/local/bank_3.toml

echo "=== Stage 5: Evaluation -- Inclusive Federated ==="
python main.py --config configs/evaluation/federated/inclusive/bank_1.toml
python main.py --config configs/evaluation/federated/inclusive/bank_2.toml
python main.py --config configs/evaluation/federated/inclusive/bank_3.toml

echo "=== Stage 6: Evaluation -- Exclusive Federated ==="
python main.py --config configs/evaluation/federated/exclusive/bank_1.toml
python main.py --config configs/evaluation/federated/exclusive/bank_2.toml
python main.py --config configs/evaluation/federated/exclusive/bank_3.toml

echo "=== Stage 7: Ensemble -- Lk + Fexcl ==="
python main.py --config configs/ensemble/exclusive/bank_1.toml
python main.py --config configs/ensemble/exclusive/bank_2.toml
python main.py --config configs/ensemble/exclusive/bank_3.toml

echo "=== Stage 8: Ensemble -- Lk + Fincl ==="
python main.py --config configs/ensemble/inclusive/bank_1.toml
python main.py --config configs/ensemble/inclusive/bank_2.toml
python main.py --config configs/ensemble/inclusive/bank_3.toml

echo "=== Stage 9: Comparison Report ==="
python scripts/comparison_report.py

echo "=== All stages completed ==="
