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

echo "=== Stage 4: Evaluation -- Local Models ==="
python main.py evaluation --preset local_bank1
python main.py evaluation --preset local_bank2
python main.py evaluation --preset local_bank3

echo "=== Stage 5: Evaluation -- Inclusive Federated ==="
python main.py evaluation --preset fincl_bank1
python main.py evaluation --preset fincl_bank2
python main.py evaluation --preset fincl_bank3

echo "=== Stage 6: Evaluation -- Exclusive Federated ==="
python main.py evaluation --preset fexcl1_bank1
python main.py evaluation --preset fexcl2_bank2
python main.py evaluation --preset fexcl3_bank3

echo "=== Stage 7: Ensemble -- Lk + Fexcl ==="
python main.py ensemble --preset L1_Fexcl1
python main.py ensemble --preset L2_Fexcl2
python main.py ensemble --preset L3_Fexcl3

echo "=== Stage 8: Ensemble -- Lk + Fincl ==="
python main.py ensemble --preset L1_Fincl
python main.py ensemble --preset L2_Fincl
python main.py ensemble --preset L3_Fincl

echo "=== Stage 9: Comparison Report ==="
python scripts/comparison_report.py

echo "=== All stages completed ==="
