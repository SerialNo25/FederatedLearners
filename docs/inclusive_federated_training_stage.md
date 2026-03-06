# Inclusive Federated Training Stage

## Purpose
The `inclusive_federated_training` stage simulates **3 institutions** training a single global model.

The stage follows the repository architecture:
- CLI (`main.py`) selects the stage.
- Composition root (`composition/run_inclusive_federated_training.py`) loads and validates config.
- Stage (`stages/inclusive_federated_training/stage.py`) orchestrates data loading, training rounds, aggregation, evaluation, and artifact persistence.
- Core modules (`core/*`) hold reusable model/training/evaluation logic.

## Configuration
Use `configs/inclusive_federated.toml`:

- `experiment_name`
- `output_dir`
- `num_rounds`
- `local_epochs`
- `learning_rate`
- `proximal_mu`
- `model_type` (`logistic_regression` or `tabnet`)
- TabNet options (used when `model_type = "tabnet"`):
  - `tabnet_decision_dim`
  - `tabnet_attention_dim`
  - `tabnet_steps`
  - `tabnet_relaxation_factor`
  - `tabnet_sparsity_weight`
  - `tabnet_device`
- `[[institutions]]` (must be exactly 3)
  - `institution_id`
  - `dataset_path`

## Input Data Requirements
Each institution CSV must follow this exact schema and order:

```text
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
```

Validation enforces:
- exact header order
- numeric feature values
- binary `Class` labels (`0`/`1`)

## Execution
```bash
python main.py inclusive_federated_training --config configs/inclusive_federated.toml
```

Or via helper script:
```bash
./scripts/run_inclusive_federated_training.sh configs/inclusive_federated.toml
```

## Outputs
Artifacts are written to:

```text
data/experiments/<experiment_name>/
```

Files:
- `config.json`
- `train.log`
- `metrics.jsonl`
- `model.pt`

`metrics.jsonl` records per round:
- `epoch`
- `train_loss`
- `val_loss`
- `metrics` (local losses and per-institution evaluation)
- `learning_rate`
