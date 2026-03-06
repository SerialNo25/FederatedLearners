# Federated Training Stage

## Purpose
The `federated_training` stage simulates **3 institutions** training a single global model.

The stage follows the repository architecture:
- CLI (`main.py`) selects the stage.
- Composition root (`run/run_federated_training.py`) loads and validates config.
- Stage (`stages/federated_training/stage.py`) orchestrates data loading, training rounds, aggregation, evaluation, and artifact persistence.
- Core modules (`core/*`) hold reusable model/training/evaluation logic.

## Configuration
Use `configs/federated.toml`:

- `experiment_name`
- `output_dir`
- `num_rounds`
- `local_epochs`
- `learning_rate`
- `proximal_mu`
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
python main.py federated_training --config configs/federated.toml
```

Or via helper script:
```bash
./scripts/run_federated_training.sh configs/federated.toml
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
