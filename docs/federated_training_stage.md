# Federated Training Stage

## Purpose
The `federated_training` stage simulates **N institutions** training a single global model with a clear client/server object model and lightweight in-process FedProx orchestration.

The stage follows the repository architecture:
- CLI (`main.py`) selects the stage.
- Composition root (`composition/run_federated_training.py`) loads and validates config.
- Stage (`stages/federated_training/stage.py`) orchestrates data loading, institution node wiring, federated rounds, evaluation, and artifact persistence.
- Core modules (`domain/*`) hold reusable model/training/evaluation logic.


## Federated Object Model
The stage now models each institution similarly to a separately deployed client:

- `InstitutionNode` (`domain/federated/fedprox_orchestrator.py`) encapsulates one bank dataset and performs local optimization.
- `FedProxOrchestrator` (`domain/federated/fedprox_orchestrator.py`) acts as the server-side coordinator.
- Aggregation is performed directly in the orchestrator using sample-weighted parameter averaging, removing framework adapter overhead while preserving FedProx local updates.

## Configuration
The stage supports named presets registered in `stages/registry.py`:

- `default` -> `configs/federated.toml`
- `banks_1_2` -> `configs/federated_banks_1_2.toml`

Both configs expose the same fields:

- `experiment_name`
- `output_dir`
- `num_rounds`
- `local_epochs`
- `learning_rate`
- `proximal_mu`
- `[model]` (discriminated model configuration)
  - `model_type` (must match a registered model in `domain/models/model_registry.py`)
  - TabNet options (used when `model_type = "tabnet"`):
    - `decision_dim`
    - `attention_dim`
    - `steps`
    - `relaxation_factor`
    - `sparsity_weight`
- Device is auto-selected at runtime by priority: `cuda > mps > cpu`
- `num_institutions` (must be >= 1 and match the number of `[[institutions]]` entries)
- `[[institutions]]`
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
python main.py federated_training --preset default
```

Or via helper scripts:
```bash
./scripts/run_federated_training.sh
./scripts/run_federated_training_banks_1_2.sh
```

You can optionally bypass presets with an explicit config path:
```bash
python main.py federated_training --config configs/federated.toml
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
- `metrics` (local loss, sample counts, model delta magnitudes, and per-institution evaluation)
- `learning_rate`

`train.log` now includes one line per institution for each federated round, including:
- local training loss
- global-model evaluation loss/accuracy on that institution
- number of local samples
- model update magnitude (`weight_delta_l2`, `bias_delta_abs`)
