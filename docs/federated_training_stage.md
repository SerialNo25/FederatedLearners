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
- `banks_1_3` -> `configs/federated_banks_1_3.toml`
- `banks_2_3` -> `configs/federated_banks_2_3.toml`

Both configs expose the same fields:

- `experiment_name`
- `output_dir`
- `num_rounds`
- `local_epochs`
- `learning_rate`
- `proximal_mu`
- `validation_fraction`
- `seed`
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
amount,log_amount,hour_of_day,day_of_week,is_fraud
```

Validation enforces:
- exact header order
- numeric feature values
- binary `is_fraud` labels (`0`/`1`)

## Validation Semantics

Each institution dataset is split into train and validation partitions before federated training
starts. Local updates are fit on the train split, while the reported `val_*` metrics are computed on
the held-out validation split. The split is controlled by `validation_fraction` and `seed`, making
the round-level metrics deterministic and meaningfully comparable across runs.

## Execution
```bash
python main.py federated_training --preset default
```

Or via helper scripts:
```bash
./scripts/run_federated_training.sh
./scripts/run_federated_training_banks_1_2.sh
./scripts/run_federated_training_banks_1_3.sh
./scripts/run_federated_training_banks_2_3.sh
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
- `training_state.pt`
- `loss_plot.svg`
- `pr_auc.svg`
- `f1_optimal_threshold.svg`
- `threshold_curves.svg`
- `per_client_performance_boxplots.svg`
- `global_vs_local_convergence.svg`

`metrics.jsonl` records per round:
- `epoch`
- `train_loss`
- `val_loss`
- `metrics` (local loss, PR-AUC, optimal-threshold F1, sample counts, model delta magnitudes, and per-institution evaluation)
- `learning_rate`

`train.log` now includes one line per institution for each federated round, including:
- local training loss
- global-model evaluation loss/accuracy on that institution
- number of local samples
- model update magnitude (`weight_delta_l2`, `bias_delta_abs`)
