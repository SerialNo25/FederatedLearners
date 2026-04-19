# Federated Training Stage

## Purpose
The `federated_training` stage simulates **N institutions** training a single global model with a clear client/server object model and lightweight in-process FedProx orchestration.

The stage follows the repository architecture:
- CLI (`main.py`) reads the `stage` value from the TOML config and dispatches to the matching composition root.
- Composition root (`composition/run_federated_training.py`) loads and validates config.
- Stage (`stages/federated_training/stage.py`) orchestrates data loading, institution node wiring, federated rounds, evaluation, and artifact persistence.
- Core modules (`domain/*`) hold reusable model/training/evaluation logic.


## Federated Object Model
The stage now models each institution similarly to a separately deployed client:

- `InstitutionNode` (`domain/federated/fedprox_orchestrator.py`) encapsulates one bank dataset and performs local optimization.
- `FedProxOrchestrator` (`domain/federated/fedprox_orchestrator.py`) acts as the server-side coordinator.
- Aggregation is performed directly in the orchestrator using sample-weighted parameter averaging, removing framework adapter overhead while preserving FedProx local updates.

## Configuration
Federated configs live under `configs/federated/`:

- `global.toml`
- `banks_1_2.toml`
- `banks_1_3.toml`
- `banks_2_3.toml`

Each runnable config includes `stage = "federated_training"` and exposes:

- `stage`
- `experiment_name`
- `output_dir`
- `model_config`
- `num_rounds`
- `proximal_mu`
- `local_training_overrides`
- `institution_configs`

Institution configs are local-training TOML files under `configs/local_training/`.
The shared model config is `configs/shared/model.toml`, whose `model_type` must match a registered model in `domain/models/model_registry.py`.
- Device is auto-selected at runtime by priority: `cuda > mps > cpu`

`local_training_overrides` is optional and keyed by institution id. It applies only inside
federated training and lets each federated client use different local parameters from standalone
local training without duplicating bank config files:

```toml
[local_training_overrides.bank_1]
local_epochs = 3
learning_rate = 0.001
fraud_weight = 20
batch_size = 1024
classification_threshold = 0.5

[local_training_overrides.bank_2]
local_epochs = 2
learning_rate = 0.0005
```

Any omitted institution or field falls back to the referenced local-training config for that
institution. Unknown institution ids are rejected during config validation.

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
python main.py --config configs/federated/global.toml
```

Or via helper scripts:
```bash
./scripts/run_federated_training.sh
./scripts/run_federated_training_banks_1_2.sh
./scripts/run_federated_training_banks_1_3.sh
./scripts/run_federated_training_banks_2_3.sh
```

You may keep the stage name as a CLI guard if desired:
```bash
python main.py federated_training --config configs/federated/global.toml
```

## Outputs
Artifacts are written to:

```text
data/experiments/<experiment_name>/run_###/
```

Each execution gets the next numbered run folder, such as `run_001` and `run_002`, so repeated
executions of the same federated config do not overwrite or append to prior results.

Files:
- `config.json`
- `train.log`
- `metrics.jsonl`
- `run_state.json`
- `loss_plot.svg`
- `pr_auc_plot.svg`
- `model.pt`

`metrics.jsonl` records per round:
- `epoch`
- `train_loss`
- `val_loss`
- `pr_auc`
- `metrics` (local loss, sample counts, model delta magnitudes, and per-institution evaluation)
- `learning_rate`

`loss_plot.svg` is generated from the per-round `train_loss` and `val_loss` values logged in
`metrics.jsonl`.
`pr_auc_plot.svg` plots the sample-weighted aggregate PR-AUC across institutions for each round.

`train.log` now includes one line per institution for each federated round, including:
- local training loss
- global-model evaluation loss/accuracy on that institution
- number of local samples
- model update magnitude (`weight_delta_l2`, `bias_delta_abs`)
