# Inclusive Federated Training Stage

## Purpose
The `inclusive_federated_training` stage simulates **N institutions** training a single global model with a clear client/server object model and Flower FedProx aggregation.

The stage follows the repository architecture:
- CLI (`main.py`) selects the stage.
- Composition root (`composition/run_inclusive_federated_training.py`) loads and validates config.
- Stage (`stages/inclusive_federated_training/stage.py`) orchestrates data loading, institution node wiring, federated rounds, evaluation, and artifact persistence.
- Core modules (`domain/*`) hold reusable model/training/evaluation logic.


## Federated Object Model
The stage now models each institution similarly to a separately deployed client:

- `InstitutionNode` (`domain/federated/fedprox_orchestrator.py`) encapsulates one bank dataset and performs local optimization.
- `FedProxOrchestrator` (`domain/federated/fedprox_orchestrator.py`) acts as the server-side coordinator.
- `flower_adapter` (`domain/federated/flower_adapter.py`) isolates Flower proxy/result adaptation used for local simulation.
- Aggregation is delegated to Flower's `FedProx` strategy via `aggregate_fit`, making federated behavior explicit and framework-aligned.

## Configuration
Use `configs/inclusive_federated.toml`:

- `experiment_name`
- `output_dir`
- `num_rounds`
- `local_epochs`
- `learning_rate`
- `proximal_mu`
- `model_type` (must match a registered model in `domain/models/model_registry.py`)
- TabNet options (used when `model_type = "tabnet"`):
  - `tabnet_decision_dim`
  - `tabnet_attention_dim`
  - `tabnet_steps`
  - `tabnet_relaxation_factor`
  - `tabnet_sparsity_weight`
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
- `metrics` (local loss, sample counts, model delta magnitudes, and per-institution evaluation)
- `learning_rate`

`train.log` now includes one line per institution for each federated round, including:
- local training loss
- global-model evaluation loss/accuracy on that institution
- number of local samples
- model update magnitude (`weight_delta_l2`, `bias_delta_abs`)
