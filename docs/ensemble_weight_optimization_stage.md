# Ensemble Weight Optimization Stage

The ensemble weight optimization stage runs Optuna over a single parameter:
`ensemble_weight`.

Default execution:

```bash
scripts/run_ensemble_weight_optimization.sh
```

Custom config execution:

```bash
scripts/run_ensemble_weight_optimization.sh configs/ensemble_weight_optimization/exclusive/bank_2.toml
```

The config uses the same checkpoint reference pattern as `model_matrix_evaluation`:

```text
{base_path}/run_{run_number:03d}/model.pt
```

Each config targets one ensemble only:

- one local checkpoint run
- one federated checkpoint run
- one bank-local dataset

The stage loads both checkpoints once, precomputes their probabilities on the
configured dataset, and lets Optuna search:

```text
ensemble = w * local + (1 - w) * federated
```

Default configs are provided for:

- exclusive bank 1, 2, 3
- inclusive bank 1, 2, 3

Outputs are written under `data/experiments/<experiment_name>/run_NNN/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `optuna_trials.csv`
- `best_params.json`
- `run_state.json`
