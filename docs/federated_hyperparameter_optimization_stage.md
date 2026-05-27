# Federated Hyperparameter Optimization Stage

## Purpose

The `federated_hyperparameter_optimization` stage runs an Optuna study around the federated
training stage. Each trial samples federated parameters and federated-only local training overrides,
then executes a complete federated training run.

The main tunable parameters are:

- `proximal_mu`
- `num_rounds`
- federated client-local `learning_rate`
- federated client-local `local_epochs`
- federated client-local `fraud_weight`
- federated client-local `batch_size`
- federated client-local `classification_threshold`
- optional TabNet architecture parameters

The local training overrides apply only to the federated trial. The referenced local training
configs remain the source of institution IDs, datasets, seeds, and fallback values. Overrides are
keyed by institution id, and Optuna samples local override parameters separately for each
institution.

## Run Inclusive Global Optimization

```bash
python main.py --config configs/federated_hyperparameter_optimization/global.toml
```

You may keep the stage name as a CLI guard:

```bash
python main.py federated_hyperparameter_optimization --config configs/federated_hyperparameter_optimization/global.toml
```

Helper script:

```bash
./scripts/2_hpo/run_federated_hyperparameter_optimization.sh
```

## Run Exclusive Optimization

Exclusive federated studies use the same Optuna stage and the same `banks_x_y` naming convention as
exclusive federated training. Each combination has its own TOML config, study name, storage database,
and output experiment directory:

```bash
python main.py --config configs/federated_hyperparameter_optimization/banks_1_2.toml
python main.py --config configs/federated_hyperparameter_optimization/banks_1_3.toml
python main.py --config configs/federated_hyperparameter_optimization/banks_2_3.toml
```

Helper scripts:

```bash
./scripts/2_hpo/run_federated_hyperparameter_optimization_banks_1_2.sh
./scripts/2_hpo/run_federated_hyperparameter_optimization_banks_1_3.sh
./scripts/2_hpo/run_federated_hyperparameter_optimization_banks_2_3.sh
```

Open the Optuna dashboard with:

```bash
optuna-dashboard sqlite:///data/experiments/hpo_federated_global_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_federated_banks_1_2_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_federated_banks_1_3_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_federated_banks_2_3_tabnet/optuna.db
```

## Configuration

The config follows the local Optuna stage structure:

- fixed defaults at the root
- `study_name`, `storage_url`, and `load_if_exists` for Optuna persistence
- `institution_configs` pointing to the existing local-training TOML files
- `local_training_overrides.<institution_id>` defaults for federated client-local values
- `search_space` sections for sampled parameters

Exclusive configs contain only the two included institutions. For example, `banks_1_2.toml` samples
and trains with `bank_1` and `bank_2`, writing results under
`data/experiments/hpo_federated_banks_1_2_tabnet/`.

Example defaults:

```toml
[local_training_overrides.bank_1]
local_epochs = 2
learning_rate = 0.001

[local_training_overrides.bank_2]
local_epochs = 3
learning_rate = 0.0005
```

Federated search-space fields:

- `proximal_mu`
- `num_rounds`

Local override search-space fields:

- `learning_rate`
- `fraud_weight`
- `local_epochs`
- `batch_size`
- `classification_threshold`

The local override search space is applied per institution. Trial parameter names include the
institution id, such as `bank_1_learning_rate` and `bank_2_learning_rate`.

TabNet search-space fields:

- `decision_dim`
- `attention_dim`
- `steps`
- `relaxation_factor`
- `sparsity_weight`

Supported objectives are:

- `val_pr_auc`
- `val_loss`

`val_pr_auc` maximizes the final round's sample-weighted aggregate PR-AUC. `val_loss` minimizes the
final round's mean validation loss.

## Outputs

The stage writes artifacts under `data/experiments/<experiment_name>/run_###/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `run_state.json`
- `optuna_trials.csv`
- `best_params.json`
- `trial_###/` subdirectories containing the federated training artifacts for each trial

Each `trial_###/` directory contains the same artifacts as `federated_training`, including
`config.json`, `metrics.jsonl`, `train.log`, plots, and `model.pt`.
