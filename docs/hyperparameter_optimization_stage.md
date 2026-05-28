# Hyperparameter Optimization Stage

## Purpose

The `hyperparameter_optimization` stage runs an Optuna study for local institution training.
It trains one local model per trial, evaluates on a stratified validation split, and optimizes a
configured validation metric.

The bank-specific configurations optimize PR-AUC using each bank's training dataset and the shared
TabNet model configuration used by `local_training`.

The bank configurations persist Optuna studies to SQLite at:

```text
data/experiments/hpo_local_bank_1_tabnet/optuna.db
data/experiments/hpo_local_bank_2_tabnet/optuna.db
data/experiments/hpo_local_bank_3_tabnet/optuna.db
```

## Run

```bash
python main.py --config configs/hyperparameter_optimization/bank_1.toml
python main.py --config configs/hyperparameter_optimization/bank_2.toml
python main.py --config configs/hyperparameter_optimization/bank_3.toml
```

You may keep the stage name as a CLI guard if desired:

```bash
python main.py hyperparameter_optimization --config configs/hyperparameter_optimization/bank_1.toml
python main.py hyperparameter_optimization --config configs/hyperparameter_optimization/bank_2.toml
python main.py hyperparameter_optimization --config configs/hyperparameter_optimization/bank_3.toml
```

Helper script:

```bash
./scripts/2_hpo/run_hyperparameter_optimization_bank_1.sh
./scripts/2_hpo/run_hyperparameter_optimization_bank_2.sh
./scripts/2_hpo/run_hyperparameter_optimization_bank_3.sh
```

Open the Optuna dashboard with:

```bash
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_1_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_2_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_3_tabnet/optuna.db
```

## Configuration

The stage config includes fixed defaults for local training plus an optional `search_space` section.
Any omitted search-space field uses the fixed value from the root config.

Optuna storage fields:

- `study_name`: study identifier used inside the storage backend
- `storage_url`: Optuna storage URL, such as `sqlite:///data/experiments/hpo_local_bank_1_tabnet/optuna.db`
- `load_if_exists`: reuse an existing study with the same name when the database already exists

Training search-space fields:

- `learning_rate`
- `fraud_weight`
- `local_epochs`
- `batch_size`
- `classification_threshold`

TabNet search-space fields:

- `decision_dim`
- `attention_dim`
- `steps`
- `relaxation_factor`
- `sparsity_weight`

Supported objectives are:

- `val_pr_auc`
- `val_roc_auc`
- `val_f1`
- `val_loss`

Metric objectives must use `direction = "maximize"`. `val_loss` must use `direction = "minimize"`.

## Outputs

The stage writes artifacts under `data/experiments/<experiment_name>/run_###/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `run_state.json`
- `optuna_trials.csv`
- `best_params.json`

`metrics.jsonl` includes per-epoch validation metrics for every trial. `best_params.json` contains
the best Optuna parameters and the final validation metrics for the best trial.
