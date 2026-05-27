# FederatedLearners

## Script Layout

Helper scripts are grouped by workflow:

- `scripts/pipeline/` for multi-stage runs
- `scripts/data_processing/` for harmonization and dataset mixing
- `scripts/training/` for local and federated training
- `scripts/hpo/` for hyperparameter optimization
- `scripts/evaluation/` for evaluation and ensemble utilities
- `scripts/analysis/` for dashboards and reports

## Run federated training

```bash
python main.py --config configs/federated/global.toml
```

Helper scripts:

```bash
./scripts/training/run_federated_training.sh
./scripts/training/run_federated_training_banks_1_2.sh
./scripts/training/run_federated_training_banks_1_3.sh
./scripts/training/run_federated_training_banks_2_3.sh
```

The runnable config declares its stage:

```toml
stage = "federated_training"
```

See stage documentation:

- `docs/harmonized_data_stage.md`
- `docs/federated_training_stage.md`

## Build harmonized datasets

```bash
python main.py --config configs/pipeline/harmonized_data.toml
```

Run this if you start from the raw bank CSV files in `data/raw/`. The stage now splits raw rows first,
fits preprocessing statistics on each train subset only, and writes the harmonized train/test CSVs used
by the training and evaluation configs.

Helper script:

```bash
./scripts/data_processing/run_harmonized_data.sh
```

## Run local-only training

```bash
python main.py --config configs/local_training/default.toml
python main.py --config configs/local_training/bank_1.toml
python main.py --config configs/local_training/bank_2.toml
python main.py --config configs/local_training/bank_3.toml
```

Helper scripts:

```bash
./scripts/training/run_local_training.sh
./scripts/training/run_local_training_bank_1.sh
./scripts/training/run_local_training_bank_2.sh
./scripts/training/run_local_training_bank_3.sh
```

The local stage reuses the same model configuration schema as federated training and can consume a federated-style TOML config while selecting one institution via `local_institution_id`.

See stage documentation:

- `docs/local_training_stage.md`

## Run hyperparameter optimization

```bash
python main.py --config configs/hyperparameter_optimization/bank_1.toml
python main.py --config configs/hyperparameter_optimization/bank_2.toml
python main.py --config configs/hyperparameter_optimization/bank_3.toml
```

Helper scripts:

```bash
./scripts/hpo/run_hyperparameter_optimization_bank_1.sh
./scripts/hpo/run_hyperparameter_optimization_bank_2.sh
./scripts/hpo/run_hyperparameter_optimization_bank_3.sh
```

The bank Optuna studies write to:

```text
data/experiments/hpo_local_bank_1_tabnet/optuna.db
data/experiments/hpo_local_bank_2_tabnet/optuna.db
data/experiments/hpo_local_bank_3_tabnet/optuna.db
```

Open the Optuna dashboard with:

```bash
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_1_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_2_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_3_tabnet/optuna.db
```

See stage documentation:

- `docs/hyperparameter_optimization_stage.md`

## Run federated hyperparameter optimization

```bash
python main.py --config configs/federated_hyperparameter_optimization/global.toml
python main.py --config configs/federated_hyperparameter_optimization/banks_1_2.toml
python main.py --config configs/federated_hyperparameter_optimization/banks_1_3.toml
python main.py --config configs/federated_hyperparameter_optimization/banks_2_3.toml
```

Helper scripts:

```bash
./scripts/hpo/run_federated_hyperparameter_optimization.sh
./scripts/hpo/run_federated_hyperparameter_optimization_banks_1_2.sh
./scripts/hpo/run_federated_hyperparameter_optimization_banks_1_3.sh
./scripts/hpo/run_federated_hyperparameter_optimization_banks_2_3.sh
```

The exclusive `banks_x_y` studies write separate Optuna databases and experiment results:

```text
data/experiments/hpo_federated_banks_1_2_tabnet/optuna.db
data/experiments/hpo_federated_banks_1_3_tabnet/optuna.db
data/experiments/hpo_federated_banks_2_3_tabnet/optuna.db
```

Open the Optuna dashboard with:

```bash
optuna-dashboard sqlite:///data/experiments/hpo_federated_global_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_federated_banks_1_2_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_federated_banks_1_3_tabnet/optuna.db
optuna-dashboard sqlite:///data/experiments/hpo_federated_banks_2_3_tabnet/optuna.db
```

See stage documentation:

- `docs/federated_hyperparameter_optimization_stage.md`

## Run evaluation

```bash
python main.py --config configs/evaluation/default.toml
```

Helper script:

```bash
./scripts/evaluation/run_evaluation.sh
```

The evaluation stage loads a persisted `model.pt` file and a dataset CSV, then reports `loss`, `accuracy`, `precision`, `recall`, `f1`, and `pr_auc`.

See stage documentation:

- `docs/evaluation_stage.md`

## Model options

The repository now includes an alternative TabNet-based model implementation in:

- `domain/models/tabnet_model.py`

This can be used when you want a nonlinear architecture instead of the baseline logistic model in `domain/models/basic_model.py`.
