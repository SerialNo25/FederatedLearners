# FederatedLearners

## Run federated training

```bash
python main.py --config configs/federated/global.toml
```

Helper scripts:

```bash
./scripts/run_federated_training.sh
./scripts/run_federated_training_banks_1_2.sh
./scripts/run_federated_training_banks_1_3.sh
./scripts/run_federated_training_banks_2_3.sh
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

## Run local-only training

```bash
python main.py --config configs/local_training/default.toml
python main.py --config configs/local_training/bank_1.toml
python main.py --config configs/local_training/bank_2.toml
python main.py --config configs/local_training/bank_3.toml
```

Helper scripts:

```bash
./scripts/run_local_training.sh
./scripts/run_local_training_bank_1.sh
./scripts/run_local_training_bank_2.sh
./scripts/run_local_training_bank_3.sh
```

The local stage reuses the same model configuration schema as federated training and can consume a federated-style TOML config while selecting one institution via `local_institution_id`.

See stage documentation:

- `docs/local_training_stage.md`

## Run hyperparameter optimization

```bash
python main.py --config configs/hyperparameter_optimization/bank_1.toml
```

Helper script:

```bash
./scripts/run_hyperparameter_optimization_bank_1.sh
```

The bank 1 Optuna study writes to:

```text
data/experiments/hpo_local_bank_1_tabnet/optuna.db
```

Open the Optuna dashboard with:

```bash
optuna-dashboard sqlite:///data/experiments/hpo_local_bank_1_tabnet/optuna.db
```

See stage documentation:

- `docs/hyperparameter_optimization_stage.md`

## Run evaluation

```bash
python main.py --config configs/evaluation/default.toml
```

Helper script:

```bash
./scripts/run_evaluation.sh
```

The evaluation stage loads a persisted `model.pt` file and a dataset CSV, then reports `loss`, `accuracy`, `precision`, `recall`, `f1`, and `pr_auc`.

See stage documentation:

- `docs/evaluation_stage.md`

## Model options

The repository now includes an alternative TabNet-based model implementation in:

- `domain/models/tabnet_model.py`

This can be used when you want a nonlinear architecture instead of the baseline logistic model in `domain/models/basic_model.py`.
