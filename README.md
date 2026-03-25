# FederatedLearners

## Run federated training

```bash
python main.py federated_training --preset default
```

Helper scripts:

```bash
./scripts/run_federated_training.sh
./scripts/run_federated_training_banks_1_2.sh
./scripts/run_federated_training_banks_1_3.sh
./scripts/run_federated_training_banks_2_3.sh
```

You can still pass an explicit config path when needed:

```bash
python main.py federated_training --config configs/federated.toml
```

See stage documentation:

- `docs/federated_training_stage.md`

## Run local-only training

```bash
python main.py local_training --preset default
python main.py local_training --preset bank_1
python main.py local_training --preset bank_2
python main.py local_training --preset bank_3
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

## Run evaluation

```bash
python main.py evaluation --preset default
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
