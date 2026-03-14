# FederatedLearners

## Run federated training

```bash
python main.py federated_training --preset default
```

Helper scripts:

```bash
./scripts/run_federated_training.sh
./scripts/run_federated_training_banks_1_2.sh
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
```

Helper script:

```bash
./scripts/run_local_training.sh
```

The local stage reuses the same model configuration schema as federated training and can consume a federated-style TOML config while selecting one institution via `local_institution_id`.

See stage documentation:

- `docs/local_training_stage.md`

## Run inference

```bash
python main.py inference --preset default
```

Helper script:

```bash
./scripts/run_inference.sh
```

The inference stage reads input rows from a CSV file and can optionally use a label column (for example `Class`) to report inference quality metrics.

See stage documentation:

- `docs/inference_stage.md`

## Model options

The repository now includes an alternative TabNet-based model implementation in:

- `domain/models/tabnet_model.py`

This can be used when you want a nonlinear architecture instead of the baseline logistic model in `domain/models/basic_model.py`.
