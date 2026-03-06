# FederatedLearners

## Run inclusive federated training

```bash
python main.py inclusive_federated_training --config configs/inclusive_federated.toml
```

Helper script:

```bash
./scripts/run_inclusive_federated_training.sh configs/inclusive_federated.toml
```

See stage documentation:

- `docs/inclusive_federated_training_stage.md`

## Model options

The repository now includes an alternative TabNet-based model implementation in:

- `domain/models/tabnet_model.py`

This can be used when you want a nonlinear architecture instead of the baseline logistic model in `domain/models/basic_model.py`.
