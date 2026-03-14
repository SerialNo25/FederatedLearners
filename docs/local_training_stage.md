# Local Training Stage

## Purpose

The `local_training` stage trains a model for a single institution without federated aggregation.
It reuses the same model registry and model configuration schema as the federated stage, so the
`[model]` block can be shared directly between `federated_training` and `local_training`.

## Run

```bash
python main.py local_training --preset default
```

or

```bash
python main.py local_training --config configs/local_training.toml
```

## Configuration compatibility

`local_training` intentionally accepts federated-style configuration files. It ignores federated-only
keys (for example `num_rounds` and `proximal_mu`) and trains only one institution selected by
`local_institution_id`.

If `local_institution_id` is omitted, the first entry in `[[institutions]]` is used.

## Outputs

The stage writes artifacts under `data/experiments/<experiment_name>/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `model.pt`
