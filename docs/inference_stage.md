# Inference Stage

The inference stage loads a model checkpoint produced by training stages, reads inference rows from a CSV file, and executes inference through the configured model.

## Run

```bash
python main.py inference --config configs/inference.toml
```

or:

```bash
./scripts/run_inference.sh configs/inference.toml
```

## Config

`configs/inference.toml` includes:

- `checkpoint_path`: path to `model.pt` JSON parameters.
- `model_type`: model registered in `domain/models/model_registry.py` (for example `tabnet`).
- `input_data_path`: CSV file containing feature rows to score.
- `feature_columns`: ordered list of feature columns to read from the CSV.
- `label_column` (optional): if provided (for example `Class`), the stage computes quality metrics (`loss` and `accuracy`) in addition to predictions.
- TabNet architecture fields (`tabnet_*`) for rebuilding the model before loading checkpoint parameters.

## Outputs

The stage writes outputs under `data/experiments/<experiment_name>/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `predictions.json` (includes predictions plus optional evaluation metrics)
