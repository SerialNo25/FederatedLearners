# Inference Stage

The inference stage loads a model checkpoint produced by training stages, reads inference rows from a CSV file, and executes inference through the configured model.

## Layered Architecture Alignment

Current implementation follows the repository layering rules:

- **CLI layer (`main.py`)** only parses arguments and dispatches to the stage registry.
- **Composition root (`composition/run_inference.py`)** loads TOML config, validates with `InferenceConfig`, and explicitly wires dependencies (`InferenceService`, `InferenceDataLoader`, and `CheckpointParameterLoader`).
- **Stage layer (`stages/inference/stage.py`)** orchestrates workflow only: logger creation, optional device selection, domain-service calls, and artifact persistence.
- **Domain layer (`domain/inference/inference_service.py`)** contains reusable inference logic: CSV validation/loading, checkpoint parsing, prediction execution, and optional loss/accuracy computation.

This resolves prior layering concerns where stage orchestration mixed domain concerns (CSV parsing, checkpoint decoding, and metric calculations) directly into stage code.

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

- `checkpoint_path`: path to `model.pt` checkpoint persisted by training stages (torch-saved payload containing `model_type` and `parameters`).
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
