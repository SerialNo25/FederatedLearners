# Evaluation Stage

The evaluation stage loads a persisted `model.pt` checkpoint, rebuilds the registered model, evaluates it against a dataset CSV, and writes experiment artifacts with the standard binary-classification metrics.

## Layered Architecture Alignment

- **CLI layer (`main.py`)** only parses arguments and routes execution from the config `stage` value.
- **Composition root (`composition/run_evaluation.py`)** loads TOML, validates it with `EvaluationConfig`, and explicitly wires checkpoint-loading and evaluation services.
- **Stage layer (`stages/evaluation/stage.py`)** orchestrates dataset loading, model evaluation, logging, and artifact persistence.
- **Domain layer (`domain/evaluation_service.py` and `domain/metrics/evaluation.py`)** owns checkpoint parsing, model reconstruction, and metric computation.

## Run

```bash
python main.py --config configs/evaluation/default.toml
```

or:

```bash
./scripts/evaluation/run_evaluation.sh
```

## Config

`configs/evaluation/default.toml` includes:

- `stage`: the composition-root stage name.
- `model_path`: path to the persisted `model.pt` checkpoint.
- `dataset_path`: path to the CSV dataset to evaluate.
- `classification_threshold`: probability cutoff used to compute accuracy, precision, recall, and F1.

## Outputs

The stage writes outputs under `data/experiments/<experiment_name>/run_###/`.
Each execution gets the next numbered run folder so repeated evaluations do not mix artifacts.

- `config.json`
- `train.log`
- `metrics.jsonl`
- `run_state.json`
- `evaluation.json`

`evaluation.json` contains:

- checkpoint metadata (`model_path`, `model_type`, and `model_config` when present)
- evaluation dataset path
- metrics: `loss`, `accuracy`, `precision`, `recall`, `f1`, and `pr_auc`
