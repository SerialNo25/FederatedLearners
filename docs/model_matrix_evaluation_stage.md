# Model Matrix Evaluation Stage

The model matrix evaluation stage evaluates a shared list of model runs against each
configured dataset.

Default execution:

```bash
scripts/evaluation/run_model_matrix_evaluation.sh
```

Custom config execution:

```bash
scripts/evaluation/run_model_matrix_evaluation.sh configs/model_matrix_evaluation/default.toml
```

The config stores a base experiment path and a run number for each checkpoint model.
The stage resolves checkpoints as:

```text
{base_path}/run_{run_number:03d}/model.pt
```

The default config contains:

- 3 local models
- 1 global federated model
- 3 exclusive federated models
- 3 exclusive ensembles
- 3 inclusive ensembles

Every configured model is evaluated on every configured dataset. Results are written
under `data/experiments/model_matrix_evaluation/run_NNN/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `evaluation_matrix.json`
- `evaluation_matrix.csv`

Ensemble models do not define their own checkpoint paths. They reference configured
checkpoint model IDs and combine probabilities as:

```text
ensemble = w * local + (1 - w) * federated
```

Each ensemble entry points to its own config TOML under `configs/ensemble/`, and the
stage loads `ensemble_weight` from that referenced config. This keeps matrix
evaluation aligned with the per-ensemble weights used by the standalone ensemble
stage.
