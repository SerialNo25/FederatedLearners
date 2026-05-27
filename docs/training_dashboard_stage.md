# Training Dashboard Stage

## Purpose

The `training_dashboard` stage starts a browser-based monitor for local and federated
training runs.
It reads experiment artifacts from `data/experiments/` and does not own model training.

The dashboard has navigation tabs for:

- local training runs
- federated training runs

It reads each numbered run folder:

```text
data/experiments/<experiment_name>/run_###/
```

It also continues to read legacy flat experiment folders. Each dashboard run name uses the
`<experiment_name>/<run_id>` format, for example `local_bank_1_tabnet/run_001`.

Required artifacts:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `run_state.json`
- `model.pt`

## Run

```bash
python main.py --config configs/training_dashboard/local.toml
```

Helper script:

```bash
./scripts/analysis/run_training_dashboard.sh
```

The default dashboard listens on:

```text
http://127.0.0.1:8765
```

Start the dashboard in one terminal, then start local training in another terminal:

```bash
./scripts/training/run_local_training_bank_1.sh
```

## Local Live Metrics

Local training writes one `metrics.jsonl` record per epoch. The dashboard polls these artifacts
and displays:

- run status
- current epoch
- train loss
- validation loss
- validation PR-AUC
- training log tail

## Federated Live Metrics

Federated training writes one `metrics.jsonl` record per round. The dashboard polls the same
artifact root and displays both inclusive global runs and exclusive `banks_i_j`-style runs.

The federated tab displays:

- run status
- current round
- run type: global, exclusive, or federated
- participating institutions
- FedProx `proximal_mu`
- weighted train loss
- validation loss and PR-AUC
- per-institution samples, local loss, evaluation loss, PR-AUC, F1, and parameter delta L2
- training log tail

The dashboard infers `global` runs from experiment names containing `global` and exclusive
runs from names containing `banks_`. Federated runs should still use the normal experiment
structure:

```text
data/experiments/<experiment_name>/run_###/
```

## Configuration

`configs/training_dashboard/local.toml` controls:

- `experiments_dir`: experiment artifact root
- `host`: HTTP bind host
- `port`: HTTP port
- `poll_interval_seconds`: browser refresh interval
- `active_timeout_seconds`: how long a non-completed run is considered active after its latest artifact update
