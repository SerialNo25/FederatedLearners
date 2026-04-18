# Training Dashboard Stage

## Purpose

The `training_dashboard` stage starts a browser-based monitor for local training runs.
It reads experiment artifacts from `data/experiments/` and does not own model training.

The dashboard focuses on `local_training` runs and reads each numbered run folder:

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
./scripts/run_training_dashboard.sh
```

The default dashboard listens on:

```text
http://127.0.0.1:8765
```

Start the dashboard in one terminal, then start local training in another terminal:

```bash
./scripts/run_local_training_bank_1.sh
```

## Live Metrics

Local training writes one `metrics.jsonl` record per epoch. The dashboard polls these artifacts
and displays:

- run status
- current epoch
- train loss
- validation loss
- validation PR-AUC
- training log tail

## Configuration

`configs/training_dashboard/local.toml` controls:

- `experiments_dir`: experiment artifact root
- `host`: HTTP bind host
- `port`: HTTP port
- `poll_interval_seconds`: browser refresh interval
- `active_timeout_seconds`: how long a non-completed run is considered active after its latest artifact update
