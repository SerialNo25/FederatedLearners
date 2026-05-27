# Local Training Stage

## Purpose

The `local_training` stage trains a model for a single institution without federated aggregation.
It reuses the same model registry and model configuration schema as the federated stage, so the
`[model]` block can be shared directly between `federated_training` and `local_training`.

## Run

```bash
python main.py --config configs/local_training/default.toml
python main.py --config configs/local_training/bank_1.toml
python main.py --config configs/local_training/bank_2.toml
python main.py --config configs/local_training/bank_3.toml
```

You may keep the stage name as a CLI guard if desired:

```bash
python main.py local_training --config configs/local_training/default.toml
```

Helper scripts:

```bash
./scripts/training/run_local_training.sh
./scripts/training/run_local_training_bank_1.sh
./scripts/training/run_local_training_bank_2.sh
./scripts/training/run_local_training_bank_3.sh
```

The `bank_1`, `bank_2`, and `bank_3` config files each set a distinct `experiment_name`, so artifacts are written into separate experiment folders under `data/experiments/`.
Each execution gets the next numbered run folder, such as `run_001` and `run_002`, so repeated executions never mix artifacts.

## Configuration compatibility

`local_training` intentionally accepts federated-style configuration files. It ignores federated-only
keys (for example `num_rounds` and `proximal_mu`) and trains only one institution selected by
`local_institution_id`.

If `local_institution_id` is omitted, the first entry in `[[institutions]]` is used.

## Outputs

The stage writes artifacts under `data/experiments/<experiment_name>/run_###/`:

- `config.json`
- `train.log`
- `metrics.jsonl`
- `run_state.json`
- `model.pt`

`metrics.jsonl` includes one record per local epoch with `train_loss` and `val_loss`.
`run_state.json` records the stage, status, experiment name, run ID, and run directory.
