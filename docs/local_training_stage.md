# Local Training Stage

## Purpose

The `local_training` stage trains a model for a single institution without federated aggregation.
It reuses the same model registry and model configuration schema as the federated stage, so the
`[model]` block can be shared directly between `federated_training` and `local_training`.

## Run

```bash
python main.py local_training --preset default
python main.py local_training --preset bank_1
python main.py local_training --preset bank_2
python main.py local_training --preset bank_3
```

or

```bash
python main.py local_training --config configs/local_training.toml
```

Helper scripts:

```bash
./scripts/run_local_training.sh
./scripts/run_local_training_bank_1.sh
./scripts/run_local_training_bank_2.sh
./scripts/run_local_training_bank_3.sh
```

The `bank_1`, `bank_2`, and `bank_3` presets each set a distinct `experiment_name`, so artifacts are written into separate experiment folders under `data/experiments/`.

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
- `loss_plot.svg`
- `pr_auc.svg`
- `f1_optimal_threshold.svg`
- `threshold_curves.svg`
- `per_client_performance_boxplots.svg`
- `global_vs_local_convergence.svg`
