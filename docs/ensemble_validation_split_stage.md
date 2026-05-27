# Ensemble Validation Split Stage

The `ensemble_validation_split` stage materializes the exact validation subset used by the `local_training` stage for internal evaluation. It exists so ensemble-weight fitting can reuse the already-established validation rows without retraining local models.

## Why this stage exists

`local_training` does not read a precomputed validation CSV. Instead, it loads each institution's train dataset and derives an internal validation split with:

- the same deterministic stratified splitter from `domain/dataset/dataset_loader.py`
- the same seed passed into `local_training`
- the same validation fraction of `0.2`

This stage calls that same shared helper when `validation_fraction = 0.2`, so the generated CSVs match the historical local-training validation split exactly for the same input dataset and seed.

## Default Run

```bash
python main.py --config configs/pipeline/ensemble_validation_split.toml
```

Helper script:

```bash
./scripts/evaluation/run_ensemble_validation_split.sh
```

## Output

The stage writes one validation CSV plus a small metadata JSON file per institution under `data/ensemble_validation_splits/`:

- `bank_1_validation.csv`
- `bank_1_validation.json`
- `bank_2_validation.csv`
- `bank_2_validation.json`
- `bank_3_validation.csv`
- `bank_3_validation.json`

The JSON records the source dataset path, seed, validation fraction, output row count, and whether the stage was run with the exact local-training validation fraction.

## Important

To reproduce the split authentically, use the same `seed` that was used when `local_training` originally ran. Changing the seed or the source train CSV changes the validation rows.
