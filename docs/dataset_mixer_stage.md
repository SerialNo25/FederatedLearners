# Dataset Mixer Stage

The `dataset_mixer` stage builds a single mixed CSV by sampling a configured number of rows from multiple bank datasets and shuffling the combined result reproducibly.

It follows the repository architecture:

- `main.py` reads the `stage` value from the TOML config and dispatches to the matching composition root.
- `composition/run_dataset_mixer.py` loads and validates TOML config, wires dependencies, and executes the stage.
- `stages/dataset_mixer/stage.py` orchestrates loading, sampling, merging, and artifact writing.
- `domain/dataset/dataset_loader.py` contains the reusable dataset IO, sampling, merging, and metadata-writing logic.

## Default Run

```bash
python main.py --config configs/dataset_mixer/80_10_10/train/bank_1_dominant.toml
```

Helper script:

```bash
./scripts/data_processing/run_dataset_mixer.sh
```

Run the full config matrix:

```bash
./scripts/data_processing/run_all_dataset_mixer.sh
```

## Output

The stage writes a mixed dataset CSV and a sidecar JSON summary under `data/mixed_datasets/`:

- `train/bank_1_dominant_80_10_10.csv`
- `train/bank_1_dominant_80_10_10.json`

Equivalent artifacts are produced for the other configs.

## Config Matrix

Dataset mixer configs live under:

```text
configs/dataset_mixer/<split>/<scenario>/bank_<n>_dominant.toml
```

The repository now includes four split families for both `train` and `test` scenarios:

- `100_0_0`
- `80_10_10`
- `60_20_20`
- `33_33_33`

The scenario selects the corresponding source files from `data/train_test_splits/*_<scenario>.csv`.

To keep every dominant-bank config feasible with the currently checked-in source datasets, totals are scenario-specific:

- `train`: 80,000 total rows
- `test`: 20,000 total rows

`100_0_0` therefore requires zero-row source entries for the non-dominant institutions, which the stage now supports directly.

## Notes

- Sampling is random without replacement.
- The output CSV keeps the standard fraud schema unchanged for downstream compatibility.
- The sidecar JSON records the seed, total size, fraud count, and per-institution sampled row counts for reproducibility.
