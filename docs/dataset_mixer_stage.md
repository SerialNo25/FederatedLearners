# Dataset Mixer Stage

The `dataset_mixer` stage builds a single mixed CSV by sampling a configured number of rows from multiple bank datasets and shuffling the combined result reproducibly.

It follows the repository architecture:

- `main.py` reads the `stage` value from the TOML config and dispatches to the matching composition root.
- `composition/run_dataset_mixer.py` loads and validates TOML config, wires dependencies, and executes the stage.
- `stages/dataset_mixer/stage.py` orchestrates loading, sampling, merging, and artifact writing.
- `domain/dataset/dataset_loader.py` contains the reusable dataset IO, sampling, merging, and metadata-writing logic.

## Default Run

```bash
python main.py --config configs/dataset_mixer/bank_1_dominant.toml
```

Helper script:

```bash
./scripts/run_dataset_mixer.sh
```

## Output

The stage writes a mixed dataset CSV and a sidecar JSON summary under `data/mixed_datasets/`:

- `bank_1_dominant_80_10_10.csv`
- `bank_1_dominant_80_10_10.json`

Equivalent artifacts are produced for the other configs.

## Notes

- Sampling is random without replacement.
- The output CSV keeps the standard fraud schema unchanged for downstream compatibility.
- The sidecar JSON records the seed, total size, fraud count, and per-institution sampled row counts for reproducibility.
