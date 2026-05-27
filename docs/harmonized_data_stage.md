# Harmonized Data Stage

The `harmonized_data` stage converts the raw bank-specific CSV files into the shared 21-feature fraud schema used by the rest of the pipeline.
It splits raw rows before fitting preprocessing statistics, then applies the train-fitted transforms to every subset.

It follows the repository architecture:

- `main.py` reads the `stage` value from the TOML config and dispatches to the matching composition root.
- `composition/run_harmonized_data.py` loads and validates TOML config, wires dependencies, and executes the stage.
- `stages/harmonized_data/stage.py` orchestrates raw-to-harmonized dataset creation and schema validation.
- `domain/harmonization/raw_data_harmonizer.py` contains the reusable bank-specific transformation logic.

## Default Run

```bash
python main.py --config configs/pipeline/harmonized_data.toml
```

Helper script:

```bash
./scripts/data_processing/run_harmonized_data.sh
```

## Output

The stage writes harmonized train/test datasets to `data/train_test_splits/` by default:

- `bank_1_train.csv`
- `bank_1_test.csv`
- `bank_1_preprocessing.json`
- equivalent train/test/artifact files for each configured institution

Each output CSV is validated against the shared dataset schema in `domain/dataset/schema.py`, so downstream stages can consume it directly.

## Notes

- Sparkov is subsampled to `sparkov_target_size` while preserving the observed fraud ratio.
- Raw rows are stratified into train/test subsets before harmonization.
- Amount z-score, amount percentile, geo, and country-mapping statistics are fit from the train subset only and reused for the test subset.
- `*_preprocessing.json` records the fitted preprocessing policy and train-only summary statistics for reproducibility.
- Category mappings and engineered time/geo features follow the harmonization specification in `configs/harmonized_data/data_sources.txt`.
