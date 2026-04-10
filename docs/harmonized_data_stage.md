# Harmonized Data Stage

The `harmonized_data` stage converts the raw bank-specific CSV files into the shared 21-feature fraud schema used by the rest of the pipeline.

It follows the repository architecture:

- `main.py` selects the stage from the registry.
- `composition/run_harmonized_data.py` loads and validates TOML config, wires dependencies, and executes the stage.
- `stages/harmonized_data/stage.py` orchestrates raw-to-harmonized dataset creation and schema validation.
- `domain/harmonization/raw_data_harmonizer.py` contains the reusable bank-specific transformation logic.

## Default Run

```bash
python main.py harmonized_data --preset default
```

Helper script:

```bash
./scripts/run_harmonized_data.sh
```

## Output

The stage writes harmonized datasets to `data/harmonized/` by default:

- `bank_a_sparkov.csv`
- `bank_b_banksim.csv`
- `bank_c_ccfraud.csv`

Each output CSV is validated against the shared dataset schema in `domain/dataset/schema.py`, so downstream stages can consume it directly.

## Notes

- Sparkov is subsampled to `sparkov_target_size` while preserving the observed fraud ratio.
- Amount z-score and percentile statistics are computed within each source dataset.
- Category mappings and engineered time/geo features follow the harmonization specification in `configs/harmonized_data/data_sources.txt`.
