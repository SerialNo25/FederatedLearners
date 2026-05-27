# FederatedLearners Experiment Codebase

This codebase contains a modular experiment pipeline for evaluating redundancy of self-data inclusion in ensemble settings with federated fraud detection on tabular transaction data. It compares local bank models, inclusive federated models, exclusive federated models, and combinations of ensembles to estimate when federated knowledge adds value. It is the experiment implementation of the project of Team **Federated Learners** of the HSG FS2026 Deep Learning course.

The project uses stage-based orchestration with Pydantic-validated TOML configs. The experiments are based on TabNet models running in PyTorch. Hyperparameter optimization uses Optuna. Generated datasets, checkpoints, metrics, and reports are written under `data/`.

## Contents

- [Experiment Design](#experiment-design)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Mixed Evaluation Protocol](#mixed-evaluation-protocol)
- [Baselines and Monitoring](#baselines-and-monitoring)
- [Outputs](#outputs)
- [Documentation](#documentation)

## Experiment Design

The core comparison is:

- `Lk`: a local model trained for bank `k`
- `Fincl`: a federated model trained with all banks
- `Fexcl`: a federated model trained without bank `k`
- `Lk + Fexcl` and `Lk + Fincl`: weighted ensembles combining local and federated predictions

This project uses our proposed Mixed evaluation protocol to measure how each model behaves when test data combines bank distributions. The default configured strategies are:

- `33_33_33`
- `60_20_20`
- `80_10_10`
- `100_0_0`

Each strategy has dominant-bank train and test mixes for banks 1, 2, and 3 under `configs/dataset_mixer/`.

## Repository Layout

```text
composition/      Composition roots that load configs and wire stage dependencies
configs/          TOML configuration files for every stage
domain/           Reusable ML, dataset, metrics, logging, and federated logic
docs/             Stage-specific documentation
scripts/          Workflow helper scripts
stages/           Stage orchestration and stage-specific config models
tests/            Unit and integration tests
data/             Generated and local-only data, checkpoints, metrics, reports
```

Helper scripts are grouped by workflow:

```text
scripts/1_data_processing/
scripts/2_hpo/
scripts/3_training/
scripts/4_evaluation/
scripts/analysis/
```

## Installation

Requirements:

- Python 3.10+
- A local virtual environment
- Raw bank CSV files placed in `data/raw/`

Create and activate an environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you use another environment manager, install the dependencies from `pyproject.toml`.

## Data Setup

Place the raw source files here:

```text
data/raw/Bank_A.csv
data/raw/Bank_B.csv
data/raw/Bank_C.csv
```

## Running the Pipeline

The full experiment pipeline is:

1. Harmonize raw data.
2. Create ensemble validation splits.
3. Create dataset mixes for the mixed evaluation strategies.
4. Run HPO and update training configs with the selected hyperparameters.
5. Train local and federated models.

Run the preparation stages:

```bash
./scripts/1_data_processing/run_harmonized_data.sh
./scripts/4_evaluation/run_ensemble_validation_split.sh
./scripts/1_data_processing/run_all_dataset_mixer.sh
```

Run local and federated HPO:

```bash
./scripts/2_hpo/run_all_hpo.sh
```

After reviewing the Optuna results, update the relevant training configs in `configs/local_training/` and `configs/federated/`.

Train the configured models:

```bash
./scripts/3_training/run_all_training.sh
```

The training helper runs the three local models plus the inclusive and exclusive federated models.

## Mixed Evaluation Protocol

For each mixed evaluation scenario (`33_33_33`, `60_20_20`, `80_10_10`, `100_0_0`), the intended evaluation flow is:

1. Optimize ensemble parameters on the scenario-specific mixed train data.
2. Write the selected ensemble weights into the corresponding `configs/ensemble/...` files.
3. Run matrix evaluation on the scenario-specific mixed test data.

Example of one ensemble-weight optimization config:

```bash
./scripts/4_evaluation/run_ensemble_weight_optimization.sh configs/ensemble_weight_optimization/exclusive/bank_1.toml
```

Run matrix evaluation:

```bash
./scripts/4_evaluation/run_model_matrix_evaluation.sh configs/model_matrix_evaluation/default.toml
```

Current checked-in ensemble optimization and model-matrix configs are set up for `33_33_33`. To evaluate `60_20_20`, `80_10_10`, or `100_0_0`, create or update the ensemble optimization configs and matrix evaluation config so their dataset paths point at the matching files in `data/mixed_datasets/train/` and `data/mixed_datasets/test/`.

## Baselines and Monitoring

Random guessing is available as a baseline using the same evaluation metrics:

```bash
./scripts/4_evaluation/run_random_guessing_evaluation.sh
```

By default, it evaluates the bank 1 test split and uses the dataset fraud prevalence as the random positive probability. Pass `--help` to see configurable options.

Training runs can be monitored with the dashboard stage:

```bash
./scripts/analysis/run_training_dashboard.sh
```

This reads experiment artifacts under `data/experiments/` and reports run status, metrics, and training progress.

## Outputs

Most stages write reproducible artifacts under `data/`, for example:

```text
data/train_test_splits/
data/mixed_datasets/
data/ensemble_validation_splits/
data/experiments/<experiment_name>/run_NNN/
```

Experiment run directories typically contain:

```text
config.json
train.log
metrics.jsonl
model.pt
run_state.json
```

Evaluation stages additionally write JSON/CSV summaries such as `evaluation.json`, `evaluation_matrix.json`, and `evaluation_matrix.csv`.

## Documentation

Stage-specific details live in `docs/`:

- `docs/harmonized_data_stage.md`
- `docs/dataset_mixer_stage.md`
- `docs/ensemble_validation_split_stage.md`
- `docs/hyperparameter_optimization_stage.md`
- `docs/federated_hyperparameter_optimization_stage.md`
- `docs/local_training_stage.md`
- `docs/federated_training_stage.md`
- `docs/ensemble_weight_optimization_stage.md`
- `docs/model_matrix_evaluation_stage.md`
- `docs/evaluation_stage.md`
- `docs/training_dashboard_stage.md`