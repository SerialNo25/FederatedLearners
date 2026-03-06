# AGENTS.md

## Overview

This repository implements a **modular experiment pipeline for Federated Fraud Detection**.

The system is designed to evaluate the **marginal contribution of federated knowledge** by comparing:

* **Local models (Lk)** trained on each bank’s data
* **Inclusive federated models (Fincl)** trained across all banks
* **Exclusive federated models (Fexcl)** trained without bank *k*
* **Ensembles (Lk + Fexcl)** combining local and external knowledge

The project simulates **multiple financial institutions (banks)** using structured dataset partitioning and trains **TabNet models** using **FedProx aggregation** orchestrated via **Flower**. 

The repository enforces:

* Clean separation between **CLI**, **experiment orchestration**, and **core ML logic**
* Strict configuration validation using **Pydantic**
* Reproducible **experiment pipelines**
* Modular **stage-based experiment workflows**

---

# Architectural Principles

## 1. Thin CLI Layer

The CLI is responsible only for:

* Parsing arguments
* Selecting the pipeline stage
* Delegating execution

The CLI **must not contain ML or experiment logic**.

---

## 2. Experiment Pipeline via Stages

Each step in the experiment pipeline is represented as a **Stage**.

Examples include:

* Dataset preparation and partitioning
* Local model training
* Federated training
* Ensemble evaluation
* Metrics analysis

Stages:

* Orchestrate workflows
* Call domain services
* Do **not implement core ML algorithms**

---

## 3. Core ML Logic Isolation

Domain logic lives inside **core modules**.

Examples:

* TabNet model definitions
* Training loops
* Federated aggregation logic
* Dataset loaders
* Evaluation metrics

Core modules must:

* Be reusable outside the CLI
* Not depend on Stage classes
* Not depend on Flower orchestration code

---

## 4. Validated Configuration

All configuration is defined using **Pydantic models**.

Configuration includes:

* Dataset settings
* Partitioning strategies
* Model hyperparameters
* Federated training parameters
* Experiment parameters

Rules:

* No raw dictionaries
* Config is validated before execution
* Defaults live inside config models

---

## 5. Explicit Dependency Wiring

All dependencies are created in **composition roots (`run_*`)**.

Responsibilities:

* Load TOML configuration
* Validate configuration
* Construct shared services
* Instantiate stages
* Execute stages

There must be:

* No hidden global state
* No implicit dependency creation

---

# File Structure

The project is organized around **domains and experiment stages**.

```
project/
│
├── main.py
│
├── run/
│   ├── run_partition.py
│   ├── run_local_training.py
│   ├── run_federated_training.py
│   ├── run_ensemble.py
│   └── run_evaluation.py
│
├── stages/
│   ├── partition/
│   ├── local_training/
│   ├── federated_training/
│   ├── ensemble/
│   └── evaluation/
│
├── core/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── federated/
│   └── metrics/
│
├── configs/
│
├── data/
│
└── docs/
```

---

# Layered Architecture

## Layer 1 — CLI Entry (`main.py`)

Responsibilities:

* Parse command-line arguments
* Select stage
* Delegate execution to composition roots

Example:

```
python main.py federated_training --config configs/federated.toml
```

The CLI **must not contain experiment logic**.

---

# Layer 2 — Composition Roots (`run_*`)

Composition roots connect configuration with execution.

Responsibilities:

* Load TOML configuration
* Validate configuration using Pydantic
* Construct shared services
* Instantiate the appropriate Stage
* Execute the stage

Example:

```
run_local_training()
run_federated_training()
run_ensemble()
```

This layer is responsible for **dependency wiring only**.

---

# Layer 3 — Stage System

Stages represent **experiment steps**.

Each stage:

* Orchestrates a workflow
* Validates stage-specific config
* Calls domain services
* Produces outputs for downstream stages

Stages do **not implement ML algorithms**.

---

## Stage Registry

Stages are registered in a **StageRegistry**.

This enables:

* Discoverable experiment stages
* Simple CLI routing
* Extensible experiment pipelines

Example:

```
partition
local_training
federated_training
ensemble
evaluation
```

---

# Individual Stage Responsibilities

## Dataset Partition Stage

Simulates **multiple financial institutions** by splitting the dataset into silos.

Responsibilities:

* Perform label-skew stratification
* Create size-imbalanced silos
* Apply feature clustering
* Produce bank datasets

Output:

```
data/banks/
  bank_1/
  bank_2/
  bank_3/
```

---

## Local Training Stage

Trains a **local TabNet model (Lk)** for each bank.

Responsibilities:

* Load bank dataset
* Train TabNet model
* Save checkpoints
* Record metrics

---

## Federated Training Stage

Runs **Flower-based federated learning** using **FedProx aggregation**.

Responsibilities:

* Initialize Flower server
* Spawn simulated bank clients
* Train global model
* Save global checkpoint

Two modes:

### Inclusive Federated Model

```
Fincl
```

Uses **all banks** in training.

### Exclusive Federated Model

```
Fexcl
```

Trains a federated model **excluding bank k**.

---

## Ensemble Stage

Evaluates **ensemble predictions**.

Responsibilities:

* Load local model
* Load exclusive federated model
* Combine predictions

Example:

```
prediction = w * Lk + (1-w) * Fexcl
```

---

## Evaluation Stage

Computes experiment metrics.

Primary metric:

```
PR-AUC
```

Secondary metrics:

* ROC-AUC
* F1 score
* False Positive Rate

Also includes:

* Prediction correlation analysis
* Model redundancy analysis

---

# Layer 4 — Pydantic Config Models

Each stage defines its own configuration schema.

Example:

```
PartitionConfig
LocalTrainingConfig
FederatedTrainingConfig
EnsembleConfig
EvaluationConfig
```

Rules:

* All configs validated
* Defaults embedded in models
* No runtime dict-based configuration

---

# Layer 5 — Core Domain Logic

The **core module** contains all reusable ML logic.

Examples:

```
core/data/
    dataset_loader.py
    partitioning.py

core/models/
    tabnet_model.py

core/training/
    trainer.py
    loss.py

core/federated/
    fedprox.py
    client.py
    server.py

core/metrics/
    pr_auc.py
    evaluation.py
```

Core modules must:

* Not depend on CLI
* Not depend on Stages
* Be usable independently

---

# Execution Flow

Example:

```
python main.py local_training --config configs/local_training.toml
```

Execution steps:

1. CLI parses arguments
2. CLI selects stage
3. Composition root loads TOML config
4. Config validated with Pydantic
5. Stage instantiated
6. Stage orchestrates workflow
7. Core services perform computation

---

# Error Handling Strategy

Error handling follows layered responsibility:

Core layer

* Raises domain exceptions

Stages

* Translate domain exceptions if necessary

Composition roots

* Map failures to CLI exit codes

---

# Data Management

All stages store outputs in the `data/` directory.

Structure example:

```
data/
  partitions/
  local_models/
  federated_models/
  ensembles/
  experiments/
```

Each stage must use a **dedicated subfolder**.

---

# Logging Guidelines

All experiments must be **fully reproducible**.

External experiment tracking tools are **not permitted**.

---

## 1. Log File (`train.log`)

Human-readable file containing:

* Full experiment configuration
* Model architecture
* Debug outputs
* Warnings
* Final evaluation results

---

## 2. Metrics File (`metrics.jsonl`)

Machine-readable JSONL file.

Each line must include:

```
epoch
train_loss
val_loss
metrics
learning_rate
```

---

## 3. Required Experiment Structure

```
experiment_name/
 ├── config.json
 ├── train.log
 ├── metrics.jsonl
 └── model.pt
```

---

# Reproducibility Requirements

All experiments must record:

* Random seed
* Dataset partitioning parameters
* Model hyperparameters
* Federated training parameters
* Aggregation method

This ensures **full reproducibility of federated experiments**.

---

If you'd like, I can also help you produce a **much stronger version that aligns with ML experiment best practices** (e.g., adding an **Experiment Manager layer**, which is usually essential for projects like this).
