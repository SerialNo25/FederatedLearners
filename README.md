# Federated Learning Demo (PyTorch + Flower)

This repository contains a **minimal full-stack federated learning demo** with:

- A simple PyTorch model (`SimpleClassifier`)
- Three institutions (`institution_id` = 0, 1, 2), each with its own local data pool
- Flower orchestration to aggregate local updates into a global model
- Entry points for both **single-command simulation** and **server/client mode**

## What this demo shows

1. Each institution trains locally on its own non-IID synthetic dataset.
2. Flower's FedAvg strategy aggregates local model updates into a global model.
3. The global model is evaluated back on each institution.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Entry points

After installation, these commands are available:

- `fl-demo` — run an in-process 3-institution simulation and report per-institution global model metrics.
- `fl-server` — start a Flower server and save global model checkpoints.
- `fl-client` — start a single institution client that connects to the server.
- `fl-evaluate` — evaluate a saved global model on one institution.

---

## Quickstart: single-command choreography demo

```bash
fl-demo --rounds 3 --local-epochs 1
```

This runs all three institutions plus server orchestration in one process and then prints how the global model performs on each institution.

---

## Full stack mode (server + three clients)

### 1) Start server

```bash
fl-server --address 0.0.0.0:8080 --rounds 3 --output global_model.npz
```

### 2) Start three institutions (three terminals)

```bash
fl-client --institution-id 0 --server-address 127.0.0.1:8080
fl-client --institution-id 1 --server-address 127.0.0.1:8080
fl-client --institution-id 2 --server-address 127.0.0.1:8080
```

### 3) Evaluate the final global model at each institution

```bash
fl-evaluate --model-path global_model.npz --institution-id 0
fl-evaluate --model-path global_model.npz --institution-id 1
fl-evaluate --model-path global_model.npz --institution-id 2
```

---

## Project layout

- `federated_demo/model.py` — PyTorch model + parameter conversion helpers.
- `federated_demo/data.py` — institution-specific synthetic data pools.
- `federated_demo/train_eval.py` — local train/eval routines.
- `federated_demo/client.py` — Flower client implementation.
- `federated_demo/server.py` — Flower server + checkpointing strategy.
- `federated_demo/main.py` — in-process 3-party federated choreography demo.
- `federated_demo/evaluate_global.py` — evaluate saved global model per institution.
