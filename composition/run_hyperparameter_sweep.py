"""Hyperparameter sweep for local fraud-detection training.

Trains one model per (fraud_weight × learning_rate) combination, then evaluates
each model across multiple classification thresholds without retraining.

Usage (from FederatedLearners/):
    python -m composition.run_hyperparameter_sweep
"""

from __future__ import annotations

import csv
import itertools
import time

import torch
from dataclasses import dataclass
from pathlib import Path

from domain.dataset.dataset_loader import load_institution_dataset, split_dataset
from domain.metrics.evaluation import evaluate_institution
from domain.models.model_registry import MODEL_REGISTRY
from domain.training.trainer import TrainingConfig, train_local_model

# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------

FRAUD_WEIGHTS = [100, 730, 1200] # [10, 25, 50, 100]
LEARNING_RATES = [0.005, 0.01, 0.02]
EPOCHS = [5, 10, 25]  # mini-batch: each epoch = ~313 updates on bank_2, ~1737 on bank_1
BATCH_SIZE = 256
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Set to a fraction (e.g. 0.2) for a faster low-fidelity sweep.
# Hyperparameter rankings are stable on subsets — use 1.0 only for the final run.
SUBSAMPLE_FRACTION: float = 0.2
MODEL_CONFIG = {
    "model_type": "tabnet",
    "decision_dim": 16,
    "attention_dim": 16,
    "steps": 4,
    "relaxation_factor": 1.5,
    "sparsity_weight": 0.0001,
}

INSTITUTION_ID = "bank_3"
DATASET_PATH = Path(f"configs/sample_data/{INSTITUTION_ID}.csv")
SEED = 42
OUTPUT_DIR = Path("data/experiments")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    fraud_weight: float
    learning_rate: float
    epochs: int
    full_dataset_epochs: int  # epochs scaled for full dataset — use this in local config
    threshold: float
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    train_seconds: float


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep() -> list[SweepResult]:
    dataset = load_institution_dataset(
        institution_id=INSTITUTION_ID,
        csv_path=DATASET_PATH,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=0.2, seed=SEED)

    if SUBSAMPLE_FRACTION < 1.0:
        # Stratified subsample of training data only — val set stays full for fair evaluation
        train_dataset, _ = split_dataset(train_dataset, val_fraction=1.0 - SUBSAMPLE_FRACTION, seed=SEED)

    n_fraud_train = sum(train_dataset.labels)
    n_fraud_val = sum(val_dataset.labels)
    subsample_note = f" [{SUBSAMPLE_FRACTION:.0%} subsample]" if SUBSAMPLE_FRACTION < 1.0 else ""
    print(
        f"Dataset loaded — train: {len(train_dataset.features):,} samples ({n_fraud_train} fraud){subsample_note} | "
        f"val: {len(val_dataset.features):,} samples ({n_fraud_val} fraud)\n"
    )

    training_combinations = list(itertools.product(FRAUD_WEIGHTS, LEARNING_RATES, EPOCHS))
    total_runs = len(training_combinations)
    results: list[SweepResult] = []

    for run_idx, (fraud_weight, lr, epochs) in enumerate(training_combinations, start=1):
        print(
            f"[{run_idx}/{total_runs}] Training  fraud_weight={fraud_weight:<5}  lr={lr}  epochs={epochs}",
            end="  ",
            flush=True,
        )

        torch.manual_seed(SEED)
        model_factory = MODEL_REGISTRY.get_factory("tabnet", MODEL_CONFIG)
        model = model_factory(len(train_dataset.features[0]))

        t0 = time.perf_counter()
        train_loss = train_local_model(
            model=model,
            features=train_dataset.features,
            labels=train_dataset.labels,
            config=TrainingConfig(
                learning_rate=lr,
                local_epochs=epochs,
                proximal_mu=0.0,
                fraud_weight=fraud_weight,
                batch_size=BATCH_SIZE,
            ),
        )
        elapsed = time.perf_counter() - t0
        print(f"({elapsed:.1f}s)  sweeping thresholds ...", flush=True)

        for threshold in THRESHOLDS:
            metrics = evaluate_institution(
                model,
                val_dataset,
                pos_weight=fraud_weight,
                threshold=threshold,
            )
            full_epochs = max(1, round(epochs * SUBSAMPLE_FRACTION)) if SUBSAMPLE_FRACTION < 1.0 else epochs
            results.append(SweepResult(
                fraud_weight=fraud_weight,
                learning_rate=lr,
                epochs=epochs,
                full_dataset_epochs=full_epochs,
                threshold=threshold,
                train_loss=train_loss,
                val_loss=metrics.loss,
                val_accuracy=metrics.accuracy,
                val_precision=metrics.precision,
                val_recall=metrics.recall,
                val_f1=metrics.f1,
                train_seconds=elapsed,
            ))

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "fraud_wt": 9,
    "lr":       7,
    "epochs":   7,
    "full_ep":  8,
    "thresh":   7,
    "train_loss": 11,
    "val_loss":   9,
    "accuracy":   9,
    "precision":  10,
    "recall":     7,
    "f1":         7,
    "secs":       5,
}

_HEADERS = list(_COL_WIDTHS.keys())


def _row_str(r: SweepResult, highlight: bool = False) -> str:
    cells = [
        f"{r.fraud_weight:<{_COL_WIDTHS['fraud_wt']}.0f}",
        f"{r.learning_rate:<{_COL_WIDTHS['lr']}.3f}",
        f"{r.epochs:<{_COL_WIDTHS['epochs']}}",
        f"{r.full_dataset_epochs:<{_COL_WIDTHS['full_ep']}}",
        f"{r.threshold:<{_COL_WIDTHS['thresh']}.2f}",
        f"{r.train_loss:<{_COL_WIDTHS['train_loss']}.4f}",
        f"{r.val_loss:<{_COL_WIDTHS['val_loss']}.4f}",
        f"{r.val_accuracy:<{_COL_WIDTHS['accuracy']}.4f}",
        f"{r.val_precision:<{_COL_WIDTHS['precision']}.4f}",
        f"{r.val_recall:<{_COL_WIDTHS['recall']}.4f}",
        f"{r.val_f1:<{_COL_WIDTHS['f1']}.4f}",
        f"{r.train_seconds:<{_COL_WIDTHS['secs']}.1f}",
    ]
    line = "  ".join(cells)
    return f">>> {line} <<<" if highlight else f"    {line}"


def print_table(results: list[SweepResult]) -> None:
    sorted_results = sorted(results, key=lambda r: r.val_f1, reverse=True)
    best = sorted_results[0]

    header_cells = [f"{h:<{w}}" for h, w in _COL_WIDTHS.items()]
    header = "    " + "  ".join(header_cells)
    separator = "    " + "-" * (sum(_COL_WIDTHS.values()) + 2 * (len(_COL_WIDTHS) - 1))

    print("\n" + "=" * len(separator))
    print("  RESULTS  (sorted by val_f1 descending, >>> best <<<)")
    print("=" * len(separator))
    print(header)
    print(separator)
    for r in sorted_results:
        print(_row_str(r, highlight=(r is best)))
    print(separator)
    print(
        f"\n  Best  →  fraud_weight={best.fraud_weight:.0f}  lr={best.learning_rate:.3f}  "
        f"threshold={best.threshold:.2f}  "
        f"precision={best.val_precision:.4f}  recall={best.val_recall:.4f}  f1={best.val_f1:.4f}"
        f"\n  Use in local config  →  local_epochs = {best.full_dataset_epochs}"
    )


def save_csv(results: list[SweepResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fraud_weight", "learning_rate", "sweep_epochs", "full_dataset_epochs", "threshold",
            "train_loss", "val_loss", "val_accuracy",
            "val_precision", "val_recall", "val_f1", "train_seconds",
        ])
        for r in sorted(results, key=lambda r: r.val_f1, reverse=True):
            writer.writerow([
                r.fraud_weight, r.learning_rate, r.epochs, r.full_dataset_epochs, r.threshold,
                round(r.train_loss, 6), round(r.val_loss, 6), round(r.val_accuracy, 6),
                round(r.val_precision, 6), round(r.val_recall, 6), round(r.val_f1, 6),
                round(r.train_seconds, 2),
            ])
    print(f"\n  CSV saved → {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(f"  Hyperparameter sweep — fraud detection ({INSTITUTION_ID}, TabNet)")
    print(f"  Grid: {len(FRAUD_WEIGHTS)} fraud_weights × {len(LEARNING_RATES)} learning_rates × {len(EPOCHS)} epoch counts = "
          f"{len(FRAUD_WEIGHTS) * len(LEARNING_RATES) * len(EPOCHS)} training runs")
    fidelity = f"{SUBSAMPLE_FRACTION:.0%} subsample" if SUBSAMPLE_FRACTION < 1.0 else "full dataset"
    print(f"  Batch size: {BATCH_SIZE}  |  Fidelity: {fidelity}  |  Threshold sweep: {THRESHOLDS} (no retraining)")
    print("=" * 60 + "\n")

    results = run_sweep()
    print_table(results)

    csv_path = OUTPUT_DIR / f"hyperparameter_sweep_{INSTITUTION_ID}.csv"
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
