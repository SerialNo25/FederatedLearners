"""Run evaluation metrics for a random guessing fraud baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain.dataset.dataset_loader import InstitutionDataset, load_institution_dataset
from domain.logging.experiment_logger import StageExperimentLogger, allocate_experiment_run_dir
from domain.metrics.evaluation import InstitutionMetrics, evaluate_from_probabilities


def _positive_probability(value: str) -> float:
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise argparse.ArgumentTypeError("--positive-probability must be between 0.0 and 1.0")
    return probability


def random_guess_probabilities(
    sample_count: int,
    positive_probability: float,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    return [1.0 if rng.random() < positive_probability else 0.0 for _ in range(sample_count)]


def dataset_positive_rate(dataset: InstitutionDataset) -> float:
    if not dataset.labels:
        raise ValueError("dataset must contain at least one row")
    return sum(dataset.labels) / len(dataset.labels)


def run_random_guessing_evaluation(
    *,
    dataset_path: Path,
    output_dir: Path,
    experiment_name: str,
    positive_probability: float | None,
    seed: int,
    classification_threshold: float,
) -> Path:
    dataset = load_institution_dataset(
        institution_id="random_guessing_dataset",
        csv_path=dataset_path,
    )
    resolved_positive_probability = (
        dataset_positive_rate(dataset) if positive_probability is None else positive_probability
    )
    probabilities = random_guess_probabilities(
        sample_count=len(dataset.labels),
        positive_probability=resolved_positive_probability,
        seed=seed,
    )
    metrics = evaluate_from_probabilities(
        institution_id=dataset.institution_id,
        labels=dataset.labels,
        probabilities=probabilities,
        threshold=classification_threshold,
    )

    experiment_dir = allocate_experiment_run_dir(output_dir, experiment_name)
    logger = StageExperimentLogger(
        experiment_dir=str(experiment_dir),
        stage_name="random_guessing_evaluation",
    )
    try:
        _write_artifacts(
            experiment_dir=experiment_dir,
            logger=logger,
            dataset_path=dataset_path,
            dataset=dataset,
            metrics=metrics,
            positive_probability=resolved_positive_probability,
            positive_probability_source="dataset_prevalence" if positive_probability is None else "explicit",
            seed=seed,
            classification_threshold=classification_threshold,
            probabilities=probabilities,
        )
    finally:
        _close_logger_handlers(logger)
    return experiment_dir


def _write_artifacts(
    *,
    experiment_dir: Path,
    logger: StageExperimentLogger,
    dataset_path: Path,
    dataset: InstitutionDataset,
    metrics: InstitutionMetrics,
    positive_probability: float,
    positive_probability_source: str,
    seed: int,
    classification_threshold: float,
    probabilities: list[float],
) -> None:
    config = {
        "stage": "random_guessing_evaluation",
        "experiment_name": experiment_dir.parent.name,
        "output_dir": str(experiment_dir.parent.parent),
        "dataset_path": str(dataset_path),
        "positive_probability": positive_probability,
        "positive_probability_source": positive_probability_source,
        "classification_threshold": classification_threshold,
        "seed": seed,
    }
    metrics_payload = {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "pr_auc": metrics.pr_auc,
        "roc_auc": metrics.roc_auc,
        "fpr_at_95_recall": metrics.fpr_at_95_recall,
    }
    random_positive_rate = sum(1 for probability in probabilities if probability >= classification_threshold) / max(
        len(probabilities),
        1,
    )
    results = {
        "model_type": "random_guessing",
        "dataset_path": str(dataset_path),
        "num_samples": len(dataset.labels),
        "num_positive_labels": sum(dataset.labels),
        "dataset_positive_rate": dataset_positive_rate(dataset),
        "positive_probability": positive_probability,
        "positive_probability_source": positive_probability_source,
        "random_positive_rate": random_positive_rate,
        "classification_threshold": classification_threshold,
        "seed": seed,
        "metrics": metrics_payload,
    }

    _write_run_state(experiment_dir, status="running")
    logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
    logger.info(f"config={json.dumps(config, indent=2)}")
    logger.info(
        "random_guessing_evaluation_complete "
        f"num_samples={len(dataset.labels)} positive_probability={positive_probability:.6f} "
        f"random_positive_rate={random_positive_rate:.6f} loss={metrics.loss:.6f} "
        f"accuracy={metrics.accuracy:.6f} precision={metrics.precision:.6f} "
        f"recall={metrics.recall:.6f} f1={metrics.f1:.6f} pr_auc={metrics.pr_auc:.6f} "
        f"roc_auc={metrics.roc_auc:.6f} fpr_at_95_recall={metrics.fpr_at_95_recall:.6f}"
    )
    (experiment_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (experiment_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.write_metrics(
        step="random_guessing_evaluation",
        values={
            "epoch": 1,
            "train_loss": None,
            "val_loss": metrics.loss,
            "learning_rate": None,
            "classification_threshold": classification_threshold,
            "metrics": metrics_payload,
        },
    )
    _write_run_state(experiment_dir, status="completed")


def _write_run_state(experiment_dir: Path, status: str) -> None:
    (experiment_dir / "run_state.json").write_text(
        json.dumps(
            {
                "stage": "random_guessing_evaluation",
                "status": status,
                "experiment_name": experiment_dir.parent.name,
                "run_id": experiment_dir.name,
                "run_dir": str(experiment_dir),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _close_logger_handlers(logger: StageExperimentLogger) -> None:
    for handler in logger.logger.handlers[:]:
        handler.close()
        logger.logger.removeHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a random guessing baseline with the repository evaluation metrics.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/train_test_splits/bank_1_test.csv"),
        help="CSV dataset to evaluate. Defaults to bank_1 test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Root directory for experiment artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        default="eval_random_guessing",
        help="Experiment folder name under the output directory.",
    )
    parser.add_argument(
        "--positive-probability",
        type=_positive_probability,
        default=None,
        help="Probability of randomly guessing fraud. Defaults to the dataset fraud prevalence.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible guesses.")
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold used by the evaluation metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_random_guessing_evaluation(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        positive_probability=args.positive_probability,
        seed=args.seed,
        classification_threshold=args.classification_threshold,
    )
    evaluation = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    metrics = evaluation["metrics"]
    print(f"Random guessing evaluation written to {output_dir}")
    print(
        "positive_probability="
        f"{evaluation['positive_probability']:.6f} "
        f"source={evaluation['positive_probability_source']} "
        f"pr_auc={metrics['pr_auc']:.6f} "
        f"roc_auc={metrics['roc_auc']:.6f} "
        f"f1={metrics['f1']:.6f}"
    )


if __name__ == "__main__":
    main()
