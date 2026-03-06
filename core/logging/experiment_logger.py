"""Core logging helpers for reproducible experiment artifacts."""

from __future__ import annotations

import logging
from pathlib import Path


def _build_file_logger(logger_name: str, file_path: Path, formatter: logging.Formatter) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_experiment_logger(experiment_dir: Path) -> logging.Logger:
    """Build logger writing human-readable experiment logs to train.log."""

    formatter = logging.Formatter("%(message)s")
    return _build_file_logger(
        logger_name=f"federated_training.train.{experiment_dir}",
        file_path=experiment_dir / "train.log",
        formatter=formatter,
    )


def build_metrics_logger(experiment_dir: Path) -> logging.Logger:
    """Build logger writing JSONL metrics lines to metrics.jsonl."""

    formatter = logging.Formatter("%(message)s")
    return _build_file_logger(
        logger_name=f"federated_training.metrics.{experiment_dir}",
        file_path=experiment_dir / "metrics.jsonl",
        formatter=formatter,
    )
