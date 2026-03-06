import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



class StageExperimentLogger:
    """Persists stage-level experiment artifacts to a local directory."""

    def __init__(self, experiment_dir: str, config: dict[str, Any], stage_name: str) -> None:
        self.stage_name = stage_name
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.experiment_dir / "config.json"
        self.log_path = self.experiment_dir / "train.log"
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        self.model_path = self.experiment_dir / "model.pt"

        self._write_config(config)
        self._setup_logger()
        self._ensure_model_artifact()

    def _write_config(self, config: dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)

    def _setup_logger(self) -> None:
        logger_name = f"stage.{self.stage_name}.{self.experiment_dir.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def _ensure_model_artifact(self) -> None:
        self.model_path.touch(exist_ok=True)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def write_metrics(self, step: str, values: dict[str, Any]) -> None:
        record = {
            "step": step,
            "stage": self.stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(f"{json.dumps(record, ensure_ascii=False)}\n")

