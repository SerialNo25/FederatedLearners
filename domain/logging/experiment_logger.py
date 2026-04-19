import json
import logging
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR_PREFIX = "run_"


def allocate_experiment_run_dir(output_dir: Path, experiment_name: str) -> Path:
    """Create and return the next run directory for an experiment."""
    experiment_root = output_dir / experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)

    next_index = _next_run_index(experiment_root)
    while True:
        run_dir = experiment_root / f"{RUN_DIR_PREFIX}{next_index:03d}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            next_index += 1
            continue
        return run_dir


def _next_run_index(experiment_root: Path) -> int:
    run_indices: list[int] = []
    for path in experiment_root.iterdir():
        if not path.is_dir() or not path.name.startswith(RUN_DIR_PREFIX):
            continue
        suffix = path.name.removeprefix(RUN_DIR_PREFIX)
        if suffix.isdigit():
            run_indices.append(int(suffix))
    return max(run_indices, default=0) + 1


class StageExperimentLogger:
    """Persists stage-level experiment artifacts to a local directory."""

    def __init__(self, experiment_dir: str, stage_name: str) -> None:
        self.stage_name = stage_name
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.experiment_dir / "train.log"
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        self.profile_path = self.experiment_dir / "profile.jsonl"

        self._setup_logger()

    def _setup_logger(self) -> None:
        logger_name = f"stage.{self.stage_name}.{self.experiment_dir.parent.name}.{self.experiment_dir.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def exception(self, message: str) -> None:
        self.logger.exception(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    def write_metrics(self, step: str, values: dict[str, object]) -> None:
        record = {
            "step": step,
            "stage": self.stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(f"{json.dumps(record, ensure_ascii=False)}\n")

    def write_profile(self, step: str, values: dict[str, object]) -> None:
        record = {
            "step": step,
            "stage": self.stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        with self.profile_path.open("a", encoding="utf-8") as profile_file:
            profile_file.write(f"{json.dumps(record, ensure_ascii=False)}\n")
