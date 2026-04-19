"""Filesystem-backed training run monitor for dashboard APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingRunMonitorConfig:
    experiments_dir: Path
    active_timeout_seconds: int = 120


class TrainingRunMonitor:
    """Reads local training run artifacts without owning training execution."""

    def __init__(self, config: TrainingRunMonitorConfig) -> None:
        self.config = config

    def list_runs(self, stage: str = "local_training") -> list[dict[str, Any]]:
        runs = [self._summarize_run(path) for path in self._candidate_run_dirs()]
        stage_runs = [run for run in runs if run is not None and run["stage"] == stage]
        return sorted(stage_runs, key=lambda run: run["updated_at"] or "", reverse=True)

    def get_run(self, run_name: str, stage: str = "local_training") -> dict[str, Any]:
        run_dir = self._resolve_run_dir(run_name)
        if run_dir is None:
            raise KeyError(run_name)

        summary = self._summarize_run(run_dir)
        if summary is None or summary["stage"] != stage:
            raise KeyError(run_name)

        return {
            **summary,
            "metrics_history": self._read_metrics(run_dir / "metrics.jsonl"),
            "profile_history": self._read_metrics(run_dir / "profile.jsonl"),
            "log_tail": self._read_log_tail(run_dir / "train.log"),
        }

    def _candidate_run_dirs(self) -> list[Path]:
        if not self.config.experiments_dir.exists():
            return []
        candidates: list[Path] = []
        for experiment_dir in self.config.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            if self._has_run_artifacts(experiment_dir):
                candidates.append(experiment_dir)
            candidates.extend(
                path
                for path in experiment_dir.iterdir()
                if path.is_dir() and self._has_run_artifacts(path)
            )
        return candidates

    def _resolve_run_dir(self, run_name: str) -> Path | None:
        if "\\" in run_name or run_name in {"", ".", ".."}:
            return None
        parts = run_name.split("/")
        if any(part in {"", ".", ".."} for part in parts) or len(parts) > 2:
            return None
        run_dir = self.config.experiments_dir.joinpath(*parts)
        try:
            run_dir.relative_to(self.config.experiments_dir)
        except ValueError:
            return None
        return run_dir

    @staticmethod
    def _has_run_artifacts(path: Path) -> bool:
        return (path / "metrics.jsonl").exists() or (path / "config.json").exists()

    def _summarize_run(self, run_dir: Path) -> dict[str, Any] | None:
        if not run_dir.exists() or not run_dir.is_dir():
            return None

        config = self._read_json_object(run_dir / "config.json")
        run_state = self._read_json_object(run_dir / "run_state.json")
        metrics = self._read_metrics(run_dir / "metrics.jsonl")
        latest = metrics[-1] if metrics else {}
        stage = str(config.get("stage") or run_state.get("stage") or latest.get("stage") or "unknown")
        updated_at = self._latest_artifact_timestamp(run_dir)
        display_name = self._display_name(run_dir)

        total_epochs = latest.get("total_epochs") or config.get("local_epochs")
        if stage == "federated_training":
            total_epochs = latest.get("total_rounds") or config.get("num_rounds")
        epoch = latest.get("epoch")
        institution_id = config.get("institution_id") or _nested_get(latest, "metrics", "institution_id")
        institutions = config.get("institutions") if isinstance(config.get("institutions"), list) else []
        institution_ids = [
            item.get("institution_id")
            for item in institutions
            if isinstance(item, dict) and item.get("institution_id")
        ]

        return {
            "name": display_name,
            "path": str(run_dir),
            "stage": stage,
            "status": self._status(run_dir, updated_at, epoch, total_epochs, run_state.get("status")),
            "institution_id": institution_id,
            "institution_ids": institution_ids,
            "experiment_name": config.get("experiment_name") or run_state.get("experiment_name") or run_dir.parent.name,
            "run_id": run_state.get("run_id") or (run_dir.name if run_dir.parent == self.config.experiments_dir else run_dir.name),
            "epoch": epoch,
            "total_epochs": total_epochs,
            "proximal_mu": config.get("proximal_mu"),
            "num_rounds": config.get("num_rounds"),
            "run_type": self._run_type(stage=stage, experiment_name=config.get("experiment_name") or run_dir.parent.name),
            "updated_at": updated_at,
            "latest_metrics": latest,
            "artifacts": {
                "config": (run_dir / "config.json").exists(),
                "metrics": (run_dir / "metrics.jsonl").exists(),
                "profile": (run_dir / "profile.jsonl").exists(),
                "log": (run_dir / "train.log").exists(),
                "model": (run_dir / "model.pt").exists(),
                "run_state": (run_dir / "run_state.json").exists(),
                "loss_plot": (run_dir / "loss_plot.svg").exists(),
                "pr_auc_plot": (run_dir / "pr_auc_plot.svg").exists(),
                "evaluation": (run_dir / "evaluation.json").exists(),
            },
        }

    def _display_name(self, run_dir: Path) -> str:
        relative_path = run_dir.relative_to(self.config.experiments_dir)
        return relative_path.as_posix()

    def _status(
        self,
        run_dir: Path,
        updated_at: str | None,
        epoch: Any,
        total_epochs: Any,
        run_state: Any,
    ) -> str:
        if run_state == "completed":
            return "completed"

        latest_mtime = self._latest_artifact_mtime(run_dir)
        now = datetime.now(timezone.utc).timestamp()
        is_recent = latest_mtime is not None and now - latest_mtime <= self.config.active_timeout_seconds
        if run_state == "running":
            return "running" if is_recent else "stale"

        if (run_dir / "model.pt").exists():
            return "completed"

        if is_recent:
            return "running"
        if updated_at is None:
            return "pending"
        if _as_int(epoch) is not None and _as_int(total_epochs) is not None and _as_int(epoch) < _as_int(total_epochs):
            return "stale"
        return "unknown"

    @staticmethod
    def _read_metrics(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _read_log_tail(path: Path, max_lines: int = 160) -> list[str]:
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]

    def _latest_artifact_timestamp(self, run_dir: Path) -> str | None:
        latest_mtime = self._latest_artifact_mtime(run_dir)
        if latest_mtime is None:
            return None
        return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()

    @staticmethod
    def _latest_artifact_mtime(run_dir: Path) -> float | None:
        mtimes = [
            path.stat().st_mtime
            for name in (
                "metrics.jsonl",
                "profile.jsonl",
                "train.log",
                "config.json",
                "model.pt",
                "run_state.json",
                "loss_plot.svg",
                "pr_auc_plot.svg",
                "evaluation.json",
            )
            if (path := run_dir / name).exists()
        ]
        return max(mtimes) if mtimes else None

    @staticmethod
    def _run_type(stage: str, experiment_name: Any) -> str | None:
        if stage != "federated_training":
            return None
        name = str(experiment_name or "")
        if "global" in name:
            return "global"
        if "banks_" in name:
            return "exclusive"
        return "federated"


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
