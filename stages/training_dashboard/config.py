"""Configuration schema for the browser-based training dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class TrainingDashboardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["training_dashboard"] = "training_dashboard"
    experiments_dir: Path = Path("data/experiments")
    host: str = "127.0.0.1"
    port: int = 8765
    poll_interval_seconds: float = 2.0
    active_timeout_seconds: int = 120

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("host must not be empty")
        return value

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("poll_interval_seconds")
    @classmethod
    def _validate_poll_interval(cls, value: float) -> float:
        if value < 0.5:
            raise ValueError("poll_interval_seconds must be >= 0.5")
        return value

    @field_validator("active_timeout_seconds")
    @classmethod
    def _validate_active_timeout(cls, value: int) -> int:
        if value < 1:
            raise ValueError("active_timeout_seconds must be >= 1")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingDashboardConfig":
        return cls.model_validate(payload)
