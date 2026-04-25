"""Configuration schema for the ensemble validation split stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from domain.dataset.dataset_loader import DEFAULT_LOCAL_TRAINING_VALIDATION_FRACTION


class InstitutionEnsembleValidationSplitConfig(BaseModel):
    institution_id: str
    dataset_path: Path


class EnsembleValidationSplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["ensemble_validation_split"] = "ensemble_validation_split"
    output_dir: Path = Path("data/ensemble_validation_splits")
    validation_fraction: float = DEFAULT_LOCAL_TRAINING_VALIDATION_FRACTION
    seed: int = 42
    institutions: list[InstitutionEnsembleValidationSplitConfig]

    @field_validator("validation_fraction")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1 (exclusive)")
        return value

    @field_validator("institutions")
    @classmethod
    def _validate_institutions(
        cls,
        value: list[InstitutionEnsembleValidationSplitConfig],
    ) -> list[InstitutionEnsembleValidationSplitConfig]:
        if not value:
            raise ValueError("At least one institution must be configured")
        ids = [institution.institution_id for institution in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")
        return value

    @classmethod
    def from_dict(cls, payload: dict) -> "EnsembleValidationSplitConfig":
        return cls.model_validate(payload)
