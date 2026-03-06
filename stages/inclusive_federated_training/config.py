"""Configuration schema for inclusive federated training stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstitutionConfig:
    institution_id: str
    dataset_path: Path


@dataclass(frozen=True)
class InclusiveFederatedTrainingConfig:
    experiment_name: str
    output_dir: Path
    institutions: list[InstitutionConfig]
    num_rounds: int
    local_epochs: int
    learning_rate: float
    proximal_mu: float

    @classmethod
    def from_dict(cls, payload: dict) -> "InclusiveFederatedTrainingConfig":
        institutions = [
            InstitutionConfig(
                institution_id=institution["institution_id"],
                dataset_path=Path(institution["dataset_path"]),
            )
            for institution in payload.get("institutions", [])
        ]
        config = cls(
            experiment_name=payload.get("experiment_name", "inclusive_federated_global_3_institutions"),
            output_dir=Path(payload.get("output_dir", "data/experiments")),
            institutions=institutions,
            num_rounds=int(payload.get("num_rounds", 5)),
            local_epochs=int(payload.get("local_epochs", 3)),
            learning_rate=float(payload.get("learning_rate", 0.05)),
            proximal_mu=float(payload.get("proximal_mu", 0.001)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.institutions) != 3:
            raise ValueError("Exactly 3 institutions must be configured")
        ids = [item.institution_id for item in self.institutions]
        if len(ids) != len(set(ids)):
            raise ValueError("Institution IDs must be unique")
        if self.num_rounds < 1 or self.local_epochs < 1:
            raise ValueError("num_rounds and local_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu must be >= 0")

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "institutions": [
                {"institution_id": item.institution_id, "dataset_path": str(item.dataset_path)}
                for item in self.institutions
            ],
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "proximal_mu": self.proximal_mu,
        }
