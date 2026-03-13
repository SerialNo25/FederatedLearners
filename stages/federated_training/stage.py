"""Stage orchestration for n-institution federated training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from domain.dataset.dataset_loader import InstitutionDataset, load_institution_dataset
from domain.federated.fedprox_orchestrator import (
    InstitutionNode,
    FedProxOrchestrator,
)
from domain.federated.model_artifact_writer import ModelArtifactWriter
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.evaluation import evaluate_institution
from domain.training.trainer import TrainingConfig
from stages.federated_training.config import FederatedTrainingConfig
from stages.federated_training.round_reporter import FederatedRoundReporter


class FederatedTrainingStage:
    def __init__(
        self,
        config: FederatedTrainingConfig,
        experiment_logger: StageExperimentLogger,
        model_factory: Callable[[int], Any],
        round_reporter: FederatedRoundReporter | None = None,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.model_factory = model_factory
        self.round_reporter = round_reporter or FederatedRoundReporter()

    def execute(self) -> Path:
        datasets = self._load_datasets()
        self._assert_dataset_invariants(datasets)
        experiment_dir = self.config.output_dir / self.config.experiment_name
        orchestrator, institutions = self._build_orchestrator(datasets)
        self._log_experiment_start(institutions)
        self._run_training_rounds(orchestrator, datasets)
        self._persist_artifacts(experiment_dir, orchestrator.global_model)
        return experiment_dir

    def _build_orchestrator(
        self,
        datasets: list[InstitutionDataset],
    ) -> tuple[FedProxOrchestrator, list[InstitutionNode]]:
        model = self.model_factory(len(datasets[0].features[0]))
        model_device = getattr(model, "device", None)
        if model_device is not None:
            self.experiment_logger.info(f"tabnet_device_selection selected={model_device}")
        training_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            local_epochs=self.config.local_epochs,
            proximal_mu=self.config.proximal_mu,
        )
        institutions = [
            InstitutionNode(
                dataset=dataset,
                training_config=training_config,
                model_factory=self.model_factory,
            )
            for dataset in datasets
        ]
        return (
            FedProxOrchestrator(
                institutions=institutions,
                initial_model=model,
            ),
            institutions,
        )

    def _log_experiment_start(
        self,
        institutions: list[InstitutionNode],
    ) -> None:
        self.experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        self.experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        self.experiment_logger.info(
            "federated_topology="
            + json.dumps([institution.institution_id for institution in institutions])
        )

    def _run_training_rounds(
        self,
        orchestrator: FedProxOrchestrator,
        datasets: list[InstitutionDataset],
    ) -> None:
        for round_index in range(1, self.config.num_rounds + 1):
            updates = orchestrator.run_round(round_index)
            evaluations = [
                evaluate_institution(orchestrator.global_model, dataset) for dataset in datasets
            ]
            round_report = self.round_reporter.build_report(
                round_index=round_index,
                updates=updates,
                evaluations=evaluations,
                learning_rate=self.config.learning_rate,
            )
            self.experiment_logger.write_metrics(
                step=f"round_{round_index}",
                values=round_report.metrics_payload,
            )
            for line in round_report.institution_lines:
                self.experiment_logger.info(line)
            self.experiment_logger.info(round_report.summary_line)

    def _load_datasets(self) -> list[InstitutionDataset]:
        return [
            load_institution_dataset(
                institution_id=institution.institution_id,
                csv_path=institution.dataset_path,
            )
            for institution in self.config.institutions
        ]

    def _assert_dataset_invariants(self, datasets: list[InstitutionDataset]) -> None:
        if not datasets:
            raise RuntimeError("Federated training requires at least one institution dataset")

        for dataset in datasets:
            if not dataset.features or not dataset.labels:
                raise RuntimeError(
                    f"Institution dataset '{dataset.institution_id}' is empty; at least one row is required"
                )

    def _persist_artifacts(self, experiment_dir: Path, model: Any) -> None:
        (experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        ModelArtifactWriter.write_model_checkpoint(
            checkpoint_path=experiment_dir / "model.pt",
            model_type=self.config.model_type,
            model=model,
        )
