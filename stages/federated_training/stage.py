"""Stage orchestration for n-institution federated training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from domain.dataset.dataset_loader import InstitutionDataset, load_institution_dataset
from domain.federated.fedprox_orchestrator import (
    InstitutionNode,
    InstitutionUpdate,
    FedProxOrchestrator,
)
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.aggregation import weighted_mean
from domain.metrics.evaluation import InstitutionMetrics, evaluate_institution
from domain.training.trainer import TrainingConfig
from stages.federated_training.config import FederatedTrainingConfig


class FederatedTrainingStage:
    def __init__(
        self,
        config: FederatedTrainingConfig,
        experiment_logger: StageExperimentLogger,
        model_factory: Callable[[int], Any],
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.model_factory = model_factory

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
            self._write_round_metrics(round_index, updates, evaluations)
            self._log_round_institution_details(round_index, updates, evaluations)
            self._log_round_summary(round_index, evaluations)

    def _log_round_summary(
        self,
        round_index: int,
        evaluations: list[InstitutionMetrics],
    ) -> None:
        round_loss = sum(metric.loss for metric in evaluations) / len(evaluations)
        round_accuracy = sum(metric.accuracy for metric in evaluations) / len(evaluations)
        self.experiment_logger.info(
            f"round={round_index} mean_loss={round_loss:.6f} mean_accuracy={round_accuracy:.6f}"
        )

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

    def _write_round_metrics(
        self,
        round_index: int,
        updates: list[InstitutionUpdate],
        evaluations: list[InstitutionMetrics],
    ) -> None:
        local_loss = {update.institution_id: update.local_loss for update in updates}
        local_num_samples = {update.institution_id: update.num_samples for update in updates}
        local_parameter_delta_l2 = {
            update.institution_id: update.parameter_delta_l2 for update in updates
        }
        eval_payload = {
            metric.institution_id: {"loss": metric.loss, "accuracy": metric.accuracy}
            for metric in evaluations
        }
        line = {
            "epoch": round_index,
            "train_loss": weighted_mean(list(local_loss.values()), list(local_num_samples.values())),
            "val_loss": sum(metric.loss for metric in evaluations) / len(evaluations),
            "metrics": {
                "local_loss": local_loss,
                "local_num_samples": local_num_samples,
                "local_parameter_delta_l2": local_parameter_delta_l2,
                "institution_evaluation": eval_payload,
            },
            "learning_rate": None,
        }
        self.experiment_logger.write_metrics(step=f"round_{round_index}", values=line)

    def _log_round_institution_details(
        self,
        round_index: int,
        updates: list[InstitutionUpdate],
        evaluations: list[InstitutionMetrics],
    ) -> None:
        evaluation_by_institution = {
            evaluation.institution_id: evaluation for evaluation in evaluations
        }
        for update in updates:
            evaluation = evaluation_by_institution[update.institution_id]
            self.experiment_logger.info(
                "round=%s institution=%s local_loss=%.6f eval_loss=%.6f eval_accuracy=%.6f "
                "num_samples=%s parameter_delta_l2=%.6f"
                % (
                    round_index,
                    update.institution_id,
                    update.local_loss,
                    evaluation.loss,
                    evaluation.accuracy,
                    update.num_samples,
                    update.parameter_delta_l2,
                )
            )

    @staticmethod
    def _model_parameters(model: Any) -> dict[str, list[float]]:
        if hasattr(model, "federated_parameters"):
            return model.federated_parameters()
        return model.parameters()

    def _persist_artifacts(self, experiment_dir: Path, model: Any) -> None:
        (experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "model_type": self.config.model_type,
                "parameters": self._model_parameters(model),
            },
            experiment_dir / "model.pt",
        )
