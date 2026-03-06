"""Stage orchestration for three-institution inclusive federated training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from domain.dataset.dataset_loader import InstitutionDataset, load_institution_dataset
from domain.federated.fedprox_orchestrator import (
    InstitutionNode,
    InstitutionUpdate,
    ThreeInstitutionFedProxOrchestrator,
)
from domain.logging.experiment_logger import StageExperimentLogger
from domain.metrics.aggregation import weighted_mean
from domain.metrics.evaluation import InstitutionMetrics, evaluate_institution
from domain.models.device_selector import DeviceSelector
from domain.models.model_registry import MODEL_REGISTRY
from domain.training.trainer import TrainingConfig
from stages.inclusive_federated_training.config import InclusiveFederatedTrainingConfig


class InclusiveFederatedTrainingStage:
    def __init__(self, config: InclusiveFederatedTrainingConfig) -> None:
        self.config = config

    def execute(self) -> Path:
        datasets = self._load_datasets()
        experiment_dir = self.config.output_dir / self.config.experiment_name
        experiment_logger = StageExperimentLogger(
            experiment_dir=str(experiment_dir),
            stage_name="inclusive_federated_training",
        )

        model_config = self.config.to_dict()
        if self.config.model_type == "tabnet":
            selector = DeviceSelector()
            selected_device = selector.select_best_device()
            model_config["tabnet_device"] = selected_device
            experiment_logger.info(
                "tabnet_device_selection selected=%s available=%s"
                % (selected_device, ",".join(selector.available_devices()))
            )

        model_factory = MODEL_REGISTRY.get_factory(self.config.model_type, model_config)
        model = model_factory(len(datasets[0].features[0]))
        training_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            local_epochs=self.config.local_epochs,
            proximal_mu=self.config.proximal_mu,
        )
        institutions = [
            InstitutionNode(
                dataset=dataset,
                training_config=training_config,
                model_factory=model_factory,
            )
            for dataset in datasets
        ]
        orchestrator = ThreeInstitutionFedProxOrchestrator(
            institutions=institutions,
            initial_model=model,
            proximal_mu=self.config.proximal_mu,
        )

        experiment_logger.info(f"start_time={datetime.now(timezone.utc).isoformat()}")
        experiment_logger.info(f"config={json.dumps(self.config.to_dict(), indent=2)}")
        experiment_logger.info(
            "federated_topology="
            + json.dumps([institution.institution_id for institution in institutions])
        )

        for round_index in range(1, self.config.num_rounds + 1):
            updates = orchestrator.run_round(round_index)
            evaluations = [
                evaluate_institution(orchestrator.global_model, dataset) for dataset in datasets
            ]
            self._write_round_metrics(experiment_logger, round_index, updates, evaluations)
            self._log_round_institution_details(experiment_logger, round_index, updates, evaluations)
            round_loss = sum(metric.loss for metric in evaluations) / len(evaluations)
            round_accuracy = sum(metric.accuracy for metric in evaluations) / len(evaluations)
            experiment_logger.info(
                f"round={round_index} mean_loss={round_loss:.6f} mean_accuracy={round_accuracy:.6f}"
            )

        self._persist_artifacts(experiment_dir, orchestrator.global_model)
        return experiment_dir

    def _load_datasets(self) -> list[InstitutionDataset]:
        return [
            load_institution_dataset(
                institution_id=institution.institution_id,
                csv_path=institution.dataset_path,
            )
            for institution in self.config.institutions
        ]

    @staticmethod
    def _write_round_metrics(
        experiment_logger: StageExperimentLogger,
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
        experiment_logger.write_metrics(step=f"round_{round_index}", values=line)

    @staticmethod
    def _log_round_institution_details(
        experiment_logger: StageExperimentLogger,
        round_index: int,
        updates: list[InstitutionUpdate],
        evaluations: list[InstitutionMetrics],
    ) -> None:
        evaluation_by_institution = {
            evaluation.institution_id: evaluation for evaluation in evaluations
        }
        for update in updates:
            evaluation = evaluation_by_institution[update.institution_id]
            experiment_logger.info(
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

    def _persist_artifacts(self, experiment_dir: Path, model: Any) -> None:
        (experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (experiment_dir / "model.pt").write_text(
            json.dumps(model.parameters()),
            encoding="utf-8",
        )
