"""Stage orchestration for n-institution federated training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from domain.dataset.dataset_loader import InstitutionDataset, load_institution_dataset
from domain.federated.fedprox_orchestrator import (
    InstitutionNode,
    FedProxOrchestrator,
)
from domain.federated.model_artifact_writer import ModelArtifactWriter
from domain.logging.experiment_logger import StageExperimentLogger
from domain.logging.loss_plot_writer import LossPlotWriter
from domain.logging.pr_auc_plot_writer import PRAUCPlotWriter
from domain.metrics.evaluation import evaluate_institution
from domain.models.federated_model_protocol import FederatedModelProtocol
from domain.models.model_registry import ModelFactoryProtocol
from domain.training.trainer import TrainingConfig
from stages.federated_training.config import FederatedTrainingConfig
from stages.federated_training.round_reporter import (
    FederatedRoundReport,
    FederatedRoundReporter,
)
from stages.stage import Stage


class FederatedTrainingStage(Stage):
    def __init__(
        self,
        config: FederatedTrainingConfig,
        experiment_logger: StageExperimentLogger,
        experiment_dir: Path,
        model_factory: ModelFactoryProtocol,
        round_reporter: FederatedRoundReporter | None = None,
    ) -> None:
        self.config = config
        self.experiment_logger = experiment_logger
        self.experiment_dir = experiment_dir
        self.model_factory = model_factory
        self.round_reporter = round_reporter or FederatedRoundReporter()

    def execute(self) -> Path:
        self._write_run_state("running")
        datasets = self._load_datasets()
        self._assert_dataset_invariants(datasets)

        orchestrator, institutions = self._build_orchestrator(datasets)

        self._log_experiment_start(institutions)
        round_reports = self._run_training_rounds(orchestrator, datasets)
        self._persist_artifacts(self.experiment_dir, orchestrator.global_model)
        self._write_training_plots(round_reports)
        self._write_run_state("completed")
        return self.experiment_dir

    def _build_orchestrator(
        self,
        datasets: list[InstitutionDataset],
    ) -> tuple[FedProxOrchestrator, list[InstitutionNode]]:
        model = self.model_factory(len(datasets[0].features[0]))
        model_device = getattr(model, "device", None)
        if model_device is not None:
            self.experiment_logger.info(f"tabnet_device_selection selected={model_device}")

        institutions = [
            InstitutionNode(
                dataset=dataset,
                training_config=TrainingConfig(
                    learning_rate=inst_config.learning_rate,
                    local_epochs=inst_config.local_epochs,
                    proximal_mu=self.config.proximal_mu,
                    fraud_weight=inst_config.fraud_weight,
                    batch_size=inst_config.batch_size,
                    seed=inst_config.seed,
                ),
                model_factory=self.model_factory,
            )
            for inst_config, dataset in zip(self.config.institutions, datasets)
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
    ) -> list[FederatedRoundReport]:
        round_reports: list[FederatedRoundReport] = []
        for round_index in tqdm(range(1, self.config.num_rounds + 1), "federated rounds"):
            updates = orchestrator.run_round()
            evaluations = [
                evaluate_institution(
                    orchestrator.global_model,
                    dataset,
                    pos_weight=inst_config.fraud_weight,
                    threshold=inst_config.classification_threshold,
                )
                for inst_config, dataset in zip(self.config.institutions, datasets)
            ]
            round_report = self.round_reporter.build_report(
                round_index=round_index,
                updates=updates,
                evaluations=evaluations,
            )
            self.experiment_logger.write_metrics(
                step=f"round_{round_index}",
                values=round_report.metrics_payload,
            )
            for line in round_report.institution_lines:
                self.experiment_logger.info(line)
            self.experiment_logger.info(round_report.summary_line)
            round_reports.append(round_report)
        return round_reports

    def _load_datasets(self) -> list[InstitutionDataset]:
        return [
            load_institution_dataset(
                institution_id=inst_config.institution_id,
                csv_path=inst_config.dataset_path,
            )
            for inst_config in self.config.institutions
        ]

    def _assert_dataset_invariants(self, datasets: list[InstitutionDataset]) -> None:
        if not datasets:
            raise RuntimeError("Federated training requires at least one institution dataset")

        for dataset in datasets:
            if not dataset.features or not dataset.labels:
                raise RuntimeError(
                    f"Institution dataset '{dataset.institution_id}' is empty; at least one row is required"
                )

    def _persist_artifacts(self, experiment_dir: Path, model: FederatedModelProtocol) -> None:
        (experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        ModelArtifactWriter.write_model_checkpoint(
            checkpoint_path=experiment_dir / "model.pt",
            model_type=self.config.model.model_type,
            model=model,
            model_config=self.config.model.model_dump(mode="python"),
        )

    def _write_training_plots(self, round_reports: list[FederatedRoundReport]) -> None:
        if not round_reports:
            return

        rounds = [int(report.metrics_payload["epoch"]) for report in round_reports]
        train_losses = [float(report.metrics_payload["train_loss"]) for report in round_reports]
        val_losses = [float(report.metrics_payload["val_loss"]) for report in round_reports]
        pr_auc_values = [float(report.metrics_payload["pr_auc"]) for report in round_reports]

        LossPlotWriter.write(
            output_path=self.experiment_dir / "loss_plot.svg",
            rounds=rounds,
            train_losses=train_losses,
            val_losses=val_losses,
        )
        PRAUCPlotWriter.write(
            output_path=self.experiment_dir / "pr_auc_plot.svg",
            rounds=rounds,
            pr_auc_values=pr_auc_values,
        )

    def _write_run_state(self, status: str) -> None:
        (self.experiment_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "stage": "federated_training",
                    "status": status,
                    "experiment_name": self.config.experiment_name,
                    "run_id": self.experiment_dir.name,
                    "run_dir": str(self.experiment_dir),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
