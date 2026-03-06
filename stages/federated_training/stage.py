"""Stage orchestration for three-institution federated training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.data.dataset_loader import InstitutionDataset, load_institution_dataset
from core.federated.fedavg import InstitutionUpdate, run_federated_round
from core.metrics.evaluation import InstitutionMetrics, evaluate_institution
from core.models.basic_model import LogisticRegressionModel
from core.training.trainer import TrainingConfig
from stages.federated_training.config import FederatedTrainingConfig


class FederatedTrainingStage:
    def __init__(self, config: FederatedTrainingConfig) -> None:
        self.config = config

    def execute(self) -> Path:
        datasets = self._load_datasets()
        experiment_dir = self.config.output_dir / self.config.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        model = LogisticRegressionModel.initialize(len(datasets[0].features[0]))
        training_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            local_epochs=self.config.local_epochs,
            proximal_mu=self.config.proximal_mu,
        )

        with (experiment_dir / "train.log").open("w", encoding="utf-8") as log_fp, (
            experiment_dir / "metrics.jsonl"
        ).open("w", encoding="utf-8") as metrics_fp:
            log_fp.write(f"start_time={datetime.now(timezone.utc).isoformat()}\n")
            log_fp.write(f"config={json.dumps(self.config.to_dict(), indent=2)}\n")

            for round_index in range(1, self.config.num_rounds + 1):
                updates = run_federated_round(model, datasets, training_config)
                evaluations = [evaluate_institution(model, dataset) for dataset in datasets]
                self._write_round_metrics(metrics_fp, round_index, updates, evaluations)
                round_loss = sum(metric.loss for metric in evaluations) / len(evaluations)
                round_accuracy = sum(metric.accuracy for metric in evaluations) / len(evaluations)
                log_fp.write(
                    f"round={round_index} mean_loss={round_loss:.6f} mean_accuracy={round_accuracy:.6f}\n"
                )

        self._persist_artifacts(experiment_dir, model)
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
        metrics_fp,
        round_index: int,
        updates: list[InstitutionUpdate],
        evaluations: list[InstitutionMetrics],
    ) -> None:
        local_loss = {update.institution_id: update.local_loss for update in updates}
        eval_payload = {
            metric.institution_id: {"loss": metric.loss, "accuracy": metric.accuracy}
            for metric in evaluations
        }
        line = {
            "epoch": round_index,
            "train_loss": sum(local_loss.values()) / len(local_loss),
            "val_loss": sum(metric.loss for metric in evaluations) / len(evaluations),
            "metrics": {"local_loss": local_loss, "institution_evaluation": eval_payload},
            "learning_rate": None,
        }
        metrics_fp.write(json.dumps(line) + "\n")

    def _persist_artifacts(self, experiment_dir: Path, model: LogisticRegressionModel) -> None:
        (experiment_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        weights, bias = model.parameters()
        payload = {"weights": weights, "bias": bias}
        (experiment_dir / "model.pt").write_text(json.dumps(payload), encoding="utf-8")

