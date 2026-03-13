"""Round-level reporting helpers for federated training orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from domain.federated.fedprox_orchestrator import InstitutionUpdate
from domain.metrics.aggregation import weighted_mean
from domain.metrics.evaluation import InstitutionMetrics


@dataclass(frozen=True)
class FederatedRoundReport:
    metrics_payload: dict[str, object]
    summary_line: str
    institution_lines: list[str]


class FederatedRoundReporter:
    """Builds machine-readable and textual reports for one federated round."""

    def build_report(
        self,
        round_index: int,
        updates: list[InstitutionUpdate],
        evaluations: list[InstitutionMetrics],
        learning_rate: float,
    ) -> FederatedRoundReport:
        local_loss = {update.institution_id: update.local_loss for update in updates}
        local_num_samples = {update.institution_id: update.num_samples for update in updates}
        local_parameter_delta_l2 = {
            update.institution_id: update.parameter_delta_l2 for update in updates
        }
        eval_payload = {
            metric.institution_id: {"loss": metric.loss, "accuracy": metric.accuracy}
            for metric in evaluations
        }
        metrics_payload = {
            "epoch": round_index,
            "train_loss": weighted_mean(list(local_loss.values()), list(local_num_samples.values())),
            "val_loss": sum(metric.loss for metric in evaluations) / len(evaluations),
            "metrics": {
                "local_loss": local_loss,
                "local_num_samples": local_num_samples,
                "local_parameter_delta_l2": local_parameter_delta_l2,
                "institution_evaluation": eval_payload,
            },
            "learning_rate": learning_rate,
        }

        summary_line = self._summary_line(round_index=round_index, evaluations=evaluations)
        institution_lines = self._institution_lines(
            round_index=round_index,
            updates=updates,
            evaluations=evaluations,
        )
        return FederatedRoundReport(
            metrics_payload=metrics_payload,
            summary_line=summary_line,
            institution_lines=institution_lines,
        )

    @staticmethod
    def _summary_line(round_index: int, evaluations: list[InstitutionMetrics]) -> str:
        round_loss = sum(metric.loss for metric in evaluations) / len(evaluations)
        round_accuracy = sum(metric.accuracy for metric in evaluations) / len(evaluations)
        return f"round={round_index} mean_loss={round_loss:.6f} mean_accuracy={round_accuracy:.6f}"

    @staticmethod
    def _institution_lines(
        round_index: int,
        updates: list[InstitutionUpdate],
        evaluations: list[InstitutionMetrics],
    ) -> list[str]:
        evaluation_by_institution = {
            evaluation.institution_id: evaluation for evaluation in evaluations
        }
        return [
            "round=%s institution=%s local_loss=%.6f eval_loss=%.6f eval_accuracy=%.6f "
            "num_samples=%s parameter_delta_l2=%.6f"
            % (
                round_index,
                update.institution_id,
                update.local_loss,
                evaluation_by_institution[update.institution_id].loss,
                evaluation_by_institution[update.institution_id].accuracy,
                update.num_samples,
                update.parameter_delta_l2,
            )
            for update in updates
        ]
