import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from stages.federated_hyperparameter_optimization.stage import (
    FederatedHyperparameterOptimizationStage,
)


class FederatedHyperparameterOptimizationStageTests(unittest.TestCase):
    def test_prepare_storage_url_creates_relative_sqlite_parent(self):
        stage = FederatedHyperparameterOptimizationStage.__new__(
            FederatedHyperparameterOptimizationStage
        )

        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "nested" / "optuna.db"
            stage.config = SimpleNamespace(storage_url=f"sqlite:///{db_path}")

            self.assertEqual(stage._prepare_storage_url(), f"sqlite:///{db_path}")
            self.assertTrue(db_path.parent.exists())

    def test_read_final_metrics_returns_last_record(self):
        stage = FederatedHyperparameterOptimizationStage.__new__(
            FederatedHyperparameterOptimizationStage
        )

        with TemporaryDirectory() as temp_dir:
            trial_dir = Path(temp_dir)
            metrics_path = trial_dir / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"epoch": 1, "pr_auc": 0.2, "val_loss": 0.8}),
                        json.dumps({"epoch": 2, "pr_auc": 0.4, "val_loss": 0.6}),
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(stage._read_final_metrics(trial_dir)["epoch"], 2)

    def test_known_shape_mismatch_is_invalid_trial_error(self):
        stage = FederatedHyperparameterOptimizationStage.__new__(
            FederatedHyperparameterOptimizationStage
        )

        self.assertTrue(
            stage._is_invalid_trial_error(
                RuntimeError("input and weight.T shapes cannot be multiplied (1028x24 and 32x21)")
            )
        )


if __name__ == "__main__":
    unittest.main()
