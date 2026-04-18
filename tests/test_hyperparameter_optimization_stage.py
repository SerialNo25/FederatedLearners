import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from stages.hyperparameter_optimization.stage import HyperparameterOptimizationStage


class HyperparameterOptimizationStageTests(unittest.TestCase):
    def test_known_shape_mismatch_is_invalid_trial_error(self):
        stage = HyperparameterOptimizationStage.__new__(HyperparameterOptimizationStage)

        self.assertTrue(
            stage._is_invalid_trial_error(
                RuntimeError("input and weight.T shapes cannot be multiplied (1028x24 and 32x21)")
            )
        )

    def test_unexpected_runtime_error_is_not_invalid_trial_error(self):
        stage = HyperparameterOptimizationStage.__new__(HyperparameterOptimizationStage)

        self.assertFalse(stage._is_invalid_trial_error(RuntimeError("out of memory")))

    def test_prepare_storage_url_creates_relative_sqlite_parent(self):
        stage = HyperparameterOptimizationStage.__new__(HyperparameterOptimizationStage)

        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "nested" / "optuna.db"
            stage.config = SimpleNamespace(storage_url=f"sqlite:///{db_path}")

            self.assertEqual(stage._prepare_storage_url(), f"sqlite:///{db_path}")
            self.assertTrue(db_path.parent.exists())


if __name__ == "__main__":
    unittest.main()
