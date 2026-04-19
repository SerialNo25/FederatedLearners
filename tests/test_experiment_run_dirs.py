import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from domain.logging.experiment_logger import allocate_experiment_run_dir
from domain.evaluation_service import EvaluationCheckpointLoader


class ExperimentRunDirectoryTests(unittest.TestCase):
    def test_allocates_incrementing_run_directories(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            first = allocate_experiment_run_dir(output_dir, "local_bank_1_tabnet")
            second = allocate_experiment_run_dir(output_dir, "local_bank_1_tabnet")

            self.assertEqual(first.relative_to(output_dir).as_posix(), "local_bank_1_tabnet/run_001")
            self.assertEqual(second.relative_to(output_dir).as_posix(), "local_bank_1_tabnet/run_002")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_skips_existing_numbered_runs(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            existing = output_dir / "federated_global" / "run_004"
            existing.mkdir(parents=True)
            (existing / "config.json").write_text(json.dumps({"stage": "federated_training"}), encoding="utf-8")

            next_run = allocate_experiment_run_dir(output_dir, "federated_global")

            self.assertEqual(next_run.relative_to(output_dir).as_posix(), "federated_global/run_005")

    def test_legacy_checkpoint_path_resolves_to_latest_numbered_run(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = output_dir / "local_bank_1_tabnet" / "run_001"
            second = output_dir / "local_bank_1_tabnet" / "run_002"
            first.mkdir(parents=True)
            second.mkdir()
            (first / "model.pt").write_text("first", encoding="utf-8")
            (second / "model.pt").write_text("second", encoding="utf-8")

            resolved = EvaluationCheckpointLoader._resolve_checkpoint_path(
                output_dir / "local_bank_1_tabnet" / "model.pt"
            )

            self.assertEqual(resolved, second / "model.pt")


if __name__ == "__main__":
    unittest.main()
