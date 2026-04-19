import json
import tempfile
import unittest
from pathlib import Path

from domain.dataset.schema import FEATURE_COLUMNS, TARGET_COLUMN
from scripts.run_random_guessing_evaluation import run_random_guessing_evaluation


class RandomGuessingEvaluationScriptTests(unittest.TestCase):
    def test_random_guessing_evaluation_defaults_to_dataset_imbalance(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            dataset_path = self._write_dataset(tmp_dir)

            output_dir = run_random_guessing_evaluation(
                dataset_path=dataset_path,
                output_dir=tmp_dir / "outputs",
                experiment_name="random_baseline",
                positive_probability=None,
                seed=7,
                classification_threshold=0.5,
            )

            payload = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["model_type"], "random_guessing")
            self.assertEqual(payload["positive_probability_source"], "dataset_prevalence")
            self.assertAlmostEqual(payload["dataset_positive_rate"], 0.25)
            self.assertAlmostEqual(payload["positive_probability"], 0.25)
            self.assertEqual(payload["seed"], 7)
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "train.log").exists())
            self.assertTrue((output_dir / "run_state.json").exists())

    def test_random_guessing_evaluation_accepts_explicit_positive_probability(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            dataset_path = self._write_dataset(tmp_dir)

            output_dir = run_random_guessing_evaluation(
                dataset_path=dataset_path,
                output_dir=tmp_dir / "outputs",
                experiment_name="random_baseline",
                positive_probability=0.75,
                seed=11,
                classification_threshold=0.5,
            )

            payload = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["positive_probability_source"], "explicit")
            self.assertAlmostEqual(payload["positive_probability"], 0.75)
            self.assertEqual(
                set(payload["metrics"]),
                {"loss", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr_at_95_recall"},
            )

    @staticmethod
    def _write_dataset(tmp_dir: Path) -> Path:
        dataset_path = tmp_dir / "dataset.csv"
        rows = [
            [10.0, 2.3, -0.5, 0.2, 0.7, -0.7, 0.0, 1.0, 0, 0, 1, 1, 0.25, 0.1, 1, 0, 0, 0, 0, 0, 0, 0],
            [120.0, 4.8, 1.4, 0.9, -0.5, -0.8, 0.4, -0.9, 0, 0, 1, 0, 0.55, 0.3, 0, 1, 0, 0, 0, 0, 0, 1],
            [15.0, 2.7, -0.4, 0.3, -0.9, 0.3, -0.4, -0.9, 1, 1, 0, 1, 0.35, 0.2, 0, 0, 1, 0, 0, 0, 0, 0],
            [250.0, 5.5, 2.1, 0.98, -0.3, 0.95, -0.8, 0.6, 1, 1, 1, 0, 0.75, 0.5, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
        dataset_path.write_text(
            ",".join([*FEATURE_COLUMNS, TARGET_COLUMN])
            + "\n"
            + "\n".join(",".join(str(value) for value in row) for row in rows)
            + "\n",
            encoding="utf-8",
        )
        return dataset_path


if __name__ == "__main__":
    unittest.main()
