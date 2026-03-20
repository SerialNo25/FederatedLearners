import unittest

from stages.evaluation.config import EvaluationConfig


class EvaluationConfigTests(unittest.TestCase):
    def test_defaults_applied(self):
        config = EvaluationConfig.from_dict(
            {
                "model_path": "data/experiments/example/model.pt",
                "dataset_path": "configs/sample_data/bank_1.csv",
            }
        )
        self.assertEqual(config.experiment_name, "evaluation")
        self.assertEqual(str(config.output_dir), "data/experiments")
        self.assertEqual(config.classification_threshold, 0.5)

    def test_invalid_classification_threshold_rejected(self):
        with self.assertRaises(ValueError):
            EvaluationConfig.from_dict(
                {
                    "model_path": "data/experiments/example/model.pt",
                    "dataset_path": "configs/sample_data/bank_1.csv",
                    "classification_threshold": 1.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
