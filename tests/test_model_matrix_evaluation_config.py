import unittest

from stages.model_matrix_evaluation.config import ModelMatrixConfig


class ModelMatrixEvaluationConfigTests(unittest.TestCase):
    def test_checkpoint_path_uses_base_path_and_run_number(self):
        config = ModelMatrixConfig.from_dict(
            {
                "datasets": [{"dataset_id": "bank_1", "path": "data/bank_1.csv"}],
                "local_models": [
                    {
                        "model_id": "local_bank_1",
                        "base_path": "data/experiments/local_bank_1_tabnet",
                        "run_number": 28,
                    }
                ],
                "global_federated_model": {
                    "model_id": "federated_global",
                    "base_path": "data/experiments/federated_global",
                    "run_number": 16,
                },
                "exclusive_federated_models": [],
                "exclusive_ensembles": [],
                "inclusive_ensembles": [
                    {
                        "model_id": "ensemble_inclusive_bank_1",
                        "local_model_id": "local_bank_1",
                        "federated_model_id": "federated_global",
                    }
                ],
            }
        )

        self.assertEqual(
            str(config.local_models[0].checkpoint_path),
            "data/experiments/local_bank_1_tabnet/run_028/model.pt",
        )
        self.assertEqual(config.ensemble_weight, 0.5)
        self.assertEqual(config.classification_threshold, 0.5)

    def test_unknown_ensemble_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelMatrixConfig.from_dict(
                {
                    "datasets": [{"dataset_id": "bank_1", "path": "data/bank_1.csv"}],
                    "local_models": [],
                    "global_federated_model": {
                        "model_id": "federated_global",
                        "base_path": "data/experiments/federated_global",
                        "run_number": 1,
                    },
                    "exclusive_federated_models": [],
                    "exclusive_ensembles": [],
                    "inclusive_ensembles": [
                        {
                            "model_id": "ensemble_inclusive_bank_1",
                            "local_model_id": "missing_local",
                            "federated_model_id": "federated_global",
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
