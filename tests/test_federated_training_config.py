from pathlib import Path
import unittest

from stages.federated_training.config import FederatedTrainingConfig, TabNetModelConfig


class FederatedTrainingConfigTests(unittest.TestCase):
    def test_tabnet_model_config_is_nested_under_model_section(self):
        config = FederatedTrainingConfig.from_dict(
            {
                "experiment_name": "exp",
                "output_dir": "data/experiments",
                "num_institutions": 1,
                "num_rounds": 1,
                "local_epochs": 1,
                "learning_rate": 0.01,
                "proximal_mu": 0.0,
                "model": {
                    "model_type": "tabnet",
                    "decision_dim": 32,
                    "attention_dim": 24,
                    "steps": 4,
                    "relaxation_factor": 1.2,
                    "sparsity_weight": 0.001,
                },
                "institutions": [
                    {
                        "institution_id": "bank_1",
                        "dataset_path": "configs/sample_data/bank_1.csv",
                    }
                ],
            }
        )

        self.assertIsInstance(config.model, TabNetModelConfig)
        self.assertEqual(config.model.model_type, "tabnet")
        self.assertEqual(config.model.decision_dim, 32)
        self.assertEqual(config.institutions[0].dataset_path, Path("configs/sample_data/bank_1.csv"))

    def test_tabnet_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            FederatedTrainingConfig.from_dict(
                {
                    "experiment_name": "exp",
                    "output_dir": "data/experiments",
                    "num_institutions": 1,
                    "num_rounds": 1,
                    "local_epochs": 1,
                    "learning_rate": 0.01,
                    "proximal_mu": 0.0,
                    "model": {
                        "model_type": "tabnet",
                        "steps": 0,
                    },
                    "institutions": [
                        {
                            "institution_id": "bank_1",
                            "dataset_path": "configs/sample_data/bank_1.csv",
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
