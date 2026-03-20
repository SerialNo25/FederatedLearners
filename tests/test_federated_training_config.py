from pathlib import Path
import unittest

from domain.models.model_config import TabNetModelConfig
from stages.federated_training.config import FederatedTrainingConfig


def _institution_dict(institution_id: str) -> dict:
    return {
        "institution_id": institution_id,
        "dataset_path": f"configs/sample_data/{institution_id}.csv",
        "local_epochs": 5,
        "learning_rate": 0.01,
        "model": {"model_type": "tabnet"},
    }


def _base_fed_dict(**overrides) -> dict:
    return {
        "num_rounds": 1,
        "proximal_mu": 0.0,
        "model": {
            "model_type": "tabnet",
            "decision_dim": 32,
            "attention_dim": 24,
            "steps": 4,
            "relaxation_factor": 1.2,
            "sparsity_weight": 0.001,
        },
        "institutions": [_institution_dict("bank_1")],
        **overrides,
    }


class FederatedTrainingConfigTests(unittest.TestCase):
    def test_tabnet_model_config_is_valid(self):
        config = FederatedTrainingConfig.model_validate(_base_fed_dict())
        self.assertIsInstance(config.model, TabNetModelConfig)
        self.assertEqual(config.model.model_type, "tabnet")
        self.assertEqual(config.model.decision_dim, 32)

    def test_institutions_are_loaded_with_training_params(self):
        config = FederatedTrainingConfig.model_validate(_base_fed_dict())
        self.assertEqual(config.institutions[0].institution_id, "bank_1")
        self.assertEqual(
            config.institutions[0].dataset_path,
            Path("configs/sample_data/bank_1.csv"),
        )
        self.assertEqual(config.institutions[0].local_epochs, 5)

    def test_tabnet_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            FederatedTrainingConfig.model_validate(
                _base_fed_dict(model={"model_type": "tabnet", "steps": 0})
            )

    def test_duplicate_institution_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            FederatedTrainingConfig.model_validate(
                _base_fed_dict(
                    institutions=[
                        _institution_dict("bank_1"),
                        _institution_dict("bank_1"),
                    ]
                )
            )

    def test_num_rounds_must_be_positive(self):
        with self.assertRaises(ValueError):
            FederatedTrainingConfig.model_validate(_base_fed_dict(num_rounds=0))


if __name__ == "__main__":
    unittest.main()
