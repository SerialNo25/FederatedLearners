from pathlib import Path
import unittest

from domain.models.model_config import TabNetModelConfig
from stages.federated_training.config import FederatedTrainingConfig


def _model_dict() -> dict:
    return {"model_type": "tabnet"}


def _institution_dict(institution_id: str) -> dict:
    return {
        "institution_id": institution_id,
        "dataset_path": f"data/train_test_splits/{institution_id}.csv",
        "local_epochs": 5,
        "learning_rate": 0.01,
        "model": _model_dict(),
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
            Path("data/train_test_splits/bank_1.csv"),
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

    def test_local_training_overrides_are_validated_and_resolved(self):
        config = FederatedTrainingConfig.model_validate(
            _base_fed_dict(
                local_training_overrides={
                    "bank_1": {
                        "local_epochs": 2,
                        "learning_rate": 0.001,
                        "fraud_weight": 20.0,
                        "batch_size": 128,
                        "classification_threshold": 0.4,
                    },
                }
            )
        )

        institution = config.institutions[0]
        self.assertEqual(config.local_epochs_for(institution), 2)
        self.assertEqual(config.learning_rate_for(institution), 0.001)
        self.assertEqual(config.fraud_weight_for(institution), 20.0)
        self.assertEqual(config.batch_size_for(institution), 128)
        self.assertEqual(config.classification_threshold_for(institution), 0.4)

    def test_invalid_local_training_override_is_rejected(self):
        with self.assertRaises(ValueError):
            FederatedTrainingConfig.model_validate(
                _base_fed_dict(local_training_overrides={"bank_1": {"learning_rate": 0.0}})
            )

    def test_unknown_local_training_override_institution_is_rejected(self):
        with self.assertRaises(ValueError):
            FederatedTrainingConfig.model_validate(
                _base_fed_dict(local_training_overrides={"bank_2": {"learning_rate": 0.001}})
            )


if __name__ == "__main__":
    unittest.main()
