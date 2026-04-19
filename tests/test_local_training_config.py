import unittest

from stages.local_training.config import LocalTrainingConfig


def _base_dict(**overrides) -> dict:
    return {
        "institution_id": "bank_1",
        "dataset_path": "data/train_test_splits/bank_1_train.csv",
        "local_epochs": 2,
        "learning_rate": 0.01,
        "model": {"model_type": "tabnet"},
        **overrides,
    }


class LocalTrainingConfigTests(unittest.TestCase):
    def test_loads_institution_directly(self):
        config = LocalTrainingConfig.from_dict(_base_dict())
        self.assertEqual(config.institution_id, "bank_1")

    def test_defaults_applied(self):
        config = LocalTrainingConfig.from_dict(_base_dict())
        self.assertEqual(config.fraud_weight, 100.0)
        self.assertEqual(config.batch_size, 4096)
        self.assertEqual(config.classification_threshold, 0.5)
        self.assertEqual(config.seed, 42)

    def test_invalid_learning_rate_rejected(self):
        with self.assertRaises(ValueError):
            LocalTrainingConfig.from_dict(_base_dict(learning_rate=0.0))

    def test_invalid_classification_threshold_rejected(self):
        with self.assertRaises(ValueError):
            LocalTrainingConfig.from_dict(_base_dict(classification_threshold=1.0))

    def test_invalid_fraud_weight_rejected(self):
        with self.assertRaises(ValueError):
            LocalTrainingConfig.from_dict(_base_dict(fraud_weight=0.0))


if __name__ == "__main__":
    unittest.main()
