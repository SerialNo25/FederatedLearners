import unittest

from stages.hyperparameter_optimization.config import HyperparameterOptimizationConfig


def _base_dict(**overrides) -> dict:
    return {
        "institution_id": "bank_1",
        "dataset_path": "data/train_test_splits/bank_1_train.csv",
        "model": {"model_type": "tabnet"},
        "search_space": {
            "training": {
                "learning_rate": {"low": 0.001, "high": 0.03, "log": True},
                "local_epochs": {"low": 2, "high": 5},
                "batch_size": {"choices": [128, 256]},
            },
            "tabnet": {
                "decision_dim": {"choices": [8, 16]},
                "steps": {"low": 2, "high": 4},
            },
        },
        **overrides,
    }


class HyperparameterOptimizationConfigTests(unittest.TestCase):
    def test_defaults_apply_for_bank_config(self):
        config = HyperparameterOptimizationConfig.from_dict(_base_dict())

        self.assertEqual(config.stage, "hyperparameter_optimization")
        self.assertEqual(config.institution_id, "bank_1")
        self.assertEqual(config.objective_metric, "val_pr_auc")
        self.assertEqual(config.direction, "maximize")
        self.assertEqual(config.n_trials, 10)
        self.assertIsNone(config.study_name)
        self.assertIsNone(config.storage_url)
        self.assertTrue(config.load_if_exists)

    def test_rejects_invalid_trial_count(self):
        with self.assertRaises(ValueError):
            HyperparameterOptimizationConfig.from_dict(_base_dict(n_trials=0))

    def test_rejects_empty_storage_url(self):
        with self.assertRaises(ValueError):
            HyperparameterOptimizationConfig.from_dict(_base_dict(storage_url=" "))

    def test_rejects_invalid_float_search_range(self):
        payload = _base_dict(
            search_space={
                "training": {
                    "learning_rate": {"low": 0.03, "high": 0.001},
                },
            }
        )

        with self.assertRaises(ValueError):
            HyperparameterOptimizationConfig.from_dict(payload)

    def test_rejects_log_search_with_non_positive_bound(self):
        payload = _base_dict(
            search_space={
                "training": {
                    "learning_rate": {"low": 0.0, "high": 0.001, "log": True},
                },
            }
        )

        with self.assertRaises(ValueError):
            HyperparameterOptimizationConfig.from_dict(payload)

    def test_rejects_float_log_search_with_step(self):
        payload = _base_dict(
            search_space={
                "training": {
                    "learning_rate": {
                        "low": 0.001,
                        "high": 0.03,
                        "log": True,
                        "step": 0.001,
                    },
                },
            }
        )

        with self.assertRaises(ValueError):
            HyperparameterOptimizationConfig.from_dict(payload)

    def test_val_loss_must_minimize(self):
        with self.assertRaises(ValueError):
            HyperparameterOptimizationConfig.from_dict(
                _base_dict(objective_metric="val_loss", direction="maximize")
            )


if __name__ == "__main__":
    unittest.main()
