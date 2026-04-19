import unittest

from stages.federated_hyperparameter_optimization.config import (
    FederatedHyperparameterOptimizationConfig,
)


def _institution_dict(institution_id: str) -> dict:
    return {
        "institution_id": institution_id,
        "dataset_path": f"data/train_test_splits/{institution_id}.csv",
        "local_epochs": 5,
        "learning_rate": 0.01,
        "model": {"model_type": "tabnet"},
    }


def _base_dict(**overrides) -> dict:
    return {
        "model": {"model_type": "tabnet"},
        "institutions": [_institution_dict("bank_1"), _institution_dict("bank_2")],
        "search_space": {
            "federated": {
                "proximal_mu": {"low": 0.0, "high": 0.1},
                "num_rounds": {"low": 1, "high": 3},
            },
            "local_training": {
                "learning_rate": {"low": 0.001, "high": 0.03, "log": True},
                "local_epochs": {"low": 1, "high": 3},
                "batch_size": {"choices": [128, 256]},
            },
        },
        **overrides,
    }


class FederatedHyperparameterOptimizationConfigTests(unittest.TestCase):
    def test_defaults_apply_for_federated_config(self):
        config = FederatedHyperparameterOptimizationConfig.from_dict(_base_dict())

        self.assertEqual(config.stage, "federated_hyperparameter_optimization")
        self.assertEqual(config.objective_metric, "val_pr_auc")
        self.assertEqual(config.direction, "maximize")
        self.assertEqual(config.n_trials, 10)
        self.assertEqual(config.num_rounds, 5)
        self.assertEqual(config.proximal_mu, 0.0)
        self.assertEqual(len(config.institutions), 2)

    def test_rejects_duplicate_institutions(self):
        with self.assertRaises(ValueError):
            FederatedHyperparameterOptimizationConfig.from_dict(
                _base_dict(
                    institutions=[
                        _institution_dict("bank_1"),
                        _institution_dict("bank_1"),
                    ]
                )
            )

    def test_rejects_invalid_proximal_mu(self):
        with self.assertRaises(ValueError):
            FederatedHyperparameterOptimizationConfig.from_dict(
                _base_dict(proximal_mu=-0.1)
            )

    def test_rejects_unknown_local_training_override_institution(self):
        with self.assertRaises(ValueError):
            FederatedHyperparameterOptimizationConfig.from_dict(
                _base_dict(local_training_overrides={"bank_3": {"learning_rate": 0.001}})
            )

    def test_val_loss_must_minimize(self):
        with self.assertRaises(ValueError):
            FederatedHyperparameterOptimizationConfig.from_dict(
                _base_dict(objective_metric="val_loss", direction="maximize")
            )


if __name__ == "__main__":
    unittest.main()
