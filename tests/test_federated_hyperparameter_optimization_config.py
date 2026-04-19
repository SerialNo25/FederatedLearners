import unittest
from pathlib import Path

import tomli

from stages.federated_hyperparameter_optimization.config import (
    FederatedHyperparameterOptimizationConfig,
)
from stages.local_training.config import LocalTrainingConfig


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

    def test_exclusive_combo_configs_have_separate_studies_and_results(self):
        expected = {
            "banks_1_2": ["bank_1", "bank_2"],
            "banks_1_3": ["bank_1", "bank_3"],
            "banks_2_3": ["bank_2", "bank_3"],
        }
        seen_storage_urls: set[str] = set()
        seen_experiment_names: set[str] = set()

        for combo, institution_ids in expected.items():
            config = _load_repo_config(
                Path("configs/federated_hyperparameter_optimization") / f"{combo}.toml"
            )

            self.assertEqual(config.stage, "federated_hyperparameter_optimization")
            self.assertEqual(
                config.experiment_name,
                f"hpo_federated_{combo}_tabnet",
            )
            self.assertEqual(config.study_name, f"hpo_federated_{combo}_tabnet")
            self.assertEqual(
                config.storage_url,
                f"sqlite:///data/experiments/hpo_federated_{combo}_tabnet/optuna.db",
            )
            self.assertEqual(
                [institution.institution_id for institution in config.institutions],
                institution_ids,
            )
            self.assertEqual(
                set(config.local_training_overrides),
                set(institution_ids),
            )
            self.assertIsNotNone(config.search_space.federated.proximal_mu)
            self.assertIsNotNone(config.search_space.federated.num_rounds)
            seen_storage_urls.add(config.storage_url or "")
            seen_experiment_names.add(config.experiment_name)

        self.assertEqual(len(seen_storage_urls), 3)
        self.assertEqual(len(seen_experiment_names), 3)


def _load_repo_config(path: Path) -> FederatedHyperparameterOptimizationConfig:
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    model_dict = tomli.loads(Path(config_dict.pop("model_config")).read_text(encoding="utf-8"))
    institution_config_paths = config_dict.pop("institution_configs")
    institutions = []
    for institution_config_path in institution_config_paths:
        institution_dict = tomli.loads(Path(institution_config_path).read_text(encoding="utf-8"))
        institution_dict.pop("model_config", None)
        institution_dict["model"] = model_dict
        institutions.append(LocalTrainingConfig.from_dict(institution_dict))

    config_dict["model"] = model_dict
    config_dict["institutions"] = institutions
    return FederatedHyperparameterOptimizationConfig.from_dict(config_dict)


if __name__ == "__main__":
    unittest.main()
