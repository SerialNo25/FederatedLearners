import unittest
from pathlib import Path

import tomli

from stages.ensemble_weight_optimization.config import EnsembleWeightOptimizationConfig


def _base_dict(**overrides) -> dict:
    payload = {
        "institution_id": "bank_1",
        "dataset_path": "data/train_test_splits/bank_1_train.csv",
        "local_model": {
            "model_id": "local_bank_1",
            "base_path": "data/experiments/local_bank_1_tabnet",
            "run_number": 28,
        },
        "federated_model": {
            "model_id": "federated_global",
            "base_path": "data/experiments/federated_global",
            "run_number": 16,
        },
    }
    payload.update(overrides)
    return payload


class EnsembleWeightOptimizationConfigTests(unittest.TestCase):
    def test_checkpoint_path_uses_base_path_and_run_number(self):
        config = EnsembleWeightOptimizationConfig.from_dict(_base_dict())

        self.assertEqual(
            str(config.local_model.checkpoint_path),
            "data/experiments/local_bank_1_tabnet/run_028/model.pt",
        )
        self.assertEqual(
            str(config.federated_model.checkpoint_path),
            "data/experiments/federated_global/run_016/model.pt",
        )
        self.assertEqual(config.search_space.ensemble_weight.low, 0.0)
        self.assertEqual(config.search_space.ensemble_weight.high, 1.0)
        self.assertEqual(config.search_space.ensemble_weight.step, 0.01)

    def test_rejects_duplicate_source_model_ids(self):
        with self.assertRaises(ValueError):
            EnsembleWeightOptimizationConfig.from_dict(
                _base_dict(
                    federated_model={
                        "model_id": "local_bank_1",
                        "base_path": "data/experiments/federated_global",
                        "run_number": 16,
                    }
                )
            )

    def test_repo_configs_target_expected_runs_and_datasets(self):
        expectations = {
            "configs/ensemble_weight_optimization/exclusive/bank_1.toml": {
                "experiment_name": "hpo_ensemble_exclusive_bank_1",
                "dataset_path": "data/train_test_splits/bank_1_train.csv",
                "local_model_base_path": "data/experiments/local_bank_1_tabnet",
                "local_model_run_number": 28,
                "federated_model_base_path": "data/experiments/federated_banks_2_3",
                "federated_model_run_number": 1,
            },
            "configs/ensemble_weight_optimization/exclusive/bank_2.toml": {
                "experiment_name": "hpo_ensemble_exclusive_bank_2",
                "dataset_path": "data/train_test_splits/bank_2_train.csv",
                "local_model_base_path": "data/experiments/local_bank_2_banksim",
                "local_model_run_number": 3,
                "federated_model_base_path": "data/experiments/federated_banks_1_3",
                "federated_model_run_number": 3,
            },
            "configs/ensemble_weight_optimization/exclusive/bank_3.toml": {
                "experiment_name": "hpo_ensemble_exclusive_bank_3",
                "dataset_path": "data/train_test_splits/bank_3_train.csv",
                "local_model_base_path": "data/experiments/local_bank_3_ccfraud",
                "local_model_run_number": 7,
                "federated_model_base_path": "data/experiments/federated_banks_1_2",
                "federated_model_run_number": 3,
            },
            "configs/ensemble_weight_optimization/inclusive/bank_1.toml": {
                "experiment_name": "hpo_ensemble_inclusive_bank_1",
                "dataset_path": "data/train_test_splits/bank_1_train.csv",
                "local_model_base_path": "data/experiments/local_bank_1_tabnet",
                "local_model_run_number": 28,
                "federated_model_base_path": "data/experiments/federated_global",
                "federated_model_run_number": 16,
            },
            "configs/ensemble_weight_optimization/inclusive/bank_2.toml": {
                "experiment_name": "hpo_ensemble_inclusive_bank_2",
                "dataset_path": "data/train_test_splits/bank_2_train.csv",
                "local_model_base_path": "data/experiments/local_bank_2_banksim",
                "local_model_run_number": 3,
                "federated_model_base_path": "data/experiments/federated_global",
                "federated_model_run_number": 16,
            },
            "configs/ensemble_weight_optimization/inclusive/bank_3.toml": {
                "experiment_name": "hpo_ensemble_inclusive_bank_3",
                "dataset_path": "data/train_test_splits/bank_3_train.csv",
                "local_model_base_path": "data/experiments/local_bank_3_ccfraud",
                "local_model_run_number": 7,
                "federated_model_base_path": "data/experiments/federated_global",
                "federated_model_run_number": 16,
            },
        }

        for config_path_str, expectation in expectations.items():
            with self.subTest(config=config_path_str):
                config = _load_repo_config(Path(config_path_str))
                self.assertEqual(config.stage, "ensemble_weight_optimization")
                self.assertEqual(config.experiment_name, expectation["experiment_name"])
                self.assertEqual(str(config.dataset_path), expectation["dataset_path"])
                self.assertEqual(str(config.local_model.base_path), expectation["local_model_base_path"])
                self.assertEqual(config.local_model.run_number, expectation["local_model_run_number"])
                self.assertEqual(
                    str(config.federated_model.base_path),
                    expectation["federated_model_base_path"],
                )
                self.assertEqual(
                    config.federated_model.run_number,
                    expectation["federated_model_run_number"],
                )
                self.assertEqual(config.study_name, expectation["experiment_name"])
                self.assertEqual(
                    config.storage_url,
                    f"sqlite:///data/experiments/{expectation['experiment_name']}/optuna.db",
                )


def _load_repo_config(path: Path) -> EnsembleWeightOptimizationConfig:
    config_dict = tomli.loads(path.read_text(encoding="utf-8"))
    return EnsembleWeightOptimizationConfig.from_dict(config_dict)


if __name__ == "__main__":
    unittest.main()
