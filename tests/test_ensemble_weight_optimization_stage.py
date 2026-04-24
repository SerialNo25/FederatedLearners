import json
import tempfile
from pathlib import Path
import unittest

from composition.run_ensemble_weight_optimization import run_ensemble_weight_optimization
from domain.dataset.schema import ALL_COLUMNS
from domain.federated.model_artifact_writer import ModelArtifactWriter
from domain.models.model_registry import MODEL_REGISTRY


class EnsembleWeightOptimizationStageIntegrationTests(unittest.TestCase):
    def test_run_ensemble_weight_optimization_writes_optuna_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            dataset_path = tmp_dir / "bank_1_train.csv"
            rows = [
                [10.0, 2.3, -0.5, 0.2, 0.7, -0.7, 0.0, 1.0, 0, 0, 1, 1, 0.25, 0.1, 1, 0, 0, 0, 0, 0, 0, 0],
                [120.0, 4.8, 1.4, 0.9, -0.5, -0.8, 0.4, -0.9, 0, 0, 1, 0, 0.55, 0.3, 0, 1, 0, 0, 0, 0, 0, 1],
                [15.0, 2.7, -0.4, 0.3, -0.9, 0.3, -0.4, -0.9, 1, 1, 0, 1, 0.35, 0.2, 0, 0, 1, 0, 0, 0, 0, 0],
                [250.0, 5.5, 2.1, 0.98, -0.3, 0.95, -0.8, 0.6, 1, 1, 1, 0, 0.75, 0.5, 0, 0, 0, 1, 0, 0, 0, 1],
            ]
            dataset_path.write_text(
                ",".join(ALL_COLUMNS)
                + "\n"
                + "\n".join(",".join(str(value) for value in row) for row in rows)
                + "\n",
                encoding="utf-8",
            )

            model_config = {
                "model_type": "tabnet",
                "decision_dim": 16,
                "attention_dim": 16,
                "steps": 3,
                "relaxation_factor": 1.5,
                "sparsity_weight": 1e-4,
            }
            model_factory = MODEL_REGISTRY.get_factory("tabnet", model_config)

            local_base = tmp_dir / "local_bank_1"
            federated_base = tmp_dir / "federated_global"
            local_checkpoint = local_base / "run_001" / "model.pt"
            federated_checkpoint = federated_base / "run_002" / "model.pt"
            local_checkpoint.parent.mkdir(parents=True)
            federated_checkpoint.parent.mkdir(parents=True)
            ModelArtifactWriter.write_model_checkpoint(
                checkpoint_path=local_checkpoint,
                model_type="tabnet",
                model=model_factory(21),
                model_config=model_config,
            )
            ModelArtifactWriter.write_model_checkpoint(
                checkpoint_path=federated_checkpoint,
                model_type="tabnet",
                model=model_factory(21),
                model_config=model_config,
            )

            config_path = tmp_dir / "ensemble_weight_hpo.toml"
            config_path.write_text(
                "stage = \"ensemble_weight_optimization\"\n"
                "experiment_name = \"tmp_ensemble_weight_hpo\"\n"
                f"output_dir = \"{(tmp_dir / 'outputs').as_posix()}\"\n"
                "institution_id = \"bank_1\"\n"
                f"dataset_path = \"{dataset_path.as_posix()}\"\n"
                "n_trials = 3\n"
                "study_name = \"tmp_ensemble_weight_hpo\"\n"
                "classification_threshold = 0.5\n"
                "\n"
                "[local_model]\n"
                "model_id = \"local_bank_1\"\n"
                f"base_path = \"{local_base.as_posix()}\"\n"
                "run_number = 1\n"
                "\n"
                "[federated_model]\n"
                "model_id = \"federated_global\"\n"
                f"base_path = \"{federated_base.as_posix()}\"\n"
                "run_number = 2\n"
                "\n"
                "[search_space.ensemble_weight]\n"
                "low = 0.0\n"
                "high = 1.0\n"
                "step = 0.5\n",
                encoding="utf-8",
            )

            output_dir = run_ensemble_weight_optimization(config_path)

            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "train.log").exists())
            self.assertTrue((output_dir / "run_state.json").exists())
            self.assertTrue((output_dir / "best_params.json").exists())
            self.assertTrue((output_dir / "optuna_trials.csv").exists())

            best_payload = json.loads((output_dir / "best_params.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["institution_id"], "bank_1")
            self.assertEqual(best_payload["local_model"]["model_id"], "local_bank_1")
            self.assertEqual(best_payload["federated_model"]["model_id"], "federated_global")
            self.assertIn("ensemble_weight", best_payload["best_params"])


if __name__ == "__main__":
    unittest.main()
