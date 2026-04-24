import json
import tempfile
from pathlib import Path
import unittest

from composition.run_model_matrix_evaluation import run_model_matrix_evaluation
from domain.dataset.schema import ALL_COLUMNS
from domain.federated.model_artifact_writer import ModelArtifactWriter
from domain.models.model_registry import MODEL_REGISTRY


class ModelMatrixEvaluationStageIntegrationTests(unittest.TestCase):
    def test_run_model_matrix_evaluation_writes_matrix_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            dataset_path = tmp_dir / "bank_1_test.csv"
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
            global_base = tmp_dir / "federated_global"
            local_checkpoint = local_base / "run_001" / "model.pt"
            global_checkpoint = global_base / "run_002" / "model.pt"
            inclusive_ensemble_config_path = tmp_dir / "ensemble_inclusive_bank_1.toml"
            local_checkpoint.parent.mkdir(parents=True)
            global_checkpoint.parent.mkdir(parents=True)
            ModelArtifactWriter.write_model_checkpoint(
                checkpoint_path=local_checkpoint,
                model_type="tabnet",
                model=model_factory(21),
                model_config=model_config,
            )
            ModelArtifactWriter.write_model_checkpoint(
                checkpoint_path=global_checkpoint,
                model_type="tabnet",
                model=model_factory(21),
                model_config=model_config,
            )
            inclusive_ensemble_config_path.write_text(
                "stage = \"ensemble\"\n"
                "experiment_name = \"ensemble_L1_Fincl\"\n"
                f"output_dir = \"{(tmp_dir / 'ensemble_outputs').as_posix()}\"\n"
                f"local_model_path = \"{local_checkpoint.as_posix()}\"\n"
                f"federated_model_path = \"{global_checkpoint.as_posix()}\"\n"
                f"dataset_path = \"{dataset_path.as_posix()}\"\n"
                "ensemble_weight = 0.15\n"
                "classification_threshold = 0.5\n",
                encoding="utf-8",
            )

            config_path = tmp_dir / "model_matrix.toml"
            config_path.write_text(
                "stage = \"model_matrix_evaluation\"\n"
                "experiment_name = \"tmp_model_matrix\"\n"
                f"output_dir = \"{(tmp_dir / 'outputs').as_posix()}\"\n"
                "classification_threshold = 0.5\n"
                "exclusive_federated_models = []\n"
                "exclusive_ensembles = []\n"
                "\n"
                "[[datasets]]\n"
                "dataset_id = \"bank_1\"\n"
                f"path = \"{dataset_path.as_posix()}\"\n"
                "\n"
                "[[local_models]]\n"
                "model_id = \"local_bank_1\"\n"
                f"base_path = \"{local_base.as_posix()}\"\n"
                "run_number = 1\n"
                "\n"
                "[global_federated_model]\n"
                "model_id = \"federated_global\"\n"
                f"base_path = \"{global_base.as_posix()}\"\n"
                "run_number = 2\n"
                "\n"
                "[[inclusive_ensembles]]\n"
                "model_id = \"ensemble_inclusive_bank_1\"\n"
                "local_model_id = \"local_bank_1\"\n"
                "federated_model_id = \"federated_global\"\n"
                f"config_path = \"{inclusive_ensemble_config_path.as_posix()}\"\n",
                encoding="utf-8",
            )

            output_dir = run_model_matrix_evaluation(config_path)

            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "train.log").exists())
            self.assertTrue((output_dir / "evaluation_matrix.json").exists())
            self.assertTrue((output_dir / "evaluation_matrix.csv").exists())

            payload = json.loads(
                (output_dir / "evaluation_matrix.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["results"]), 3)
            self.assertEqual(
                {record["model_id"] for record in payload["results"]},
                {"local_bank_1", "federated_global", "ensemble_inclusive_bank_1"},
            )
            ensemble_record = next(
                record
                for record in payload["results"]
                if record["model_id"] == "ensemble_inclusive_bank_1"
            )
            self.assertEqual(ensemble_record["ensemble_weight"], 0.15)
            self.assertEqual(
                ensemble_record["ensemble_config_path"],
                inclusive_ensemble_config_path.as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
