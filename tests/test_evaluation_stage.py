import ast
import json
import tempfile
from pathlib import Path
import unittest

from composition.run_evaluation import run_evaluation
from domain.federated.model_artifact_writer import ModelArtifactWriter
from domain.models.model_registry import MODEL_REGISTRY


class EvaluationStageArchitectureTests(unittest.TestCase):
    def test_evaluation_config_exposes_model_and_dataset_paths(self):
        source = Path("stages/evaluation/config.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        field_names = set()
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == "EvaluationConfig":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        field_names.add(item.target.id)

        self.assertIn("model_path", field_names)
        self.assertIn("dataset_path", field_names)
        self.assertIn("classification_threshold", field_names)

    def test_evaluation_stage_uses_domain_services_and_logs_pr_auc(self):
        source = Path("stages/evaluation/stage.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        service_calls = {"load": False, "evaluate": False}
        calls_loader_directly = False

        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in service_calls:
                    service_calls[node.func.attr] = True
                if node.func.attr == "DictReader":
                    calls_loader_directly = True

        self.assertTrue(all(service_calls.values()))
        self.assertFalse(calls_loader_directly)
        self.assertIn("pr_auc", source)


class EvaluationStageIntegrationTests(unittest.TestCase):
    def test_run_evaluation_writes_metrics_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            dataset_path = tmp_dir / "dataset.csv"
            columns = [
                "amount",
                "log_amount",
                "amount_zscore",
                "amount_percentile",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_night",
                "is_round_amount",
                "gender_M",
                "age_normalized",
                "geo_encoded",
                "cat_grocery",
                "cat_shopping",
                "cat_entertainment",
                "cat_gas_transport",
                "cat_food_dining",
                "cat_health_personal",
                "cat_other",
                "is_fraud",
            ]
            rows = [
                [10.0, 2.3, -0.5, 0.2, 0.7, -0.7, 0.0, 1.0, 0, 0, 1, 1, 0.25, 0.1, 1, 0, 0, 0, 0, 0, 0, 0],
                [120.0, 4.8, 1.4, 0.9, -0.5, -0.8, 0.4, -0.9, 0, 0, 1, 0, 0.55, 0.3, 0, 1, 0, 0, 0, 0, 0, 1],
                [15.0, 2.7, -0.4, 0.3, -0.9, 0.3, -0.4, -0.9, 1, 1, 0, 1, 0.35, 0.2, 0, 0, 1, 0, 0, 0, 0, 0],
                [250.0, 5.5, 2.1, 0.98, -0.3, 0.95, -0.8, 0.6, 1, 1, 1, 0, 0.75, 0.5, 0, 0, 0, 1, 0, 0, 0, 1],
            ]
            dataset_path.write_text(
                ",".join(columns)
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
            model = model_factory(21)

            checkpoint_path = tmp_dir / "model.pt"
            ModelArtifactWriter.write_model_checkpoint(
                checkpoint_path=checkpoint_path,
                model_type="tabnet",
                model=model,
                model_config=model_config,
            )

            config_path = tmp_dir / "evaluation.toml"
            config_path.write_text(
                "stage = \"evaluation\"\n"
                "experiment_name = \"tmp_eval\"\n"
                f"output_dir = \"{(tmp_dir / 'outputs').as_posix()}\"\n"
                f"model_path = \"{checkpoint_path.as_posix()}\"\n"
                f"dataset_path = \"{dataset_path.as_posix()}\"\n"
                "classification_threshold = 0.5\n",
                encoding="utf-8",
            )

            output_dir = run_evaluation(config_path)

            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "train.log").exists())
            self.assertTrue((output_dir / "evaluation.json").exists())

            payload = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["model_type"], "tabnet")
            self.assertEqual(payload["dataset_path"], dataset_path.as_posix())
            self.assertEqual(
                set(payload["metrics"]),
                {"loss", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr_at_95_recall"},
            )

            for value in payload["metrics"].values():
                self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
