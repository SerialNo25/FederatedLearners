import ast
from pathlib import Path
import unittest


class FederatedTrainingSimplificationArchitectureTests(unittest.TestCase):
    def test_stage_uses_round_reporter(self):
        source = Path("stages/federated_training/stage.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        has_reporter_call = False
        has_stage_metrics_write = False
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "build_report":
                    has_reporter_call = True
                if node.func.attr == "write_metrics":
                    has_stage_metrics_write = True

        self.assertTrue(has_reporter_call)
        self.assertTrue(has_stage_metrics_write)

    def test_stage_uses_model_artifact_writer(self):
        source = Path("stages/federated_training/stage.py").read_text(encoding="utf-8")
        self.assertIn("ModelArtifactWriter.write_model_checkpoint", source)

    def test_fedprox_uses_shared_model_parameter_helper(self):
        source = Path("domain/federated/fedprox_orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("get_model_parameters", source)
        self.assertNotIn("def _get_model_parameters", source)

    def test_fedavg_uses_shared_model_parameter_helper(self):
        source = Path("domain/federated/fedavg.py").read_text(encoding="utf-8")
        self.assertIn("get_model_parameters", source)
        self.assertNotIn("def _get_model_parameters", source)


if __name__ == "__main__":
    unittest.main()
