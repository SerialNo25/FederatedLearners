import ast
from pathlib import Path
import unittest

from domain.metrics.aggregation import weighted_mean


class FederatedMetricsTests(unittest.TestCase):
    def test_weighted_mean_uses_sample_counts(self):
        mean = weighted_mean(values=[0.9, 0.3], weights=[1, 9])
        self.assertAlmostEqual(mean, 0.36)

    def test_round_reporter_computes_weighted_train_loss(self):
        source = Path("stages/federated_training/round_reporter.py").read_text()
        module = ast.parse(source)

        weighted_mean_used = False
        for node in ast.walk(module):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "train_loss":
                        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                            weighted_mean_used = value.func.id == "weighted_mean"

        self.assertTrue(weighted_mean_used)

    def test_stage_persists_model_via_artifact_writer(self):
        source = Path("stages/federated_training/stage.py").read_text()
        self.assertIn("ModelArtifactWriter.write_model_checkpoint", source)


if __name__ == "__main__":
    unittest.main()
