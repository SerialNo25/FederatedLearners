import ast
from pathlib import Path
import unittest


class FederatedTrainingStageArchitectureTests(unittest.TestCase):
    def test_federated_training_stage_logs_selected_model_device(self):
        source = Path("stages/federated_training/stage.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        has_device_log = False
        for node in ast.walk(module):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "tabnet_device_selection selected=" in node.value:
                    has_device_log = True

        self.assertTrue(has_device_log)


if __name__ == "__main__":
    unittest.main()
