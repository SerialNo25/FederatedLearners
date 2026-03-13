import ast
from pathlib import Path
import unittest


class FederatedTrainingStageInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("stages/federated_training/stage.py").read_text()
        cls.module = ast.parse(cls.source)

    def test_execute_runs_dataset_invariant_check(self):
        execute = None
        for node in ast.walk(self.module):
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                execute = node
                break

        self.assertIsNotNone(execute)
        calls_invariant_check = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_assert_dataset_invariants"
            for node in ast.walk(execute)
        )
        self.assertTrue(calls_invariant_check)

    def test_invariant_check_rejects_empty_datasets(self):
        invariant = None
        for node in ast.walk(self.module):
            if isinstance(node, ast.FunctionDef) and node.name == "_assert_dataset_invariants":
                invariant = node
                break

        self.assertIsNotNone(invariant)
        runtime_error_messages = []
        for node in ast.walk(invariant):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "RuntimeError":
                    if node.exc.args and isinstance(node.exc.args[0], ast.Constant):
                        runtime_error_messages.append(node.exc.args[0].value)

        self.assertTrue(
            any("at least one institution dataset" in message for message in runtime_error_messages)
        )


if __name__ == "__main__":
    unittest.main()
