import ast
from pathlib import Path
import unittest


class StageRegistryPresetArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("stages/registry.py").read_text(encoding="utf-8")
        cls.module = ast.parse(cls.source)

    def test_registry_tracks_presets(self):
        init_fn = None
        for node in ast.walk(self.module):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                init_fn = node
                break

        self.assertIsNotNone(init_fn)
        assigns_preset_dict = any(
            (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Attribute) and target.attr == "_presets" for target in node.targets)
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and node.target.attr == "_presets"
            )
            for node in init_fn.body
        )
        self.assertTrue(assigns_preset_dict)



    def test_default_registry_registers_evaluation_stage(self):
        found = False
        for node in ast.walk(self.module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "register" and len(node.args) >= 2:
                    stage = node.args[0]
                    if isinstance(stage, ast.Constant) and stage.value == "evaluation":
                        found = True
                        break

        self.assertTrue(found)

    def test_default_registry_registers_evaluation_default_preset(self):
        found = False
        for node in ast.walk(self.module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "register_preset" and len(node.args) >= 3:
                    stage = node.args[0]
                    preset = node.args[1]
                    if (
                        isinstance(stage, ast.Constant)
                        and stage.value == "evaluation"
                        and isinstance(preset, ast.Constant)
                        and preset.value == "default"
                    ):
                        found = True
                        break

        self.assertTrue(found)

    def test_default_registry_registers_local_training_stage(self):
        found = False
        for node in ast.walk(self.module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "register" and len(node.args) >= 2:
                    stage = node.args[0]
                    if isinstance(stage, ast.Constant) and stage.value == "local_training":
                        found = True
                        break

        self.assertTrue(found)


    def test_default_registry_registers_local_bank_presets(self):
        expected = {"bank_1", "bank_2", "bank_3"}
        found = set()

        for node in ast.walk(self.module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "register_preset" and len(node.args) >= 3:
                    stage = node.args[0]
                    preset = node.args[1]
                    if (
                        isinstance(stage, ast.Constant)
                        and stage.value == "local_training"
                        and isinstance(preset, ast.Constant)
                        and preset.value in expected
                    ):
                        found.add(preset.value)

        self.assertEqual(found, expected)

    def test_default_registry_registers_banks_1_2_preset(self):
        found = False
        for node in ast.walk(self.module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "register_preset" and len(node.args) >= 3:
                    stage = node.args[0]
                    preset = node.args[1]
                    if (
                        isinstance(stage, ast.Constant)
                        and stage.value == "federated_training"
                        and isinstance(preset, ast.Constant)
                        and preset.value == "banks_1_2"
                    ):
                        found = True
                        break

        self.assertTrue(found)

    def test_resolve_config_enforces_mutually_exclusive_inputs(self):
        resolve_fn = None
        for node in ast.walk(self.module):
            if isinstance(node, ast.FunctionDef) and node.name == "resolve_config_path":
                resolve_fn = node
                break

        self.assertIsNotNone(resolve_fn)
        has_both_guard = False
        for node in ast.walk(resolve_fn):
            if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
                if isinstance(node.test.op, ast.And) and len(node.test.values) == 2:
                    has_both_guard = True
        self.assertTrue(has_both_guard)


if __name__ == "__main__":
    unittest.main()
