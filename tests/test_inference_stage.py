import ast
from pathlib import Path
import unittest


class InferenceStageTests(unittest.TestCase):
    def test_inference_config_exposes_file_and_label_fields(self):
        source = Path("stages/inference/config.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        field_names = set()
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == "InferenceConfig":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        field_names.add(item.target.id)

        self.assertIn("input_data_path", field_names)
        self.assertIn("label_column", field_names)
        self.assertIn("feature_columns", field_names)

    def test_inference_stage_reads_csv_and_computes_optional_metrics(self):
        source = Path("stages/inference/stage.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        has_csv_dict_reader = False
        computes_accuracy = False
        computes_loss = False

        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "csv":
                    if node.func.attr == "DictReader":
                        has_csv_dict_reader = True
                if node.func.attr == "get" and node.args:
                    if isinstance(node.args[0], ast.Constant) and node.args[0].value == "loss":
                        computes_loss = True
            if isinstance(node, ast.Constant) and node.value == "accuracy":
                computes_accuracy = True

        self.assertTrue(has_csv_dict_reader)
        self.assertTrue(computes_accuracy)
        self.assertTrue(computes_loss)


if __name__ == "__main__":
    unittest.main()
