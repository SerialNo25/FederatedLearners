import ast
from pathlib import Path
import unittest


class InferenceStageArchitectureTests(unittest.TestCase):
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
        self.assertNotIn("tabnet_device", field_names)

    def test_inference_stage_orchestrates_domain_services(self):
        source = Path("stages/inference/stage.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        has_csv_usage = False
        service_calls = {"load_csv": False, "load": False, "run": False}
        uses_device_selector = False

        for node in ast.walk(module):
            if isinstance(node, ast.Name) and node.id == "DeviceSelector":
                uses_device_selector = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "csv":
                    has_csv_usage = True
                if node.func.attr in service_calls:
                    service_calls[node.func.attr] = True

        self.assertFalse(has_csv_usage)
        self.assertFalse(uses_device_selector)
        self.assertTrue(all(service_calls.values()))

    def test_inference_domain_service_handles_csv_checkpoint_and_metrics(self):
        source = Path("domain/inference/inference_service.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        has_csv_dict_reader = False
        has_torch_load = False
        computes_accuracy = False
        computes_loss = False

        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "csv":
                        if node.func.attr == "DictReader":
                            has_csv_dict_reader = True
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                        if node.func.attr == "load":
                            has_torch_load = True
                if isinstance(node.func, ast.Name) and node.func.id == "binary_cross_entropy":
                    computes_loss = True
            if isinstance(node, ast.Constant) and node.value == "accuracy":
                computes_accuracy = True

        self.assertTrue(has_csv_dict_reader)
        self.assertTrue(has_torch_load)
        self.assertTrue(computes_accuracy)
        self.assertTrue(computes_loss)


if __name__ == "__main__":
    unittest.main()
