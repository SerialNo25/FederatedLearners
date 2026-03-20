import ast
from pathlib import Path
import unittest


class TrainerFedProxTests(unittest.TestCase):
    def test_tabnet_model_no_longer_uses_dataclass_decorator(self):
        source = Path("domain/models/tabnet_model.py").read_text()
        module = ast.parse(source)

        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == "TabNetModel":
                self.assertFalse(
                    any(
                        isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                        for decorator in node.decorator_list
                    )
                )
                return

        self.fail("TabNetModel class definition not found")


if __name__ == "__main__":
    unittest.main()
