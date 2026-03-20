from pathlib import Path
import ast
import unittest


class EvaluationMetricsArchitectureTests(unittest.TestCase):
    def test_evaluation_prefers_sklearn_precision_recall_curve(self):
        source = Path("domain/metrics/evaluation.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        imported_names = set()
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module == "sklearn.metrics":
                imported_names.update(alias.asname or alias.name for alias in node.names)

        self.assertIn("sklearn_auc", imported_names)
        self.assertIn("sklearn_precision_recall_curve", imported_names)
        self.assertIn("_compute_threshold_curve_sklearn", source)

    def test_pyproject_declares_scikit_learn_dependency(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"scikit-learn>=1.5.0"', pyproject)


if __name__ == "__main__":
    unittest.main()
