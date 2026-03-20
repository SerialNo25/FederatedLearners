from pathlib import Path
import ast
import unittest


class PlotGenerationStructureTests(unittest.TestCase):
    def test_plotter_separates_history_and_summary_methods(self):
        source = Path("domain/plotting/experiment_plotter.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        method_names = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("write_local_epoch_plots", method_names)
        self.assertIn("write_local_summary_plots", method_names)
        self.assertIn("write_federated_round_plots", method_names)
        self.assertIn("write_federated_summary_plots", method_names)

    def test_local_stage_calls_summary_plots_after_training(self):
        source = Path("stages/local_training/stage.py").read_text(encoding="utf-8")
        self.assertIn("write_local_epoch_plots", source)
        self.assertIn("write_local_summary_plots", source)

    def test_federated_stage_calls_summary_plots_after_round_collection(self):
        source = Path("stages/federated_training/stage.py").read_text(encoding="utf-8")
        self.assertIn("write_federated_round_plots", source)
        self.assertIn("write_federated_summary_plots", source)


if __name__ == "__main__":
    unittest.main()
