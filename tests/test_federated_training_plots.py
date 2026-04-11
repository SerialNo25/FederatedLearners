from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from domain.logging.loss_plot_writer import LossPlotWriter
from domain.logging.pr_auc_plot_writer import PRAUCPlotWriter


class FederatedTrainingPlotWriterTests(unittest.TestCase):
    def test_loss_plot_writer_persists_svg_with_expected_labels(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "loss_plot.svg"

            LossPlotWriter.write(
                output_path=output_path,
                rounds=[1, 2, 3],
                train_losses=[0.8, 0.6, 0.4],
                val_losses=[0.9, 0.7, 0.5],
            )

            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("Federated Loss Over Rounds", content)
            self.assertIn("Train Loss", content)
            self.assertIn("Validation Loss", content)

    def test_pr_auc_plot_writer_persists_svg_with_expected_labels(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pr_auc_plot.svg"

            PRAUCPlotWriter.write(
                output_path=output_path,
                rounds=[1, 2, 3],
                pr_auc_values=[0.2, 0.35, 0.5],
            )

            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("Federated PR-AUC Over Rounds", content)
            self.assertIn("PR-AUC", content)


class FederatedTrainingPlotIntegrationTests(unittest.TestCase):
    def test_round_reporter_exposes_top_level_pr_auc_metric(self):
        source = Path("stages/federated_training/round_reporter.py").read_text(encoding="utf-8")
        self.assertIn('"pr_auc"', source)

    def test_federated_stage_writes_loss_and_pr_auc_plots(self):
        source = Path("stages/federated_training/stage.py").read_text(encoding="utf-8")
        self.assertIn("LossPlotWriter.write", source)
        self.assertIn("PRAUCPlotWriter.write", source)


if __name__ == "__main__":
    unittest.main()
