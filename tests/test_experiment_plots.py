from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from domain.dataset.schema import ALL_COLUMNS
from domain.logging.experiment_logger import StageExperimentLogger
from domain.models.model_registry import MODEL_REGISTRY
from stages.federated_training.config import FederatedTrainingConfig
from stages.federated_training.stage import FederatedTrainingStage
from stages.local_training.config import LocalTrainingConfig
from stages.local_training.stage import LocalTrainingStage


class ExperimentPlotGenerationTests(unittest.TestCase):
    def test_local_training_stage_generates_required_plot_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            bank_1 = self._write_dataset(output_dir / "bank_1.csv", offset=0.0)
            config = LocalTrainingConfig.from_dict(
                {
                    "experiment_name": "local_plot_test",
                    "output_dir": str(output_dir),
                    "institutions": [
                        {
                            "institution_id": "bank_1",
                            "dataset_path": str(bank_1),
                        }
                    ],
                    "local_epochs": 2,
                    "learning_rate": 0.05,
                    "model": {"model_type": "logistic_regression"},
                    "local_institution_id": "bank_1",
                }
            )
            experiment_dir = output_dir / config.experiment_name
            model_factory = MODEL_REGISTRY.get_factory(
                config.model.model_type,
                config.model.model_dump(mode="python"),
            )
            stage = LocalTrainingStage(
                config=config,
                experiment_logger=StageExperimentLogger(str(experiment_dir), "local_training"),
                experiment_dir=experiment_dir,
                model_factory=model_factory,
            )

            stage.execute()

            for file_name in self._required_plot_files():
                artifact = experiment_dir / file_name
                self.assertTrue(artifact.exists(), msg=f"Expected plot artifact {file_name}")
                self.assertIn("<svg", artifact.read_text(encoding="utf-8"))

    def test_federated_training_stage_generates_required_plot_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            bank_1 = self._write_dataset(output_dir / "bank_1.csv", offset=0.0)
            bank_2 = self._write_dataset(output_dir / "bank_2.csv", offset=0.7)
            config = FederatedTrainingConfig.from_dict(
                {
                    "experiment_name": "federated_plot_test",
                    "output_dir": str(output_dir),
                    "num_institutions": 2,
                    "institutions": [
                        {
                            "institution_id": "bank_1",
                            "dataset_path": str(bank_1),
                        },
                        {
                            "institution_id": "bank_2",
                            "dataset_path": str(bank_2),
                        },
                    ],
                    "num_rounds": 2,
                    "local_epochs": 2,
                    "learning_rate": 0.05,
                    "proximal_mu": 0.001,
                    "model": {"model_type": "logistic_regression"},
                }
            )
            experiment_dir = output_dir / config.experiment_name
            model_factory = MODEL_REGISTRY.get_factory(
                config.model.model_type,
                config.model.model_dump(mode="python"),
            )
            stage = FederatedTrainingStage(
                config=config,
                experiment_logger=StageExperimentLogger(str(experiment_dir), "federated_training"),
                experiment_dir=experiment_dir,
                model_factory=model_factory,
            )

            stage.execute()

            for file_name in self._required_plot_files():
                artifact = experiment_dir / file_name
                self.assertTrue(artifact.exists(), msg=f"Expected plot artifact {file_name}")
                self.assertIn("<svg", artifact.read_text(encoding="utf-8"))

    @staticmethod
    def _required_plot_files() -> list[str]:
        return [
            "loss_plot.svg",
            "pr_auc.svg",
            "f1_optimal_threshold.svg",
            "threshold_curves.svg",
            "per_client_performance_boxplots.svg",
            "global_vs_local_convergence.svg",
        ]

    @staticmethod
    def _write_dataset(path: Path, offset: float) -> Path:
        rows = [
            [1.0 + offset, 0.0 + offset, 2.0, 1.0, 0],
            [1.5 + offset, 0.2 + offset, 4.0, 1.0, 0],
            [2.0 + offset, 0.5 + offset, 6.0, 2.0, 0],
            [3.0 + offset, 0.8 + offset, 8.0, 2.0, 1],
            [3.5 + offset, 1.1 + offset, 10.0, 3.0, 1],
            [4.0 + offset, 1.3 + offset, 12.0, 3.0, 1],
        ]
        contents = [",".join(ALL_COLUMNS)]
        contents.extend(",".join(str(value) for value in row) for row in rows)
        path.write_text("\n".join(contents) + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
