import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from domain.dashboard.training_run_monitor import TrainingRunMonitor, TrainingRunMonitorConfig
from stages.training_dashboard.config import TrainingDashboardConfig


class TrainingDashboardConfigTests(unittest.TestCase):
    def test_defaults_to_local_experiment_root(self):
        config = TrainingDashboardConfig.from_dict({"stage": "training_dashboard"})
        self.assertEqual(config.experiments_dir, Path("data/experiments"))
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8765)

    def test_invalid_port_rejected(self):
        with self.assertRaises(ValueError):
            TrainingDashboardConfig.from_dict({"stage": "training_dashboard", "port": 70000})


class TrainingRunMonitorTests(unittest.TestCase):
    def test_lists_only_local_training_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_run = root / "local_bank_1_tabnet"
            local_run.mkdir()
            (local_run / "config.json").write_text(
                json.dumps(
                    {
                        "stage": "local_training",
                        "experiment_name": "local_bank_1_tabnet",
                        "institution_id": "bank_1",
                        "local_epochs": 2,
                    }
                ),
                encoding="utf-8",
            )
            (local_run / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "stage": "local_training",
                        "epoch": 1,
                        "total_epochs": 2,
                        "train_loss": 0.4,
                        "val_loss": 0.5,
                        "metrics": {"institution_id": "bank_1", "val_pr_auc": 0.8},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (local_run / "run_state.json").write_text(
                json.dumps({"stage": "local_training", "status": "running"}),
                encoding="utf-8",
            )
            (local_run / "model.pt").write_text("old checkpoint", encoding="utf-8")

            federated_run = root / "federated_global"
            federated_run.mkdir()
            (federated_run / "config.json").write_text(
                json.dumps({"stage": "federated_training"}),
                encoding="utf-8",
            )

            monitor = TrainingRunMonitor(TrainingRunMonitorConfig(experiments_dir=root))
            runs = monitor.list_runs()

            self.assertEqual([run["name"] for run in runs], ["local_bank_1_tabnet"])
            self.assertEqual(runs[0]["institution_id"], "bank_1")
            self.assertEqual(runs[0]["epoch"], 1)
            self.assertEqual(runs[0]["status"], "running")

    def test_returns_history_and_log_tail_for_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "local_bank_2_tabnet"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"stage": "local_training", "institution_id": "bank_2"}),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"stage": "local_training", "epoch": 1, "train_loss": 0.9, "val_loss": 1.1}),
                        "not-json",
                        json.dumps({"stage": "local_training", "epoch": 2, "train_loss": 0.7, "val_loss": 0.8}),
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "train.log").write_text("a\nb\n", encoding="utf-8")

            monitor = TrainingRunMonitor(TrainingRunMonitorConfig(experiments_dir=root))
            run = monitor.get_run("local_bank_2_tabnet")

            self.assertEqual(len(run["metrics_history"]), 2)
            self.assertEqual(run["metrics_history"][-1]["epoch"], 2)
            self.assertEqual(run["log_tail"], ["a", "b"])

    def test_lists_numbered_runs_inside_experiment_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "local_bank_1_tabnet" / "run_001"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "stage": "local_training",
                        "experiment_name": "local_bank_1_tabnet",
                        "institution_id": "bank_1",
                        "local_epochs": 3,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "stage": "local_training",
                        "epoch": 2,
                        "total_epochs": 3,
                        "train_loss": 0.4,
                        "val_loss": 0.5,
                        "metrics": {"institution_id": "bank_1", "val_pr_auc": 0.8},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_state.json").write_text(
                json.dumps(
                    {
                        "stage": "local_training",
                        "status": "completed",
                        "experiment_name": "local_bank_1_tabnet",
                        "run_id": "run_001",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "model.pt").write_text("checkpoint", encoding="utf-8")

            monitor = TrainingRunMonitor(TrainingRunMonitorConfig(experiments_dir=root))
            runs = monitor.list_runs()
            run = monitor.get_run("local_bank_1_tabnet/run_001")

            self.assertEqual([item["name"] for item in runs], ["local_bank_1_tabnet/run_001"])
            self.assertEqual(run["experiment_name"], "local_bank_1_tabnet")
            self.assertEqual(run["run_id"], "run_001")
            self.assertTrue(run["artifacts"]["model"])


if __name__ == "__main__":
    unittest.main()
