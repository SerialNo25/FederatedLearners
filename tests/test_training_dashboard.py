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

    def test_lists_federated_runs_when_stage_requested(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_run = root / "local_bank_1_tabnet" / "run_001"
            local_run.mkdir(parents=True)
            (local_run / "config.json").write_text(
                json.dumps({"stage": "local_training", "institution_id": "bank_1"}),
                encoding="utf-8",
            )

            federated_run = root / "federated_banks_1_2" / "run_001"
            federated_run.mkdir(parents=True)
            (federated_run / "config.json").write_text(
                json.dumps(
                    {
                        "stage": "federated_training",
                        "experiment_name": "federated_banks_1_2",
                        "num_rounds": 3,
                        "proximal_mu": 0.15,
                        "institutions": [
                            {"institution_id": "bank_1"},
                            {"institution_id": "bank_2"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (federated_run / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "stage": "federated_training",
                        "epoch": 2,
                        "train_loss": 0.41,
                        "val_loss": 0.51,
                        "pr_auc": 0.72,
                        "metrics": {
                            "local_loss": {"bank_1": 0.4, "bank_2": 0.42},
                            "local_num_samples": {"bank_1": 100, "bank_2": 120},
                            "local_parameter_delta_l2": {"bank_1": 1.2, "bank_2": 1.4},
                            "institution_evaluation": {
                                "bank_1": {"loss": 0.5, "pr_auc": 0.7, "f1": 0.6},
                                "bank_2": {"loss": 0.52, "pr_auc": 0.74, "f1": 0.62},
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (federated_run / "profile.jsonl").write_text(
                json.dumps({"stage": "federated_training", "epoch": 2, "round_seconds": 1.5})
                + "\n",
                encoding="utf-8",
            )

            monitor = TrainingRunMonitor(TrainingRunMonitorConfig(experiments_dir=root))
            runs = monitor.list_runs(stage="federated_training")
            run = monitor.get_run("federated_banks_1_2/run_001", stage="federated_training")

            self.assertEqual([item["name"] for item in runs], ["federated_banks_1_2/run_001"])
            self.assertEqual(run["stage"], "federated_training")
            self.assertEqual(run["run_type"], "exclusive")
            self.assertEqual(run["institution_ids"], ["bank_1", "bank_2"])
            self.assertEqual(run["total_epochs"], 3)
            self.assertEqual(run["proximal_mu"], 0.15)
            self.assertTrue(run["artifacts"]["profile"])
            self.assertEqual(run["profile_history"][0]["round_seconds"], 1.5)

    def test_identifies_global_federated_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "federated_global" / "run_001"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "stage": "federated_training",
                        "experiment_name": "federated_global",
                        "num_rounds": 1,
                        "institutions": [{"institution_id": "bank_1"}],
                    }
                ),
                encoding="utf-8",
            )

            monitor = TrainingRunMonitor(TrainingRunMonitorConfig(experiments_dir=root))
            run = monitor.get_run("federated_global/run_001", stage="federated_training")

            self.assertEqual(run["run_type"], "global")


if __name__ == "__main__":
    unittest.main()
