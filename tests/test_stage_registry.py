from pathlib import Path
import unittest

from main import StageInvocationConfig
from stages.registry import resolve_stage_runner


class StageRoutingTests(unittest.TestCase):
    def test_config_declares_stage_for_cli_routing(self):
        config = StageInvocationConfig.model_validate({"stage": "evaluation"})

        self.assertEqual(config.stage, "evaluation")

    def test_config_rejects_invalid_stage_name(self):
        with self.assertRaises(ValueError):
            StageInvocationConfig.model_validate({"stage": "../evaluation"})

    def test_resolves_runner_by_composition_root_convention(self):
        runner = resolve_stage_runner("evaluation")

        self.assertEqual(runner.__name__, "run_evaluation")

    def test_rejects_unknown_stage(self):
        with self.assertRaises(KeyError):
            resolve_stage_runner("missing_stage")

    def test_dataset_split_stage_has_been_retired(self):
        with self.assertRaises(KeyError):
            resolve_stage_runner("dataset_split")

    def test_runnable_configs_include_stage_parameter(self):
        runnable_configs = [
            path
            for path in Path("configs").rglob("*.toml")
            if "shared" not in path.parts
        ]

        self.assertTrue(runnable_configs)
        for config_path in runnable_configs:
            with self.subTest(config_path=str(config_path)):
                source = config_path.read_text(encoding="utf-8")
                self.assertIn("stage =", source)


if __name__ == "__main__":
    unittest.main()
