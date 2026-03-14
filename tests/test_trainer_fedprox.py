import ast
from pathlib import Path
import unittest

from domain.training.trainer import TrainingConfig, train_local_model


class _ManualTrainerModel:
    def __init__(self):
        self.weights = [0.0, 0.0]
        self.bias = 0.0

    def parameters(self):
        return {"weights": list(self.weights), "bias": [self.bias]}

    def predict_proba(self, features):
        return [0.5 for _ in features]


class TrainerFedProxTests(unittest.TestCase):
    def test_train_local_model_manual_branch_handles_empty_dataset(self):
        model = _ManualTrainerModel()
        config = TrainingConfig(learning_rate=0.1, local_epochs=2, proximal_mu=0.5)

        loss = train_local_model(
            model=model,
            features=[],
            labels=[],
            config=config,
            global_parameters={"weights": [0.2, -0.2], "bias": [0.0]},
        )

        self.assertEqual(loss, 0.0)
        self.assertEqual(model.weights, [0.0, 0.0])
        self.assertEqual(model.bias, 0.0)

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
