import ast
from pathlib import Path
import unittest

from domain.training.trainer import TrainingConfig, train_local_model


class _CapturingFitModel:
    def __init__(self):
        self.received_kwargs = None

    def fit(
        self,
        features,
        labels,
        learning_rate,
        epochs,
        global_parameters=None,
        proximal_mu=0.0,
    ):
        self.received_kwargs = {
            "features": features,
            "labels": labels,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "global_parameters": global_parameters,
            "proximal_mu": proximal_mu,
        }
        return 0.0


class _LegacyFitModel:
    def fit(self, features, labels, learning_rate, epochs):
        return float(len(features) + len(labels) + learning_rate + epochs)


class TrainerFedProxTests(unittest.TestCase):
    def test_train_local_model_passes_proximal_arguments_when_supported(self):
        model = _CapturingFitModel()
        features = [[0.0, 1.0]]
        labels = [1]
        config = TrainingConfig(learning_rate=0.1, local_epochs=2, proximal_mu=0.5)
        global_parameters = {"weights": [0.2, -0.2], "bias": [0.0]}

        train_local_model(
            model=model,
            features=features,
            labels=labels,
            config=config,
            global_parameters=global_parameters,
        )

        assert model.received_kwargs is not None
        self.assertEqual(model.received_kwargs["global_parameters"], global_parameters)
        self.assertEqual(model.received_kwargs["proximal_mu"], 0.5)

    def test_train_local_model_keeps_backward_compatibility_for_legacy_fit(self):
        model = _LegacyFitModel()
        features = [[0.0, 1.0]]
        labels = [1]
        config = TrainingConfig(learning_rate=0.1, local_epochs=2, proximal_mu=0.5)

        loss = train_local_model(
            model=model,
            features=features,
            labels=labels,
            config=config,
            global_parameters={"weights": [0.2, -0.2], "bias": [0.0]},
        )

        self.assertEqual(loss, 4.1)

    def test_tabnet_fit_signature_supports_fedprox_arguments(self):
        source = Path("domain/models/tabnet_model.py").read_text()
        module = ast.parse(source)

        fit_args = None
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == "TabNetModel":
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef) and class_node.name == "fit":
                        fit_args = [arg.arg for arg in class_node.args.args]
                        break

        self.assertIsNotNone(fit_args)
        self.assertIn("global_parameters", fit_args)
        self.assertIn("proximal_mu", fit_args)


if __name__ == "__main__":
    unittest.main()
