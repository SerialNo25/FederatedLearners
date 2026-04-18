import unittest

import torch

from domain.models.tabnet_model import TabNetModel


class TabNetModelTests(unittest.TestCase):
    def test_forward_supports_distinct_decision_and_attention_dims(self):
        model = TabNetModel.initialize(
            n_features=30,
            decision_dim=8,
            attention_dim=16,
            n_steps=2,
            device="cpu",
        )

        logits, sparsity_loss = model(torch.zeros((4, 30), dtype=torch.float32))

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(tuple(sparsity_loss.shape), ())

    def test_predict_proba_supports_distinct_decision_and_attention_dims(self):
        model = TabNetModel.initialize(
            n_features=30,
            decision_dim=32,
            attention_dim=8,
            n_steps=2,
            device="cpu",
        )

        probabilities = model.predict_proba([[0.0] * 30, [1.0] * 30])

        self.assertEqual(len(probabilities), 2)
        self.assertTrue(all(0.0 <= probability <= 1.0 for probability in probabilities))


if __name__ == "__main__":
    unittest.main()
