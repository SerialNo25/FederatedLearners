import unittest

import torch
from torch import nn

from domain.training.trainer import TrainingConfig, train_local_model


class TinyBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.device = "cpu"
        self.sparsity_weight = 0.0

    def forward(self, inputs):
        return self.linear(inputs).squeeze(-1), torch.tensor(0.0)


class TrainerProgressTests(unittest.TestCase):
    def test_epoch_callback_receives_each_epoch_loss(self):
        model = TinyBinaryModel()
        observed: list[tuple[int, float]] = []

        final_loss = train_local_model(
            model=model,
            features=[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            labels=[0, 1, 0, 1],
            config=TrainingConfig(
                learning_rate=0.01,
                local_epochs=3,
                batch_size=2,
                seed=7,
            ),
            on_epoch_end=lambda epoch, loss: observed.append((epoch, loss)),
        )

        self.assertEqual([epoch for epoch, _ in observed], [1, 2, 3])
        self.assertAlmostEqual(final_loss, observed[-1][1])


if __name__ == "__main__":
    unittest.main()
