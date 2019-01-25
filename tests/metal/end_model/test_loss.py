import unittest

import torch
import torch.nn as nn

from metal.end_model.loss import SoftCrossEntropyLoss
from metal.utils import hard_to_soft


class LossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)

    def test_sce_equals_ce(self):
        # All correct predictions
        Y = torch.tensor([1, 2, 3], dtype=torch.long)
        Y_s = hard_to_soft(Y, k=4).float()

        sce = SoftCrossEntropyLoss(reduction="none")
        ce = nn.CrossEntropyLoss(reduction="none")
        for _ in range(10):
            Y_ps = torch.rand_like(Y_s)
            Y_ps = Y_ps / Y_ps.sum(dim=1).reshape(-1, 1)
            self.assertTrue((sce(Y_ps, Y_s) == ce(Y_ps, Y - 1)).all())

        sce = SoftCrossEntropyLoss(reduction="sum")
        ce = nn.CrossEntropyLoss(reduction="sum")
        for _ in range(10):
            Y_ps = torch.rand_like(Y_s)
            Y_ps = Y_ps / Y_ps.sum(dim=1).reshape(-1, 1)
            self.assertAlmostEqual(
                sce(Y_ps, Y_s).numpy(), ce(Y_ps, Y - 1).numpy(), places=5
            )

        sce = SoftCrossEntropyLoss(reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")
        for _ in range(10):
            Y_ps = torch.rand_like(Y_s)
            Y_ps = Y_ps / Y_ps.sum(dim=1).reshape(-1, 1)
            self.assertAlmostEqual(
                sce(Y_ps, Y_s).numpy(), ce(Y_ps, Y - 1).numpy(), places=5
            )

    def test_perfect_predictions(self):
        Y = torch.tensor([1, 2, 3], dtype=torch.long)
        Y_s = hard_to_soft(Y, k=4)

        sce = SoftCrossEntropyLoss()
        # Guess nearly perfectly
        Y_ps = Y_s.clone().float()
        Y_ps[Y_ps == 1] = 100
        Y_ps[Y_ps == 0] = -100
        self.assertAlmostEqual(sce(Y_ps, Y_s).numpy(), 0)

    def test_soft_labels(self):
        Y_s = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
        Y_ps1 = torch.tensor([[0.1, 0.2], [1.0, 0.0]])
        Y_ps2 = torch.tensor([[0.1, 0.3], [1.0, 0.0]])
        Y_ps3 = torch.tensor([[0.1, 0.3], [0.0, 1.0]])
        sce = SoftCrossEntropyLoss()
        self.assertLess(sce(Y_ps2, Y_s), sce(Y_ps1, Y_s))
        self.assertEqual(sce(Y_ps2, Y_s), sce(Y_ps3, Y_s))

    def test_loss_weights(self):
        # All incorrect predictions
        Y = torch.tensor([1, 1, 2], dtype=torch.long)
        Y_s = hard_to_soft(Y, k=3)
        Y_ps = torch.tensor(
            [
                [-100.0, 100.0, -100.0],
                [-100.0, 100.0, -100.0],
                [-100.0, 100.0, -100.0],
            ]
        )
        weight1 = torch.tensor([1, 2, 1], dtype=torch.float)
        weight2 = torch.tensor([10, 20, 10], dtype=torch.float)
        ce1 = nn.CrossEntropyLoss(weight=weight1, reduction="none")
        sce1 = SoftCrossEntropyLoss(weight=weight1)
        sce2 = SoftCrossEntropyLoss(weight=weight2)

        self.assertAlmostEqual(
            float(ce1(Y_ps, Y - 1).mean()), float(sce1(Y_ps, Y_s)), places=3
        )
        self.assertAlmostEqual(
            float(sce1(Y_ps, Y_s)) * 10, float(sce2(Y_ps, Y_s)), places=3
        )


if __name__ == "__main__":
    unittest.main()
