import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.append("../metal")
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.utils import (
    hard_to_soft,
)

class LossTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)

    def test_sce_equals_ce(self):
        # All correct predictions
        Y_h = torch.tensor([1, 2, 3], dtype=torch.long)
        target = Y_h - 1  # Account for CrossEntropyLoss expecting 0-index
        Y = hard_to_soft(Y_h, k=4)  # hard_to_soft converts to 0-index for us
        
        sce = SoftCrossEntropyLoss(reduce=False)
        ce = nn.CrossEntropyLoss(reduce=False)
        for _ in range(10):
            Y_p = torch.randn(Y.shape)
            self.assertTrue((sce(Y_p, Y) == ce(Y_p, target)).all())

        sce = SoftCrossEntropyLoss(size_average=False)
        ce = nn.CrossEntropyLoss(size_average=False)
        for _ in range(10):
            self.assertAlmostEqual(sce(Y_p, Y).numpy(), ce(Y_p, target).numpy(),
                places=5)

        sce = SoftCrossEntropyLoss(size_average=True)
        ce = nn.CrossEntropyLoss(size_average=True)
        for _ in range(10):
            self.assertAlmostEqual(sce(Y_p, Y).numpy(), ce(Y_p, target).numpy(),
            places=5)

    def test_perfect_predictions(self):
        Y_h = torch.tensor([1, 2, 3], dtype=torch.long)
        target = Y_h - 1
        Y = hard_to_soft(Y_h, k=4)

        sce = SoftCrossEntropyLoss()
        # Guess nearly perfectly
        Y_p = Y.clone()
        Y_p[Y_p == 1] = 100
        Y_p[Y_p == 0] = -100
        self.assertAlmostEqual(sce(Y_p, Y).numpy(), 0)

    def test_soft_labels(self):
        Y = torch.tensor([
            [0.1, 0.9],
            [0.5, 0.5],
        ])
        Y_p1 = torch.tensor([
            [0.1, 0.2],
            [1.0, 0.0],
        ])
        Y_p2 = torch.tensor([
            [0.1, 0.3],
            [1.0, 0.0],
        ])
        Y_p3 = torch.tensor([
            [0.1, 0.3],
            [0.0, 1.0],
        ])
        sce = SoftCrossEntropyLoss()
        self.assertLess(sce(Y_p2, Y), sce(Y_p1, Y))
        self.assertEqual(sce(Y_p2, Y), sce(Y_p3, Y))

    def test_loss_weights(self):
        # All incorrect predictions
        Y_h = torch.tensor([1,1,2], dtype=torch.long)
        target = Y_h - 1
        K_t = 2
        Y = hard_to_soft(Y_h, k=K_t)
        Y_p1 = hard_to_soft(torch.tensor([1,1,1]), k=K_t)
        Y_p1[Y_p1 == 0] = -100
        Y_p1[Y_p1 == 1] = 100
        Y_p2 = hard_to_soft(torch.tensor([2,2,2]), k=K_t)
        Y_p2[Y_p2 == 0] = -100
        Y_p2[Y_p2 == 1] = 100
        
        print(Y_p1)
        print(Y_p2)

        weight1 = torch.tensor([1,1], dtype=torch.float)
        weight2 = torch.tensor([1,1], dtype=torch.float) * 10
        ce1 = nn.CrossEntropyLoss(weight=weight1)
        ce2 = nn.CrossEntropyLoss(weight=weight2)
        # sce1 = SoftCrossEntropyLoss(weight=weight1)
        # sce2 = SoftCrossEntropyLoss(weight=weight2)

        loss1 = 
        loss2 = 
        print(ce1(Y_p1, target))
        print(ce2(Y_p1, target))
        # print()
        # print(sce1(Y_p1, Y))
        # print(sce2(Y_p2, Y))

        import pdb; pdb.set_trace()

if __name__ == '__main__':
    unittest.main()