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
            self.assertAlmostEqual(sce(Y_p, Y), ce(Y_p, target), places=4)

        sce = SoftCrossEntropyLoss(size_average=True)
        ce = nn.CrossEntropyLoss(size_average=True)
        for _ in range(10):
            self.assertAlmostEqual(sce(Y_p, Y), ce(Y_p, target), places=4)

    def test_perfect_predictions(self):
        # All incorrect predictions
        Y_h = torch.tensor([1, 2, 3], dtype=torch.long)
        target = Y_h - 1
        Y = hard_to_soft(Y_h, k=4)

        sce = SoftCrossEntropyLoss()

        # Guess nearly perfectly
        Y_p = Y.clone()
        Y_p[Y_p == 1] = 100
        Y_p[Y_p == 0] = -100
        self.assertEqual(sce(Y_p, Y), 0)

if __name__ == '__main__':
    unittest.main()