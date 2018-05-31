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

    def test_loss1(self):
        # All correct predictions
        Y_h = torch.tensor([1, 2, 3], dtype=torch.long)
        target = Y_h - 1
        # Hard to soft converts from 1-index labels to 0-index probabilities
        Y = hard_to_soft(Y_h, k=4)
        Y_p = Y.clone()
        sce = SoftCrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        # CrossEntropyLoss expects 0-indexed labels
        self.assertEqual(sce(Y_p, Y), ce(Y_p, target))

    def test_loss2(self):
        # All incorrect predictions
        Y_h = torch.tensor([1, 2, 3], dtype=torch.long)
        target = Y_h - 1
        Y = hard_to_soft(Y_h, k=4)
        # Always guess one class too high
        Y_p = Y.clone() + 1
        sce = SoftCrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        self.assertEqual(sce(Y_p, Y), ce(Y_p, target))

    def test_loss3(self):
        # Partially correct predictions
        Y_h = torch.tensor([1, 2, 3], dtype=torch.long)
        target = Y_h - 1
        Y_right = hard_to_soft(Y_h, k=4)
        Y_wrong = hard_to_soft(Y_h + 1, k=4)
        # Guess uniform distribution
        Y_p = torch.full_like(Y_right, 1/4)
        sce = SoftCrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        # Guessing uniform distribution should be worse than guessing perfectly 
        # and better than guessing all wrong
        self.assertGreater(sce(Y_p, Y_right), ce(Y_right, target))
        self.assertLess(sce(Y_p, Y_right), ce(Y_wrong, target))

if __name__ == '__main__':
    unittest.main()