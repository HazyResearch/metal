import sys
import unittest

import numpy as np
import torch

sys.path.append("../metal")
from metal.utils import (
    rargmax,
    hard_to_soft,
)

class UtilsTest(unittest.TestCase):
    def test_rargmax(self):
        x = np.array([2, 1, 2])
        self.assertEqual(sorted(list(set(rargmax(x) for _ in range(10)))), [0, 2])

    def test_hard_to_soft(self):
        x = torch.tensor([1,2,2,1])
        target = torch.tensor([
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
        ], dtype=torch.float)
        self.assertTrue(((hard_to_soft(x, 2) == target).sum() == 8))


if __name__ == '__main__':
    unittest.main()