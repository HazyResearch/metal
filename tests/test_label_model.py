import sys
import unittest

import numpy as np
import torch

sys.path.append("../metal")
from metal.label_model.label_model import LabelModel
from metal.label_model.baselines import (
    RandomVoter,
    MajorityClassVoter,
    MajorityLabelVoter,
)

sys.path.append("../synthetics")
from synthetics.generate import generate_single_task_unipolar


class LabelModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set seed
        np.random.seed(1)

        # Generate unipolar L for single task
        N, M = 10000, 20
        L, Y, metadata = generate_single_task_unipolar(N, M, acc=[0.4, 0.8], 
            rec=[0.5])

        cls.single = (L, Y, metadata)

    def test_single_random(self):
        L, Y, _ = self.single
        model = RandomVoter()
        model.train(L)
        score = model.score(L, Y, verbose=False)
        self.assertAlmostEqual(score, 0.499, places=2)

    def test_single_mc(self):
        L, Y, metadata = self.single
        balances = [metadata['class_balance']]
        model = MajorityClassVoter()
        model.train(L, balances)
        score = model.score(L, Y, verbose=False)
        self.assertAlmostEqual(score, 0.496, places=2)

    def test_single_mv(self):
        L, Y, _ = self.single
        model = MajorityLabelVoter()
        model.train(L)
        score = model.score(L, Y, verbose=False)
        self.assertAlmostEqual(score, 0.787, places=2)

    def test_single_lm(self):
        L, Y, metadata = self.single
        model = LabelModel()
        model.train(L, accs=metadata['accs'], verbose=False)
        score = model.score(L, Y, verbose=False)
        accs_score = model.get_accs_score(metadata['accs'])
        self.assertAlmostEqual(score, 0.826, places=2)
        self.assertLess(accs_score, 0.001)


if __name__ == '__main__':
    unittest.main()