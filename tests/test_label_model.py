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
        np.random.seed(9)

        # Generate unipolar L for single task
        N, M = 10000, 40
        L, Y, metadata = generate_single_task_unipolar(N, M, 
            class_balance=[0.5, 0.5], alpha_range=[0.4, 0.8], beta_range=[0.5])

        cls.single = (L, Y, metadata)

    def test_single_random(self):
        np.random.seed(1)
        L, Y, _ = self.single
        model = RandomVoter()
        model.train(L)
        score = model.score(L, Y, verbose=False)
        self.assertAlmostEqual(score, 0.4963, places=2)

    def test_single_mc(self):
        np.random.seed(1)
        L, Y, metadata = self.single
        balance = metadata['class_balance']
        model = MajorityClassVoter()
        model.train(L, balance)
        score = model.score(L, Y, verbose=False)
        self.assertAlmostEqual(score, 0.5046, places=2)

    def test_single_mv(self):
        np.random.seed(1)
        L, Y, _ = self.single
        model = MajorityLabelVoter()
        model.train(L)
        score = model.score(L, Y, verbose=False, break_ties='abstain')
        self.assertAlmostEqual(score, 0.7416, places=2)

    # def test_single_lm(self):
    #     np.random.seed(1)
    #     L, Y, metadata = self.single
    #     model = LabelModel()
    #     model.train(L, accs=metadata['cond_probs'], verbose=False)
    #     score = model.score(L, Y, verbose=False)
    #     accs_score = model.get_accs_score(metadata['accs'])
    #     self.assertAlmostEqual(score, 0.826, places=2)
    #     self.assertLess(accs_score, 0.001)


if __name__ == '__main__':
    unittest.main()