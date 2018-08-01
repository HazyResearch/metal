import sys
import unittest

import numpy as np
import torch

from metal.label_model.label_model import LabelModel
from metal.label_model.baselines import (
    RandomVoter,
    MajorityClassVoter,
    MajorityLabelVoter,
)

sys.path.append("../synthetics")
from synthetics.generate import SingleTaskTreeDepsGenerator


# TODO: Put in tests for LabelModel baseline again!
class LabelModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10000
        cls.m = 10
        cls.k = 2
    
    def _test_label_model(self, data, test_acc=True):
        label_model = LabelModel(data.m, data.k, p=data.p, deps=data.E)
        label_model.train(data.L, n_epochs=500, print_every=100)
        
        # Test parameter estimation error
        c_probs_est = label_model.get_conditional_probs()
        err = np.linalg.norm(data.c_probs - c_probs_est)**2
        print(f"Parameter Estimation Error={err}")
        self.assertLess(err, 0.01)

        # Test label prediction accuracy
        if test_acc:
            Y_pred = label_model.get_label_probs(data.L).argmax(axis=1) + 1
            acc = np.where(data.Y == Y_pred, 1, 0).sum() / data.n
            print(f"Label Prediction Accuracy={acc}")
            self.assertGreater(acc, 0.95)
    
    def test_no_deps(self):
        # Test for 5 random seeds
        for seed in range(5):
            np.random.seed(seed)
            print(f">>> Testing for seed={seed}")
            data = SingleTaskTreeDepsGenerator(self.n, self.m, k=self.k, 
                edge_prob=0.0)
            self._test_label_model(data)

    def test_augmented_L_construction(self):
        # 5 LFs: a triangle, a connected edge to it, and a singleton source
        n = 3
        m = 5
        k = 2
        E = [(0,1), (1,2), (2,0), (0,3)]
        L = np.array([
            [1, 1, 1, 2, 1],
            [1, 2, 2, 1, 0],
            [1, 1, 1, 1, 0]
        ])
        lm = LabelModel(m, k=k, deps=E)
        L_aug = lm._get_augmented_label_matrix(L, offset=1, higher_order=True)

        # Should have 22 columns:
        # - 5 * 2 = 10 for the sources
        # - 8 + 4 for the 3- and 2-clique resp. --> = 22
        self.assertEqual(L_aug.shape, (3,22))

        # Same as above but minus 2 abstains = 19 total nonzero entries
        self.assertEqual(L_aug.sum(), 19)

        # Next, check the singleton entries
        for i in range(n):
            for j in range(m):
                if L[i,j] > 0:
                    self.assertEqual(L_aug[i, j * k + L[i,j] - 1], 1)

        # Finally, check the clique entries
        # Triangle clique
        self.assertEqual(len(lm.c_tree.node[1]['members']), 3)
        j = lm.c_tree.node[1]['start_index']
        self.assertEqual(L_aug[0, j], 1)
        self.assertEqual(L_aug[1, j + 3], 1)
        self.assertEqual(L_aug[2, j], 1)
        # Binary clique
        self.assertEqual(len(lm.c_tree.node[2]['members']), 2)
        j = lm.c_tree.node[2]['start_index']
        self.assertEqual(L_aug[0, j+1], 1)
        self.assertEqual(L_aug[1, j], 1)
        self.assertEqual(L_aug[2, j], 1)
    
    def test_with_deps(self):
        # Test for 5 random seeds
        for seed in range(5):
            np.random.seed(seed)
            print(f">>> Testing for seed={seed}")
            data = SingleTaskTreeDepsGenerator(self.n, self.m, k=self.k, 
                edge_prob=1.0)
            self._test_label_model(data, test_acc=False)


if __name__ == '__main__':
    unittest.main()