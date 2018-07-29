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