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
    
    def _test_label_model(self, data):
        label_model = LabelModel(data.p, data.m, deps=data.E)
        label_model.train(data.L, n_epochs=500, print_every=100)
        c_probs_est = label_model.get_conditional_probs()
        err = np.linalg.norm(data.c_probs - c_probs_est)**2
        print(f"Err={err}")
        self.assertLess(err, 0.01)
    
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
            self._test_label_model(data)


if __name__ == '__main__':
    unittest.main()