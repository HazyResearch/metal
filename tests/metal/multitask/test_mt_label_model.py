import sys
import unittest

import numpy as np

from metal.multitask import MTLabelModel
from synthetic.generate import HierarchicalMultiTaskTreeDepsGenerator

sys.path.append("../synthetic")


class MTLabelModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_iters = 1
        cls.n = 10000
        cls.m = 10

    def _test_label_model(self, data, test_acc=True):
        label_model = MTLabelModel(task_graph=data.task_graph, verbose=False)
        label_model.train(
            data.L,
            deps=data.E,
            class_balance=data.p,
            n_epochs=1000,
            print_every=200,
        )

        # Test parameter estimation error
        c_probs_est = label_model.get_conditional_probs()
        err = np.mean(np.abs(data.c_probs - c_probs_est))
        self.assertLess(err, 0.025)

        # Test label prediction accuracy
        if test_acc:
            acc = label_model.score((data.L, data.Y))
            self.assertGreater(acc, 0.95)

    def test_multitask(self):
        for seed in range(self.n_iters):
            np.random.seed(seed)
            data = HierarchicalMultiTaskTreeDepsGenerator(
                self.n, self.m, edge_prob=0.0
            )
            self._test_label_model(data, test_acc=True)


if __name__ == "__main__":
    unittest.main()
