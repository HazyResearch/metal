import sys
import unittest
from itertools import product

import numpy as np
import torch

from metal.label_model.class_balance import ClassBalanceModel

sys.path.append("../synthetic")


# TODOs:
# (2) Add abstains
# (3) Noisy tests- starting from L


class ClassBalanceModelTest(unittest.TestCase):
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _test_class_balance_estimation(self, k, m, verbose=True):

        # Generate the true class balance
        p_Y = np.random.random(k)
        p_Y /= p_Y.sum()

        # Generate the true conditional probability tables for the m
        # conditionally independent LFs
        # Note: We bias these along the diagonal to enforce assumption that the
        # LFs are better than random!
        alphas = []
        for i in range(m):
            a = np.random.random((k, k)) + (k - 1) * np.eye(k)
            alphas.append(a @ np.diag(1 / a.sum(axis=0)))
        alpha = np.array(alphas)

        # Compute O; mask out diagonal entries
        mask = torch.ones((m, m, m, k, k, k)).byte()
        for a, b, c in product(range(m), repeat=3):
            if len(set((a, b, c))) < 3:
                mask[a, b, c, :, :, :] = 0
        O = np.einsum("aby,cdy,efy,y->acebdf", alpha, alpha, alpha, p_Y)
        O = torch.from_numpy(O).float()
        O[1 - mask] = 0

        # Test recovery of the class balance
        model = ClassBalanceModel(k)
        model.train_model(O=O)
        if verbose:
            print(f"True class balance: {p_Y}")
            print(f"Estimated class balance: {model.class_balance}")
        self.assertLess(np.mean(np.abs(p_Y - model.class_balance)), 1e-3)

    def test_class_balance_estimation_2(self):
        self._set_seed(123)
        self._test_class_balance_estimation(2, 25)

    def test_class_balance_estimation_3(self):
        self._set_seed(123)
        self._test_class_balance_estimation(3, 25)

    # Note: This should pass! However, commented out because too slow...
    # def test_class_balance_estimation_5(self):
    #     self._set_seed(123)
    #     self._test_class_balance_estimation(5, 25)


if __name__ == "__main__":
    unittest.main()
