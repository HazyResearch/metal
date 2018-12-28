import sys
import unittest
from itertools import product

import numpy as np
import torch

from metal.label_model.class_balance import ClassBalanceModel

sys.path.append("../synthetic")


# TODOs:
# (3) Noisy tests- starting from L


class ClassBalanceModelTest(unittest.TestCase):
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _generate_class_balance(self, k):
        """Generate class balance"""
        p_Y = np.random.random(k)
        p_Y /= p_Y.sum()
        return p_Y

    def _generate_cond_probs(self, k, m, bias_diag=True, abstains=False):
        """Generate conditional probability tables for the m conditionally ind.
        LFs, such that:
            cpts[i, y1, y2] = P(\lambda_i = y1 | Y = y2)

        Args:
            k: (int) Number of classes
            m: (int) Number of LFs
            bias_diag: (bool) If True, adds a bias (proportional to (k-1)) to
                the diagonal of the randomly generated conditional probability
                tables, to enforce assumption that LFs are better than random
            abstains: (bool) Incorporate abstains

        Outputs:
            C: (np.array) An (m, k, k) tensor, if abstains=False; or, if
                abstains=True, (m, k+1, k)
        """
        cpts = []
        k_lf = k + 1 if abstains else k
        for i in range(m):
            a = np.random.random((k_lf, k))
            if bias_diag:
                if abstains:
                    a[1:, :] += (k - 1) * np.eye(k)
                else:
                    a += (k - 1) * np.eye(k)
            cpts.append(a @ np.diag(1 / a.sum(axis=0)))
        return np.array(cpts)

    def _test_class_balance_estimation(
        self, k, m, abstains=False, verbose=True
    ):
        model = ClassBalanceModel(k, abstains=abstains)

        # Generate the true class balance
        p_Y = self._generate_class_balance(k)

        # Generate the true conditional probability tables for the m
        # conditionally independent LFs
        # Note: We bias these along the diagonal to enforce assumption that the
        # LFs are better than random!
        C = self._generate_cond_probs(k, m, bias_diag=True, abstains=abstains)

        # Compute O; mask out diagonal entries
        mask = model.get_mask(m)
        O = np.einsum("aby,cdy,efy,y->acebdf", C, C, C, p_Y)
        O = torch.from_numpy(O).float()
        O[1 - mask] = 0

        # Test recovery of the class balance
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

    def test_class_balance_estimation_2_abstains(self):
        self._set_seed(123)
        self._test_class_balance_estimation(2, 25, abstains=True)


if __name__ == "__main__":
    unittest.main()
