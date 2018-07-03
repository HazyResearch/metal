from collections import Counter
import sys
import unittest

import numpy as np

sys.path.append("../metal")
from metal.metrics import (
    accuracy_score,
    coverage_score,
)

from synthetics import (
    generate_single_task_unipolar,
)

class SyntheticsTest(unittest.TestCase):

    def check_L(self, L, n, m, k):
        self.assertTrue(L.shape[0] == n)
        self.assertTrue(L.shape[1] == m)
        self.assertTrue(L.min() >= 0)
        self.assertTrue(L.max() <= k)
    
    def check_class_balance(self, L, Y, class_balance):
        counter = Counter(Y.numpy())
        class_counts = np.array([v for k, v in sorted(counter.items())])
        emp_class_balance = class_counts / sum(class_counts)
        for a, b in zip(class_balance, emp_class_balance):
            self.assertLess(abs(a - b), 0.01)

    def check_accuracies(self, L, Y, m, accs):
        emp_accs = []
        for j in range(m):
            emp_accs.append(accuracy_score(Y, L[:,j], ignore_in_pred=[0]))
        for a, b in zip(accs, emp_accs):
            self.assertLess(abs(a - b), 0.01)

    def check_recalls(self, L, Y, m, k, recs, polarities):
        emp_recs = []
        for j in range(m):
            other_labels = [x for x in range(k + 1) if x != polarities[j]]
            emp_recs.append(coverage_score(Y, L[:,j], ignore_in_gold=other_labels))
        for a, b in zip(recs, emp_recs):
            self.assertLess(abs(a - b), 0.01)

    # def check_all(self, L, Y, k, alphas, betas, polarities, coverages, 
    #     cond_probs):
    #     n, m = L.shape
    #     self.check_class_balance(L, Y, class_balance)
    #     self.check_accuracies(L, Y, m, accs)
    #     self.check_recalls(L, Y, m, k, recs, polarities)

        # 'k': k,
        # 'alphas': alphas,
        # 'betas': betas,
        # 'polarities': polarities,
        # 'coverages': np.where(L.todense() != 0, 1, 0).sum(axis=0) / n,
        # 'cond_probs': alphas * betas

    # def test_single_unipolar(self):
    #     n = 1000
    #     m = 10
    #     L, Y, metadata = generate_single_task_unipolar(
    #         n, m, k=2, alpha_range=[0.6, 0.9], beta_range=[0.1, 0.2], 
    #         class_balance=None, seed=1)
    #     self.check_all(L, Y, **metadata)

    # def test_single_unipolar_nonbinary(self):
    #     n = 10000
    #     m = 50
    #     L, Y, metadata = generate_single_task_unipolar(
    #         n, m, k=5, alpha_range=[0.6, 0.9], beta_range=[0.1, 0.2], 
    #         class_balance=None, seed=1)
    #     self.check_all(L, Y, **metadata)

    # def test_single_unipolar_imbalanced(self):
    #     n = 10000
    #     m = 10
    #     L, Y, metadata = generate_single_task_unipolar(
    #         n, m, k=2, alpha_range=[0.6, 0.9], beta_range=[0.1, 0.2], 
    #         class_balance=[0.7, 0.3], seed=1)
    #     self.check_all(L, Y, **metadata)


if __name__ == '__main__':
    unittest.main()