import sys
import unittest

import numpy as np
import scipy.sparse as sparse

sys.path.append("../metal")
from metal.analysis import (
    label_coverage,
    label_overlap,
    label_conflict,
    LF_coverages,
    LF_overlaps,
    LF_conflicts,
    LF_empirical_accuracies,
)

class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = np.array([[1, 0, 1], [1, 3, 2], [0, 0, 0], [0, 0, 2], [0, 1, 2]])
        cls.L = sparse.csr_matrix(L)
        cls.Y = np.array([1, 2, 1, 2, 2])
    
    def test_label_coverage(self):
        self.assertEqual(label_coverage(self.L), 0.8)

    def test_label_overlap(self):
        self.assertEqual(label_overlap(self.L), 0.6)

    def test_label_conflict(self):
        self.assertEqual(label_conflict(self.L), 0.4)

    def test_LF_empirical_accuracies(self):
        self.assertTrue(np.all(
            LF_empirical_accuracies(self.L, self.Y) == np.array([0.5, 0, 1])))

    def test_LF_coverages(self):
        self.assertTrue(
            (LF_coverages(self.L) == np.array([0.4, 0.4, 0.8])).all())

    def test_LF_overlaps(self):
        self.assertTrue(
            (LF_overlaps(self.L) == np.array([0.4, 0.4, 0.6])).all())

    def test_LF_conflicts(self):
        self.assertTrue(
            (LF_conflicts(self.L) == np.array([0.2, 0.4, 0.4])).all())


if __name__ == '__main__':
    unittest.main()