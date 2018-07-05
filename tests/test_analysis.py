import sys
import unittest

import numpy as np
import scipy.sparse as sparse

sys.path.append("../metal")
from metal.analysis import (
    item_coverage,
    item_overlap,
    item_conflict,
    LF_accuracies,
    LF_coverages,
    LF_overlaps,
    LF_conflicts,
)

class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = np.array([[1, 0, 1], [1, 3, 2], [0, 0, 0], [0, 0, 2], [0, 1, 2]])
        cls.L = sparse.csr_matrix(L)
    
    def test_item_coverage(self):
        self.assertEqual(item_coverage(self.L), 0.8)

    def test_item_overlap(self):
        self.assertEqual(item_overlap(self.L), 0.6)

    def test_item_conflict(self):
        self.assertEqual(item_conflict(self.L), 0.4)

    def test_LF_accuracies(self):
        pass

    def test_LF_coverages(self):
        self.assertTrue((LF_coverages(self.L) == np.array([0.4, 0.4, 0.8])).all())

    def test_LF_overlaps(self):
        self.assertTrue((LF_overlaps(self.L) == np.array([0.4, 0.4, 0.6])).all())

    def test_LF_conflicts(self):
        self.assertTrue((LF_conflicts(self.L) == np.array([0.2, 0.4, 0.4])).all())


if __name__ == '__main__':
    unittest.main()