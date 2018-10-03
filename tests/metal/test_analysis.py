import unittest

import numpy as np
import scipy.sparse as sparse

from metal.analysis import (
    error_buckets,
    label_conflict,
    label_coverage,
    label_overlap,
    lf_conflicts,
    lf_coverages,
    lf_empirical_accuracies,
    lf_overlaps,
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

    def test_lf_empirical_accuracies(self):
        self.assertTrue(
            np.all(
                lf_empirical_accuracies(self.L, self.Y) == np.array([0.5, 0, 1])
            )
        )

    def test_lf_coverages(self):
        self.assertTrue(
            (lf_coverages(self.L) == np.array([0.4, 0.4, 0.8])).all()
        )

    def test_lf_overlaps(self):
        self.assertTrue(
            (lf_overlaps(self.L) == np.array([0.4, 0.4, 0.6])).all()
        )

    def test_lf_conflicts(self):
        self.assertTrue(
            (lf_conflicts(self.L) == np.array([0.2, 0.4, 0.4])).all()
        )

    def test_error_buckets(self):
        gold = [1, 1, 2, 1, 2]
        pred = [1, 2, 1, 1, 2]
        e_buckets = error_buckets(gold, pred)
        self.assertEqual(e_buckets[1, 1], [0, 3])
        self.assertEqual(e_buckets[1, 2], [2])
        self.assertEqual(e_buckets[2, 2], [4])
        self.assertEqual(e_buckets[2, 1], [1])


if __name__ == "__main__":
    unittest.main()
