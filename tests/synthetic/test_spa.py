import sys
import unittest

import numpy as np

from synthetic.generate_spa import ClusterDependencies, DataGenerator

sys.path.append("../synthetic")


class SPAClusterDependenciesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 1000  # Note: Not currently used
        cls.m = 5
        cls.n_clusters = 3
        cls.k = 2
        cls.abstains = False
        cls.n_iter = 3
        cls.seed = 123

        # Create dependencies graph
        np.random.seed(cls.seed)
        cls.deps_graph = ClusterDependencies(cls.m, cls.n_clusters)

        # Create data generator
        cls.data = DataGenerator(
            cls.n,
            cls.m,
            k=cls.k,
            abstains=cls.abstains,
            deps_graph=cls.deps_graph,
        )

    def test_spa_acc(self):
        """Test accuracy computation P(\lf_i, Y)."""
        np.random.seed(self.seed)
        for _ in range(self.n_iter):
            i = np.random.choice(self.m)
            for query in self.data.iter_vals([i, self.m]):
                self.assertAlmostEqual(
                    self.data.P_marginal(query),
                    self.data.P_marginal_brute_force(query),
                    places=10,
                )

    def test_spa_overlap(self):
        """Test overlap computation P(\lf_i, \lf_j)."""
        np.random.seed(self.seed)
        for _ in range(self.n_iter):
            i, j = np.random.choice(self.m, 2)
            for query in self.data.iter_vals([i, j]):
                self.assertAlmostEqual(
                    self.data.P_marginal(query),
                    self.data.P_marginal_brute_force(query),
                    places=10,
                )

    def test_spa_acc_pair(self):
        """Test overlap computation P(\lf_i, \lf_j, Y)."""
        np.random.seed(self.seed)
        for _ in range(self.n_iter):
            i, j = np.random.choice(self.m, 2)
            for query in self.data.iter_vals([i, j, self.m]):
                self.assertAlmostEqual(
                    self.data.P_marginal(query),
                    self.data.P_marginal_brute_force(query),
                    places=10,
                )

    def test_spa_overlap_triple(self):
        """Test overlap computation P(\lf_i, \lf_j, \lf_k)."""
        np.random.seed(self.seed)
        for _ in range(self.n_iter):
            i, j, k = np.random.choice(self.m, 3)
            for query in self.data.iter_vals([i, j, k]):
                self.assertAlmostEqual(
                    self.data.P_marginal(query),
                    self.data.P_marginal_brute_force(query),
                    places=10,
                )


if __name__ == "__main__":
    unittest.main()
