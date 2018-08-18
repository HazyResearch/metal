import unittest

import numpy as np


class TravisTest(unittest.TestCase):
    def test_sanity(self):
        self.assertTrue(1 + 1 == 2)
        # Confirm import of third-party package also works
        self.assertTrue(int(np.array([1]) + np.array([1])) == 2)


if __name__ == "__main__":
    unittest.main()
