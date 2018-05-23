import numpy as np
import unittest

class TravisTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_sanity(self):
        self.assertTrue(1 + 1 == 2)
        # Confirm import of third-party package also works
        self.assertTrue(int(np.array([1]) + np.array([1])) == 2)

if __name__ == '__main__':
    unittest.main()