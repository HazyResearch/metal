import unittest

import numpy as np
import scipy
import torch

from metal.utils import (
    rargmax,
    hard_to_soft,
    recursive_merge_dicts,
)

class UtilsTest(unittest.TestCase):
    def test_rargmax(self):
        x = np.array([2, 1, 2])
        np.random.seed(1)
        self.assertEqual(sorted(list(set(rargmax(x) for _ in range(10)))), [0, 2])

    def test_hard_to_soft(self):
        x = torch.tensor([1,2,2,1])
        target = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
        ], dtype=torch.float)
        self.assertTrue((hard_to_soft(x, 2) == target).sum() 
            == torch.prod(torch.tensor(target.shape)))

    def test_recursive_merge_dicts(self):
        x = {
            'foo': {'Foo': {'FOO': 1}},
            'bar': 2,
            'baz': 3,
        }
        y = {
            'FOO': 4,
            'bar': 5,
        }
        z = {
            'foo': 6
        }
        w = recursive_merge_dicts(x, y, verbose=False)
        self.assertEqual(w['bar'], 5)
        self.assertEqual(w['foo']['Foo']['FOO'], 4)
        with self.assertRaises(ValueError):
            recursive_merge_dicts(x, z, verbose=False)
            
if __name__ == '__main__':
    unittest.main()