import unittest

import numpy as np
import scipy
import torch

from metal.utils import (
    rargmax,
    hard_to_soft,
    recursive_merge_dicts,
    make_unipolar_matrix
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
            
    def test_make_unipolar_matrix(self):
        a = np.ones((5,1))
        b = a*2
        c = a*3
        d = a*0
        col = np.vstack([a,b,c,d])
        mat = np.hstack([col,col])
        mat_up = make_unipolar_matrix(scipy.sparse.csr_matrix(mat)).todense()
        self.assertTrue(np.array_equal(mat_up[:,0]+mat_up[:,1]+mat_up[:,2], col))
        self.assertTrue(mat_up.shape[1] == 6)
        
if __name__ == '__main__':
    unittest.main()