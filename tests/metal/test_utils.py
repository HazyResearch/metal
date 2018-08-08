import unittest

import numpy as np
import scipy
import torch

from metal.utils import (
    rargmax,
    hard_to_soft,
    recursive_merge_dicts,
    split_data
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
        
    def test_split_data(self):
        # Creating data
        X = np.random.randint(0,100, size=100000)
        Y = np.random.randint(0,4,size=100000).astype(int)

        # Creating splits of correct size
        splits = [100, 1000, 100000-1000-100]
        data_out, labels_out, split_list = split_data(X, splits, input_labels=Y, shuffle=True, stratify=None, seed=None)
        out_dim = [len(data_out[0]), len(data_out[1]), len(data_out[2])]
        self.assertTrue(np.array_equal(out_dim, splits))

        # Checking functionality with fractional arguments
        splits_frac = [float(a)/np.sum(splits) for a in splits]
        data_out, labels_out, split_list = split_data(X, splits_frac, input_labels=Y, shuffle=True, stratify=None, seed=None)
        out_dim = [len(data_out[0]), len(data_out[1]), len(data_out[2])]
        self.assertTrue(np.array_equal(out_dim, splits))

        #Checking to make sure that we've actually shuffled!
        self.assertTrue(X[0] != data_out[0][0])

        # Turning off shuffling
        splits_frac = [float(a)/np.sum(splits) for a in splits]
        data_out, labels_out, split_list = split_data(X, splits_frac, input_labels=Y, shuffle=False, stratify=None, seed=None)
        out_dim = [len(data_out[0]), len(data_out[1]), len(data_out[2])]
        # Making sure we haven't shuffled!
        self.assertTrue(X[0] == data_out[0][0])

        # Testing stratification -- making sure proportion of label `test_label` is constant in splits!
        test_label = 3
        splits_frac = [float(a)/np.sum(splits) for a in splits]
        data_out, labels_out, split_list = split_data(X, splits_frac, input_labels=Y, shuffle=True, stratify=True, seed=1701)
        out_dim = [len(data_out[0]), len(data_out[1]), len(data_out[2])]
        props = [np.sum([b==test_label for b in labels_out[a]])/len(labels_out[a]) for a in range(len(labels_out))]
        self.assertTrue(np.max([np.abs(a-props[0]) for a in props])<0.01)

if __name__ == '__main__':
    unittest.main()