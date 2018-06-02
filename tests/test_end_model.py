import sys
import unittest

import numpy as np
import torch

sys.path.append("../metal")
from metal.end_model import EndModel
from metal.utils import hard_to_soft

class EndModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set seed
        np.random.seed(1)

        N = 1200
        X = np.random.random((N,2)) * 2 - 1
        Y = np.zeros((N,2))
        Y[:,0] = (X[:,0] > X[:,1] + 0.5).astype(int) + 1
        Y[:,1] = (X[:,0] > X[:,1] + 0.25).astype(int) + 1

        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.short)
        
        cls.X_train = X[:1000]
        cls.X_dev   = X[1000:1100]
        cls.X_test  = X[1100:]

        cls.Y_train = [Y[:1000, 0],     Y[:1000, 1]]
        cls.Y_dev   = [Y[1000:1100, 0], Y[1000:1100, 1]]
        cls.Y_test  = [Y[1100:, 0],     Y[1100:, 1]]

        cls.Y_train = [hard_to_soft(Y_t, k=2) for Y_t in cls.Y_train]

    def test_single_task(self):
        em = EndModel(seed=1)
        Y_train = self.Y_train[0]
        Y_dev = self.Y_dev[0]
        Y_test = self.Y_test[0]
        em.train(self.X_train, Y_train, self.X_dev, Y_dev, 
            n_epochs=10,
            dropout=0.0,
            layer_output_dims=[2, 4, 2],
        )
        score = em.score(self.X_test, Y_test, reduce='mean')
        self.assertEqual(score, 0.74)


if __name__ == '__main__':
    unittest.main()        