import sys
import unittest

import numpy as np
import torch

from metal.end_model import EndModel, LogisticRegression

class EndModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set seed
        np.random.seed(1)

        N = 1200

        X = np.random.random((N,2)) * 2 - 1
        Y = (X[:,0] > X[:,1] + 0.25).astype(int) + 1

        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.long)

        Xs = [X[:1000], X[1000:1100], X[1100:]]
        Ys = [Y[:1000], Y[1000:1100], Y[1100:]]
        cls.single_problem = (Xs, Ys)

    def test_logreg(self):
        em = LogisticRegression(seed=1, verbose=False, input_dim=2)
        Xs, Ys = self.single_problem
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], verbose=False, n_epochs=10)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 0.99)

    def test_singletask(self):
        em = EndModel(seed=1, verbose=False, 
            layer_output_dims=[2,4,2], batchnorm=False, dropout=0.0)
        Xs, Ys = self.single_problem
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], verbose=False, n_epochs=10)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 1.0)

    def test_singletask_extras(self):
        em = EndModel(seed=1, verbose=False, 
            layer_output_dims=[2,4,2], batchnorm=True, dropout=0.05)
        Xs, Ys = self.single_problem
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], verbose=False, n_epochs=10)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 0.92)
        
        
if __name__ == '__main__':
    unittest.main()        