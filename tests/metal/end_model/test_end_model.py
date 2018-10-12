import unittest

import numpy as np
import torch
import torch.nn as nn

from metal.end_model import (
    EndModel,
    LogisticRegression,
    SparseLogisticRegression,
)
from metal.modules import IdentityModule


class EndModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set seed
        np.random.seed(1)

        n = 2000

        X = np.random.random((n, 2)) * 2 - 1
        Y = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.long)

        Xs = [X[:1000], X[1000:1500], X[1500:]]
        Ys = [Y[:1000], Y[1000:1500], Y[1500:]]
        cls.single_problem = (Xs, Ys)

    def test_logreg(self):
        em = LogisticRegression(seed=1, input_dim=2, verbose=False)
        Xs, Ys = self.single_problem
        em.train((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_softmax(self):
        em = LogisticRegression(
            seed=1, input_dim=2, output_dim=3, verbose=False
        )
        Xs, _ = self.single_problem
        Ys = []
        for X in Xs:
            class1 = X[:, 0] < X[:, 1]
            class2 = X[:, 0] > X[:, 1] + 0.5
            class3 = X[:, 0] > X[:, 1]
            Y = (
                torch.argmax(
                    torch.stack([class1, class2, class3], dim=1), dim=1
                )
                + 1
            )
            Ys.append(Y)
        em.train((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), lr=0.1, n_epochs=10)
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_sparselogreg(self):
        """Confirm sparse logreg can overfit, works on padded data"""
        F = 1000  # total number of possible features
        N = 50  # number of data points
        S = [10, 100]  # range of features per data point

        X = np.zeros((N, S[1]))
        for i in range(N):
            Si = np.random.randint(S[0], S[1])
            X[i, :Si] = np.random.randint(F, size=(1, Si))
        X = torch.from_numpy(X).long()
        Y = torch.from_numpy(np.random.randint(1, 3, size=(N,)))

        em = SparseLogisticRegression(
            seed=1, input_dim=F, padding_idx=0, verbose=False
        )
        em.train((X, Y), n_epochs=5, optimizer="sgd", lr=0.0005)
        self.assertEqual(float(em.network[-1].W.weight.data[0, :].sum()), 0.0)
        score = em.score((X, Y), verbose=False)
        self.assertGreater(score, 0.95)

    def test_singletask(self):
        """Test basic single-task end model"""
        em = EndModel(
            seed=1,
            batchnorm=False,
            dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_singletask_extras(self):
        """Test batchnorm and dropout"""
        em = EndModel(
            seed=1,
            batchnorm=True,
            dropout=0.01,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_custom_modules(self):
        """Test custom input/head modules"""
        input_module = nn.Sequential(IdentityModule(), nn.Linear(2, 10))
        middle_modules = [nn.Linear(10, 8), IdentityModule()]
        head_module = nn.Sequential(nn.Linear(8, 2), IdentityModule())
        em = EndModel(
            seed=1,
            input_module=input_module,
            middle_modules=middle_modules,
            head_module=head_module,
            layer_out_dims=[10, 8, 8],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train(
            (Xs[0], Ys[0]),
            dev_data=(Xs[1], Ys[1]),
            n_epochs=5,
            verbose=False,
            show_plots=False,
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
