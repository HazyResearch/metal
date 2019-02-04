import os
import unittest

import numpy as np
import torch
import torch.nn as nn

from metal.end_model import EndModel, LogisticRegression
from metal.end_model.identity_module import IdentityModule
from metal.metrics import METRICS


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
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=5, checkpoint=False
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_softmax(self):
        em = LogisticRegression(seed=1, input_dim=2, output_dim=3, verbose=False)
        Xs, _ = self.single_problem
        Ys = []
        for X in Xs:
            class1 = X[:, 0] < X[:, 1]
            class2 = X[:, 0] > X[:, 1] + 0.5
            class3 = X[:, 0] > X[:, 1]
            Y = torch.argmax(torch.stack([class1, class2, class3], dim=1), dim=1) + 1
            Ys.append(Y)
        em.train_model(
            (Xs[0], Ys[0]),
            valid_data=(Xs[1], Ys[1]),
            lr=0.1,
            n_epochs=10,
            checkpoint=False,
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_singletask(self):
        """Test basic single-task end model"""
        em = EndModel(
            seed=1,
            input_batchnorm=False,
            middle_batchnorm=False,
            input_dropout=0.0,
            middle_dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=5, checkpoint=False
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_singletask_extras(self):
        """Test batchnorm and dropout"""
        em = EndModel(
            seed=1,
            input_batchnorm=True,
            middle_batchnorm=True,
            input_dropout=0.01,
            middle_dropout=0.01,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=5, checkpoint=False
        )
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
        em.train_model(
            (Xs[0], Ys[0]),
            valid_data=(Xs[1], Ys[1]),
            n_epochs=5,
            verbose=False,
            checkpoint=False,
            show_plots=False,
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_scoring(self):
        """Test the metrics whole way through"""
        em = EndModel(
            seed=1,
            batchnorm=False,
            dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=5, checkpoint=False
        )
        metrics = list(METRICS.keys())
        scores = em.score((Xs[2], Ys[2]), metric=metrics, verbose=False)
        for i, metric in enumerate(metrics):
            self.assertGreater(scores[i], 0.95)

    def test_determinism(self):
        """Test whether training and scoring is deterministic given seed"""
        em = EndModel(
            seed=123,
            batchnorm=True,
            dropout=0.1,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=1, checkpoint=False
        )
        score_1 = em.score((Xs[2], Ys[2]), verbose=False)

        # Test scoring determinism
        score_2 = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertEqual(score_1, score_2)

        # Test training determinism
        em_2 = EndModel(
            seed=123,
            batchnorm=True,
            dropout=0.1,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        em_2.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=1, checkpoint=False
        )
        score_3 = em_2.score((Xs[2], Ys[2]), verbose=False)
        self.assertEqual(score_1, score_3)

    def test_save_and_load(self):
        """Test basic saving and loading"""
        em = EndModel(
            seed=1337,
            input_batchnorm=False,
            middle_batchnorm=False,
            input_dropout=0.0,
            middle_dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=3, checkpoint=False
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)

        # Save model
        SAVE_PATH = "test_save_model.pkl"
        em.save(SAVE_PATH)

        # Reload and make sure (a) score and (b) non-buffer, non-Parameter
        # attributes are the same
        em_2 = EndModel.load(SAVE_PATH)
        self.assertEqual(em.seed, em_2.seed)
        score_2 = em_2.score((Xs[2], Ys[2]), verbose=False)
        self.assertEqual(score, score_2)

        # Clean up
        os.remove(SAVE_PATH)


if __name__ == "__main__":
    unittest.main()
