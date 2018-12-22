import copy
import json
import os
import unittest
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn

from metal.end_model import (
    EndModel,
    LogisticRegression,
    SparseLogisticRegression,
)
from metal.metrics import METRICS
from metal.modules import IdentityModule
from metal.utils import LogWriter


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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
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
        em.train_model(
            (Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), lr=0.1, n_epochs=10
        )
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
        em.train_model((X, Y), n_epochs=5, optimizer="sgd", lr=0.0005)
        self.assertEqual(float(em.network[-1].W.weight.data[0, :].sum()), 0.0)
        score = em.score((X, Y), verbose=False)
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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
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
            dev_data=(Xs[1], Ys[1]),
            n_epochs=5,
            verbose=False,
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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
        metrics = list(METRICS.keys())
        scores = em.score((Xs[2], Ys[2]), metric=metrics, verbose=False)
        for i, metric in enumerate(metrics):
            self.assertGreater(scores[i], 0.95)

    def test_checkpointing(self):
        """Test the metrics whole way through"""
        em = EndModel(
            seed=1,
            batchnorm=False,
            dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
        test_model = copy.deepcopy(em.state_dict())

        # 0 indexed
        new_model = torch.load("checkpoints/model_checkpoint_3.pth")
        self.assertFalse(
            torch.all(
                torch.eq(
                    test_model["network.1.0.weight"],
                    new_model["model"]["network.1.0.weight"],
                )
            )
        )

        new_model = torch.load("checkpoints/model_checkpoint_4.pth")
        self.assertTrue(
            torch.all(
                torch.eq(
                    test_model["network.1.0.weight"],
                    new_model["model"]["network.1.0.weight"],
                )
            )
        )

    def test_resume_training(self):
        em = EndModel(
            seed=1,
            batchnorm=False,
            dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=5)
        em.resume_training(
            (Xs[0], Ys[0]),
            model_path="checkpoints/model_checkpoint_2.pth",
            dev_data=(Xs[1], Ys[1]),
        )

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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=1)
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
        em_2.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=1)
        score_3 = em_2.score((Xs[2], Ys[2]), verbose=False)
        self.assertEqual(score_1, score_3)

    def test_logging(self):
        """Test the basic LogWriter class"""
        log_writer = LogWriter(run_dir="test_dir", run_name="test")
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
            (Xs[0], Ys[0]),
            dev_data=(Xs[1], Ys[1]),
            n_epochs=7,
            log_writer=log_writer,
        )

        # Load the log
        with open(log_writer.log_path, "r") as f:
            run_log = json.load(f)
        self.assertEqual(run_log["config"]["train_config"]["n_epochs"], 7)
        self.assertEqual(len(run_log["run-log"]["train-loss"]), 7)

        # Clean up
        rmtree(log_writer.log_subdir)

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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=3)
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
