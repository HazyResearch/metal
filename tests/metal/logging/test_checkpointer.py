import copy
import os
import unittest
from shutil import rmtree

import numpy as np
import torch

from metal.end_model import EndModel


class CheckpointerTest(unittest.TestCase):
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
        cls.checkpoint_dir = "tests/checkpoints/"

    @classmethod
    def tearDownClass(cls):
        print("TODO: Confirm this is deleting checkpoint directory")
        rmtree(cls.checkpoint_dir)

    def test_checkpointing(self):
        """Confirm that different checkpoints are being saved with checkpoint_every on"""
        em = EndModel(
            seed=1,
            batchnorm=False,
            dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]),
            valid_data=(Xs[1], Ys[1]),
            n_epochs=5,
            checkpoint=True,
            checkpoint_every=1,
        )
        test_model = copy.deepcopy(em.state_dict())

        new_model = torch.load("checkpoints/model_checkpoint_4.pth")
        self.assertFalse(
            torch.all(
                torch.eq(
                    test_model["network.1.0.weight"],
                    new_model["model"]["network.1.0.weight"],
                )
            )
        )
        new_model = torch.load("checkpoints/model_checkpoint_5.pth")
        self.assertTrue(
            torch.all(
                torch.eq(
                    test_model["network.1.0.weight"],
                    new_model["model"]["network.1.0.weight"],
                )
            )
        )

    def test_resume_training(self):
        """Confirm that a checkpoint can be saved and reloaded without throwing error"""
        em = EndModel(
            seed=1,
            batchnorm=False,
            dropout=0.0,
            layer_out_dims=[2, 10, 2],
            verbose=False,
        )
        Xs, Ys = self.single_problem
        em.train_model(
            (Xs[0], Ys[0]),
            valid_data=(Xs[1], Ys[1]),
            n_epochs=5,
            checkpoint=True,
            checkpoint_every=1,
        )
        em.resume_training(
            (Xs[0], Ys[0]),
            valid_data=(Xs[1], Ys[1]),
            model_path="checkpoints/model_checkpoint_2.pth",
        )

    def test_checkpoint_metric(self):
        """Confirm that a non-standard checkpoint_metric can be used"""
        pass

    def test_checkpoint_metric_mode(self):
        """Confirm that metric_mode is used properly"""
        pass

    def test_checkpoint_runway(self):
        """Confirm that no checkpoints are saved the first checkpoint_runway iters"""
        pass
