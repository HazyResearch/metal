import json
import os
import unittest
from shutil import rmtree

import numpy as np
import torch

from metal.end_model import EndModel


class LogWriterTest(unittest.TestCase):
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
        cls.log_dir = "tests/logs/"

    @classmethod
    def tearDownClass(cls):
        print("TODO: Confirm that this is deleting logs directory")
        # Clean up
        rmtree(cls.log_dir)

    def test_logwriter(self):
        """Test the basic LogWriter class"""
        writer_kwargs = {
            "log_dir": self.log_dir,
            "run_dir": "test_dir",
            "run_name": "test",
        }

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
            valid_data=(Xs[1], Ys[1]),
            n_epochs=7,
            checkpoint=False,
            writer="json",
            **writer_kwargs,
        )
        # Load the log
        with open(em.writer.log_path, "r") as f:
            log_dict = json.load(f)

        self.assertEqual(log_dict["config"]["train_config"]["n_epochs"], 7)
        self.assertEqual(len(log_dict["run_log"]["train/loss"]), 7)

    def test_tensorboard(self):
        """Test the TensorBoardWriter class"""
        pass
        # log_dir = os.path.join(self.log_dir, "tensorboard")
        # writer_kwargs = {"log_dir": log_dir, "run_dir": "test_dir", "run_name": "test"}

        # em = EndModel(
        #     seed=1,
        #     input_batchnorm=False,
        #     middle_batchnorm=False,
        #     input_dropout=0.0,
        #     middle_dropout=0.0,
        #     layer_out_dims=[2, 10, 2],
        #     verbose=False,
        # )
        # Xs, Ys = self.single_problem
        # em.train_model(
        #     (Xs[0], Ys[0]),
        #     valid_data=(Xs[1], Ys[1]),
        #     n_epochs=2,
        #     checkpoint=False,
        #     writer="tensorboard",
        #     **writer_kwargs,
        # )

        # # Load the log
        # with open(em.writer.log_path, "r") as f:
        #     pass

        # # Confirm that the event file was written
        # self.assertTrue(False)
