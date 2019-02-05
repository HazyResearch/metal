import unittest

import numpy as np
import torch


class LoggerTest(unittest.TestCase):
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

    def test_seconds(self):
        pass

    def test_examples(self):
        pass

    def test_batches(self):
        pass

    def test_epochs(self):
        pass

    def test_tqdm(self):
        """Confirm that nothing breaks with tqdm on or off"""
        pass

    def test_train_metrics(self):
        """Confirm non-default train metrics can be passed"""
        pass

    def test_valid_metrics(self):
        """Confirm non-default valid metrics can be passed"""
        pass

    def test_custom_metrics(self):
        """Confirm custom metrics can be passed"""
        pass

    def test_label_model_logger(self):
        """Confirm that logger works at a basic level with LabelModel"""
