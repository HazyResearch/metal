import json
import unittest
from shutil import rmtree

import numpy as np
import torch

from metal.end_model import EndModel
from metal.modules import EmbeddingsEncoder, LSTMModule
from metal.tuners.random_tuner import RandomSearchTuner
from metal.utils import LogWriter


class RandomSearchModelTunerTest(unittest.TestCase):
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

    def test_config_constant(self):
        search_space = {"a": 1}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 1)

    def test_config_list(self):
        search_space = {"a": [1, 2]}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 2)

    def test_config_two_values(self):
        search_space = {"a": [1], "b": [1, 2, 3]}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 3)

    def test_config_range(self):
        search_space = {"a": [1], "b": [1, 2, 3], "c": {"range": [1, 10]}}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 10)

    def test_config_unbounded_max_search(self):
        search_space = {"a": [1], "b": [1, 2, 3], "c": {"range": [1, 10]}}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=0))
        self.assertEqual(len(configs), 3)

    def test_config_log_range(self):
        search_space = {
            "a": [1],
            "b": [1, 2, 3],
            "c": {"range": [1, 10]},
            "d": {"range": [1, 10], "scale": "log"},
        }
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=20))
        self.assertEqual(len(configs), 20)
        self.assertGreater(
            np.mean([c["c"] for c in configs]),
            np.mean([c["d"] for c in configs]),
        )

    def test_tuner_and_logging(self):
        Xs, Ys = self.single_problem

        # Set up RandomSearchTuner
        tuner = RandomSearchTuner(EndModel, log_writer_class=LogWriter)

        # Run the search
        init_kwargs = {
            "seed": 1,
            "input_batchnorm": False,
            "middle_batchnorm": False,
            "layer_out_dims": [2, 10, 2],
            "verbose": False,
        }
        search_space = {"middle_dropout": [0.0, 1.0]}
        tuner.search(
            search_space,
            (Xs[1], Ys[1]),
            init_kwargs=init_kwargs,
            train_args=[(Xs[0], Ys[0])],
            train_kwargs={"n_epochs": 10},
            verbose=False,
        )

        # Load the log
        with open(tuner.report_path, "r") as f:
            tuner_report = json.load(f)

        # Confirm that when input dropout = 1.0, score tanks, o/w does well
        self.assertLess(tuner_report[0]["score"], 0.65)
        self.assertGreater(tuner_report[1]["score"], 0.95)

        # Clean up
        rmtree(tuner.log_rootdir)

    def test_tuner_with_lstm(self):
        """Test basic functionality *and* determinism/seeding of the tuner
        with a more complex EndModel having an input module"""
        # From tests/metal/modules/test_lstm.py; TODO: Refactor this
        n = 1000
        SEQ_LEN = 5
        MAX_INT = 8
        X = torch.randint(1, MAX_INT + 1, (n, SEQ_LEN)).long()
        Y = torch.zeros(n).long()
        needles = np.random.randint(1, SEQ_LEN - 1, n)
        for i in range(n):
            X[i, needles[i]] = MAX_INT + 1
            Y[i] = X[i, needles[i] + 1]
        Xs = [X[:800], X[800:900], X[900:]]
        Ys = [Y[:800], Y[800:900], Y[900:]]

        embed_size = 4
        hidden_size = 10

        # Set up RandomSearchTuner
        tuner = RandomSearchTuner(
            EndModel,
            module_classes={"input_module": LSTMModule},
            log_writer_class=LogWriter,
            seed=123,
        )

        # EndModel init kwargs
        init_kwargs = {
            "seed": 123,
            "batchnorm": True,
            "k": MAX_INT,
            "layer_out_dims": [hidden_size * 2, MAX_INT],
            "input_batchnorm": True,
            "verbose": False,
        }

        # LSTMModule args & kwargs
        module_args = {}
        module_args["input_module"] = (embed_size, hidden_size)
        module_kwargs = {}
        module_kwargs["input_module"] = {
            "seed": 123,
            "bidirectional": True,
            "verbose": False,
            "lstm_reduction": "attention",
            "encoder_class": EmbeddingsEncoder,
            "encoder_kwargs": {"vocab_size": MAX_INT + 2},
        }

        # Set up search space
        # NOTE: No middle layers here, so these should return the same scores!
        search_space = {"middle_dropout": [0.0, 1.0]}

        # Run random grid search
        tuner.search(
            search_space,
            (Xs[1], Ys[1]),
            init_kwargs=init_kwargs,
            train_args=[(Xs[0], Ys[0])],
            train_kwargs={"n_epochs": 2},
            module_args=module_args,
            module_kwargs=module_kwargs,
            verbose=False,
        )

        # Load the log
        with open(tuner.report_path, "r") as f:
            tuner_report = json.load(f)

        # Confirm determinism
        self.assertEqual(tuner_report[0]["score"], tuner_report[1]["score"])

        # Clean up
        rmtree(tuner.log_rootdir)


if __name__ == "__main__":
    unittest.main()
