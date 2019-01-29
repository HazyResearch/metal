import json
import unittest
from shutil import rmtree

import numpy as np
import torch

from metal.contrib.modules import EmbeddingsEncoder, LSTMModule
from metal.end_model import EndModel
from metal.logging import LogWriter
from metal.tuners.random_tuner import RandomSearchTuner

n = 1000
SEQ_LEN = 5
MAX_INT = 8


class LSTMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set seed
        torch.manual_seed(1)
        np.random.seed(1)

    def _split_dataset(self, X):
        return [X[:800], X[800:900], X[900:]]

    def test_lstm_memorize_first(self):
        """Confirm that lstm can memorize the first token in a long sequence"""
        X = torch.randint(1, MAX_INT + 1, (n, SEQ_LEN)).long()
        Y = X[:, 0]

        Xs = self._split_dataset(X)
        Ys = self._split_dataset(Y)

        embed_size = 4
        hidden_size = 10

        lstm_module = LSTMModule(
            embed_size,
            hidden_size,
            bidirectional=False,
            verbose=False,
            lstm_reduction="attention",
            encoder_class=EmbeddingsEncoder,
            encoder_kwargs={"vocab_size": MAX_INT + 1},
        )
        em = EndModel(
            k=MAX_INT,
            input_module=lstm_module,
            layer_out_dims=[hidden_size, MAX_INT],
            optimizer="adam",
            batchnorm=True,
            seed=1,
            verbose=False,
        )
        em.train_model((Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=10)
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_lstm_memorize_marker(self):
        """Confirm that lstm can return the token that comes after a special marker"""
        X = torch.randint(1, MAX_INT + 1, (n, SEQ_LEN)).long()
        Y = torch.zeros(n).long()
        needles = np.random.randint(1, SEQ_LEN - 1, n)
        for i in range(n):
            X[i, needles[i]] = MAX_INT + 1
            Y[i] = X[i, needles[i] + 1]

        Xs = self._split_dataset(X)
        Ys = self._split_dataset(Y)

        embed_size = 4
        hidden_size = 10

        lstm_module = LSTMModule(
            embed_size,
            hidden_size,
            bidirectional=True,
            verbose=False,
            lstm_reduction="attention",
            encoder_class=EmbeddingsEncoder,
            encoder_kwargs={"vocab_size": MAX_INT + 2},
        )
        em = EndModel(
            k=MAX_INT,
            input_module=lstm_module,
            layer_out_dims=[hidden_size * 2, MAX_INT],
            batchnorm=True,
            seed=1,
            verbose=False,
        )
        em.train_model(
            (Xs[0], Ys[0]),
            valid_data=(Xs[1], Ys[1]),
            n_epochs=15,
            verbose=False,
        )
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_lstm_embeddings_freeze(self):
        """Confirm that if embeddings are frozen, they do not change during training"""
        X = torch.randint(1, MAX_INT + 1, (n, SEQ_LEN)).long()
        Y = torch.zeros(n).long()
        needles = np.random.randint(1, SEQ_LEN - 1, n)
        for i in range(n):
            X[i, needles[i]] = MAX_INT + 1
            Y[i] = X[i, needles[i] + 1]

        Xs = self._split_dataset(X)
        Ys = self._split_dataset(Y)

        embed_size = 4
        hidden_size = 10

        for freeze_embs in [True, False]:
            lstm_module = LSTMModule(
                embed_size,
                hidden_size,
                verbose=False,
                encoder_class=EmbeddingsEncoder,
                encoder_kwargs={
                    "vocab_size": MAX_INT + 2,
                    "freeze": freeze_embs,
                },
            )
            em = EndModel(
                k=MAX_INT,
                input_module=lstm_module,
                layer_out_dims=[hidden_size * 2, MAX_INT],
                verbose=False,
            )

            before = lstm_module.encoder.embeddings.weight.clone()
            em.train_model(
                (Xs[0], Ys[0]),
                valid_data=(Xs[1], Ys[1]),
                n_epochs=15,
                verbose=False,
            )
            after = lstm_module.encoder.embeddings.weight.clone()

            if freeze_embs:
                self.assertEqual(torch.abs(before - after).sum().item(), 0.0)
            else:
                self.assertNotEqual(torch.abs(before - after).sum().item(), 0.0)

    def test_lstm_direct_features(self):
        """Confirm that lstm can work over features passed in directly (rather
        than embedded)."""
        X = torch.randint(1, MAX_INT + 1, (n, SEQ_LEN)).long()
        Y = X[:, 0]

        # Convert X to one-hot features
        Xf = torch.zeros((n, SEQ_LEN, MAX_INT)).long()
        for i in range(n):
            for j in range(SEQ_LEN):
                Xf[i, j, X[i, j] - 1] = 1
        X = Xf

        Xs = self._split_dataset(X)
        Ys = self._split_dataset(Y)

        encoded_size = MAX_INT
        hidden_size = 10

        lstm_module = LSTMModule(
            encoded_size,
            hidden_size,
            bidirectional=False,
            verbose=False,
            lstm_reduction="attention",
        )
        em = EndModel(
            k=MAX_INT,
            input_module=lstm_module,
            layer_out_dims=[hidden_size, MAX_INT],
            optimizer="adam",
            batchnorm=True,
            seed=1,
            verbose=False,
        )
        em.train_model((Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=15)
        score = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertGreater(score, 0.95)

    def test_lstm_determinism(self):
        """Test whether training and scoring is deterministic given seed"""
        X = torch.randint(1, MAX_INT + 1, (n, SEQ_LEN)).long()
        Y = torch.zeros(n).long()
        needles = np.random.randint(1, SEQ_LEN - 1, n)
        for i in range(n):
            X[i, needles[i]] = MAX_INT + 1
            Y[i] = X[i, needles[i] + 1]

        Xs = self._split_dataset(X)
        Ys = self._split_dataset(Y)

        embed_size = 4
        hidden_size = 10

        lstm_module = LSTMModule(
            embed_size,
            hidden_size,
            seed=123,
            bidirectional=True,
            verbose=False,
            lstm_reduction="attention",
            encoder_class=EmbeddingsEncoder,
            encoder_kwargs={"vocab_size": MAX_INT + 2},
        )
        em = EndModel(
            k=MAX_INT,
            input_module=lstm_module,
            layer_out_dims=[hidden_size * 2, MAX_INT],
            batchnorm=True,
            seed=123,
            verbose=False,
        )
        em.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=2, verbose=False
        )
        score_1 = em.score((Xs[2], Ys[2]), verbose=False)

        # Test scoring determinism
        score_2 = em.score((Xs[2], Ys[2]), verbose=False)
        self.assertEqual(score_1, score_2)

        # Test training determinism
        lstm_module_2 = LSTMModule(
            embed_size,
            hidden_size,
            seed=123,
            bidirectional=True,
            verbose=False,
            lstm_reduction="attention",
            encoder_class=EmbeddingsEncoder,
            encoder_kwargs={"vocab_size": MAX_INT + 2},
        )
        em_2 = EndModel(
            k=MAX_INT,
            input_module=lstm_module_2,
            layer_out_dims=[hidden_size * 2, MAX_INT],
            batchnorm=True,
            seed=123,
            verbose=False,
        )
        em_2.train_model(
            (Xs[0], Ys[0]), valid_data=(Xs[1], Ys[1]), n_epochs=2, verbose=False
        )
        score_3 = em_2.score((Xs[2], Ys[2]), verbose=False)
        self.assertEqual(score_1, score_3)

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
