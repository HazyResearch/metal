import unittest

import numpy as np
import torch

from metal.end_model import EndModel
from metal.modules import LSTMModule

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
        vocab_size = MAX_INT + 1

        lstm_module = LSTMModule(
            embed_size,
            hidden_size,
            vocab_size,
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
        em.train_model((Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=10)
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
        vocab_size = MAX_INT + 2

        lstm_module = LSTMModule(
            embed_size,
            hidden_size,
            vocab_size,
            bidirectional=True,
            verbose=False,
            lstm_reduction="attention",
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
            (Xs[0], Ys[0]), dev_data=(Xs[1], Ys[1]), n_epochs=15, verbose=False
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
        vocab_size = MAX_INT + 2

        for freeze_embs in [True, False]:
            lstm_module = LSTMModule(
                embed_size,
                hidden_size,
                vocab_size,
                freeze=freeze_embs,
                verbose=False,
            )
            em = EndModel(
                k=MAX_INT,
                input_module=lstm_module,
                layer_out_dims=[hidden_size * 2, MAX_INT],
                verbose=False,
            )

            print(lstm_module.embeddings.weight.requires_grad)
            before = lstm_module.embeddings.weight.clone()
            print(before)
            em.train_model(
                (Xs[0], Ys[0]),
                dev_data=(Xs[1], Ys[1]),
                n_epochs=15,
                verbose=False,
            )
            print(lstm_module.embeddings.weight.requires_grad)
            after = lstm_module.embeddings.weight.clone()
            print(after)

            if freeze_embs:
                self.assertEqual(torch.abs(before - after).sum().item(), 0.0)
            else:
                self.assertNotEqual(torch.abs(before - after).sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
