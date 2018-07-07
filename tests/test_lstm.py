import sys
import unittest

import numpy as np
import torch

sys.path.append("../metal")
from metal.input_modules import LSTMModule
from metal.end_model import EndModel


N = 1200
SEQ_LEN = 5
MAX_INT = 8


class LSTMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set seed
        torch.manual_seed(1)
        np.random.seed(1)

        X = torch.randint(1, MAX_INT + 1, (N,SEQ_LEN)).long()
        Y = X[:,0]

        Xs = [X[:1000], X[1000:1100], X[1100:]]
        Ys = [Y[:1000], Y[1000:1100], Y[1100:]]

        cls.data = (Xs, Ys)

    def test_lstm_memorize_first(self):
        input_size = 1
        hidden_size = 10
        vocab_size = MAX_INT + 1

        Xs, Ys = self.data
        lstm_module = LSTMModule(input_size, hidden_size, vocab_size, 
            verbose=False)
        em = EndModel(
            cardinality=MAX_INT, 
            input_module=lstm_module, 
            layer_output_dims=[hidden_size, MAX_INT],
            seed=1,
            verbose=False)
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], n_epochs=20, verbose=False)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()        