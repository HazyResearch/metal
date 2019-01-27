import numpy as np
import torch

from metal.end_model import SparseLogisticRegression


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
