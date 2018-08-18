import numpy as np
from torch.utils.data import Dataset


class MultiYDataset(Dataset):
    """A dataset that group each item in X with its labels for t tasks from Y

    Args:
        X: an n-dim iterable of inputs
        Y: a t-length list of n-dim iterables corresponding to labels for t
            tasks
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.t = len(Y)
        n = len(X)
        assert np.all([len(Y_t) == n for Y_t in Y])

    def __getitem__(self, index):
        return tuple([self.X[index], [self.Y[t][index] for t in range(self.t)]])

    def __len__(self):
        return len(self.X)


class MultiXYDataset(Dataset):
    """A dataset that groups each item's t inputs from X and t labels from Y

    Args:
        X: a t-length list of n-dim iterables corresponding to inputs for t
            tasks
        Y: a t-length list of n-dim iterables corresponding to labels for t
            tasks
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.t = len(Y)
        n = len(X[0])
        assert np.all([len(Y_t) == n for Y_t in Y])
        assert np.all([len(X_t) == n for X_t in X])

    def __getitem__(self, index):
        return tuple(
            [
                [self.X[t][index] for t in range(self.t)],
                [self.Y[t][index] for t in range(self.t)],
            ]
        )

    def __len__(self):
        return len(self.X[0])
