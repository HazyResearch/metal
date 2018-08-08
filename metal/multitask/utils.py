import numpy as np
import torch
from torch.utils.data import Dataset

class MultiYDataset(Dataset):
    """A dataset that group each item in X with its labels for T tasks from Y
    
    Args:
        X: an N-dim iterable of inputs
        Y: a T-dim list of N-dim iterables corresponding to labels for T tasks
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.T = len(Y)
        N = len(X)
        assert np.all([len(Y_t) == N for Y_t in Y])

    def __getitem__(self, index):
        return tuple([self.X[index], [self.Y[t][index] for t in range(self.T)]])

    def __len__(self):
        return len(self.X)


class MultiXYDataset(Dataset):
    """A dataset that groups each item's T inputs from X and T labels from Y
    
    Args:
        X: a T-dim list of N-dim iterables corresponding to inputs for T tasks
        Y: a T-dim list of N-dim iterables corresponding to labels for T tasks
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.T = len(Y)
        N = len(X[0])
        assert np.all([len(Y_t) == N for Y_t in Y])
        assert np.all([len(X_t) == N for X_t in X])

    def __getitem__(self, index):
        return tuple([[self.X[t][index] for t in range(self.T)], 
                      [self.Y[t][index] for t in range(self.T)]])

    def __len__(self):
        return len(self.X[0])