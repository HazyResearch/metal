import numpy as np
import torch
from torch.utils.data import Dataset

class MTMetalDataset(Dataset):
    """A dataset that group each item in X with its labels for T tasks from Y
    
    Args:
        X: an N-dim iterable of items
        Y: a T-dim list of N-dim iterables corresponding to labels for T tasks
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.T = len(Y)
        assert np.all([len(Y_t) == len(X) for Y_t in Y])

    def __getitem__(self, index):
        return tuple([self.X[index], [self.Y[t][index] for t in range(self.T)]])

    def __len__(self):
        return len(self.X)