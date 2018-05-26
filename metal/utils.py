import numpy as np
import torch
from torch.utils.data import Dataset


class MultilabelDataset(Dataset):
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

def rargmax(x, eps=1e-8):
    """Argmax with random tie-breaking
    
    Args:
        x: a 1-dim numpy array
    Returns:
        the argmax index
    """
    idxs = np.where(abs(x - np.max(x, axis=0)) < eps)[0]
    return np.random.choice(idxs)

def hard_to_soft(Y_h, k):
    """Converts a 1D tensor of hard labels into a 2D tensor of soft labels

    Args:
        Y_h: an [N], or [N,1] tensor of hard (int) labels >= 1
        k: the target cardinality of the soft label matrix
    """
    Y_h = Y_h.squeeze()
    assert(Y_h.dim() == 1)
    N = Y_h.shape[0]
    Y_s = torch.zeros(N, k, dtype=torch.float)
    for i, j in enumerate(Y_h):
        Y_s[i, j-1] = 1.0
    return Y_s