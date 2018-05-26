import numpy as np
import torch

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