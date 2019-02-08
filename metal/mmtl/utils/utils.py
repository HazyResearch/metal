import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, TensorDataset


def to_numpy(Z):
    """Converts a None, list, np.ndarray, or torch.Tensor to np.ndarray;
    also handles converting sparse input to dense."""
    if Z is None:
        return Z
    elif issparse(Z):
        return Z.toarray()
    elif isinstance(Z, np.ndarray):
        return Z
    elif isinstance(Z, list):
        return np.array(Z)
    elif isinstance(Z, torch.Tensor):
        return Z.cpu().numpy()
    else:
        msg = (
            f"Expected None, list, numpy.ndarray or torch.Tensor, "
            f"got {type(Z)} instead."
        )
        raise Exception(msg)


def stack_batches(X):
    """Stack a list of np.ndarrays along the first axis, returning an
    np.ndarray; note this is mainly for smooth hanlding of the multi-task
    setting."""
    X = [to_numpy(Xb) for Xb in X]
    if len(X[0].shape) == 1:
        return np.hstack(X)
    elif len(X[0].shape) == 2:
        return np.vstack(X)
    else:
        raise ValueError(f"Can't stack {len(X[0].shape)}-dim batches.")


def break_ties(Y_s, break_ties="random"):
    """Break ties in each row of a tensor according to the specified policy

    Args:
        Y_s: An [n, k] np.ndarray of probabilities
        break_ties: A tie-breaking policy:
            "abstain": return an abstain vote (0)
            "random": randomly choose among the tied options
                NOTE: if break_ties="random", repeated runs may have
                slightly different results due to difference in broken ties
            [int]: ties will be broken by using this label
    """
    n, k = Y_s.shape
    Y_h = np.zeros(n)
    diffs = np.abs(Y_s - Y_s.max(axis=1).reshape(-1, 1))

    TOL = 1e-5
    for i in range(n):
        max_idxs = np.where(diffs[i, :] < TOL)[0]
        if len(max_idxs) == 1:
            Y_h[i] = max_idxs[0] + 1
        # Deal with "tie votes" according to the specified policy
        elif break_ties == "random":
            Y_h[i] = np.random.choice(max_idxs) + 1
        elif break_ties == "abstain":
            Y_h[i] = 0
        elif isinstance(break_ties, int):
            Y_h[i] = break_ties
        else:
            ValueError(f"break_ties={break_ties} policy not recognized.")
    return Y_h
