import os
import random

import numpy as np
import torch
from scipy.sparse import issparse


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
    np.ndarray; note this is mainly for smooth handling of the multi-task
    setting."""
    X = [to_numpy(Xb) for Xb in X]
    if X[0].ndim == 1:
        return np.hstack(X)
    elif X[0].ndim == 2:
        return np.vstack(X)
    else:
        raise ValueError(f"Can't stack {X[0].ndim}-dim batches.")
