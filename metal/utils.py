import numpy as np
from scipy.sparse import issparse
import torch
from torch.utils.data import Dataset


class MetalDataset(Dataset):
    """A dataset that group each item in X with it label from Y
    
    Args:
        X: an N-dim iterable of items
        Y: a torch.Tensor of labels
            This may be hard labels [N] or soft labels [N, k]
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert(len(X) == len(Y))

    def __getitem__(self, index):
        return tuple([self.X[index], self.Y[index]])

    def __len__(self):
        return len(self.X)

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
        Y_h: an [N], or [N,1] tensor of hard (int) labels between 0 and k 
            (inclusive), where 0 = abstain.
        k: the largest possible label in Y_h
    Returns:
        Y_s: a torch.FloatTensor of shape [N, k + 1] where Y_s[i,j] is the soft
            label for item i and class j.
    """
    Y_h = Y_h.clone()
    Y_h = Y_h.squeeze()
    assert(Y_h.dim() == 1)
    assert((Y_h >= 0).all())
    assert((Y_h <= k).all())
    N = Y_h.shape[0]
    Y_s = torch.zeros((N, k+1))
    for i, j in enumerate(Y_h):
        Y_s[i, j] = 1.0
    return Y_s

def arraylike_to_numpy(array_like):
    """Convert a 1d array-like (e.g,. list, tensor, etc.) to an np.ndarray"""

    orig_type = type(array_like)
    
    # Convert to np.ndarray
    if isinstance(array_like, np.ndarray):
        pass
    elif isinstance(array_like, list):
        array_like = np.array(array_like)
    elif issparse(array_like):
        array_like = array_like.toarray()
    elif isinstance(array_like, torch.Tensor):
        array_like = array_like.numpy()
    elif not isinstance(array_like, np.ndarray):
        array_like = np.array(array_like)
    else:
        raise ValueError(f"Input of type {orig_type} could not be converted "
            "to 1d np.ndarray")
        
    # Correct shape
    if (array_like.ndim > 1) and (1 in array_like.shape):
        array_like = array_like.flatten()
    if array_like.ndim != 1:
        raise ValueError("Input could not be converted to 1d np.array")

    # Convert to ints
    if any(array_like % 1):
        raise ValueError("Input contains at least one non-integer value.")
    array_like = array_like.astype(np.dtype(int))

    return array_like

def binary_to_categorical(X):
    """Convert a matrix from [-1,0,1] labels to [0,1,2] labels

    Args:
        X: A np.ndarray or torch.Tensor with the following label interpretations:
            -1: negative
            0: abstain
            1: positive
        After the conversion, the labels will be:
            0: abstain
            1: positive
            2: negative
    """
    if X is None: return X
    X = X.copy()
    X[X == -1] = 2
    return X

def categorical_to_binary(X):
    """Convert a matrix from [0,1,2] labels to [-1,0,1] labels

    Args:
        X: A np.ndarray or torch.Tensor with the following label interpretations:
            0: abstain
            1: positive
            2: negative
        After the conversion, the labels will be:
            -1: negative
            0: abstain
            1: positive
    """
    if X is None: return X
    X = X.copy()
    X[X == 2] = -1
    return X


def recursive_merge_dicts(x, y, misses='report', verbose=None):
    """
    Merge dictionary y into a copy of x, overwriting elements of x when there 
    is a conflict, except if the element is a dictionary, in which case recurse.

    misses: what to do if a key in y is not in x
        'insert'    -> set x[key] = value
        'exception' -> raise an exception
        'report'    -> report the name of the missing key
        'ignore'    -> do nothing

    TODO: give example here (pull from tests)
    """
    def recurse(x, y, misses='report', verbose=1):
        found = True
        for k, v in y.items():
            found = False
            if k in x:
                found = True
                if isinstance(x[k], dict):
                    if not isinstance(v, dict):
                        msg = (f"Attempted to overwrite dict {k} with "
                            f"non-dict: {v}")
                        raise ValueError(msg)
                    recurse(x[k], v, misses, verbose)
                else:
                    if x[k] == v:
                        msg = f"Reaffirming {k}={x[k]}"
                    else:
                        msg = f"Overwriting {k}={x[k]} to {k}={v}"
                        x[k] = v
                    if verbose > 1 and k != 'verbose':
                        print(msg)
            else:
                for kx, vx in x.items():
                    if isinstance(vx, dict):
                        found = recurse(vx, {k: v}, 
                            misses='ignore', verbose=verbose)
                    if found:
                        break
            if not found:
                msg = f'Could not find kwarg "{k}" in default config.'
                if misses == 'insert':
                    x[k] = v
                    if verbose > 1: 
                        print(f"Added {k}={v} from second dict to first")
                elif misses == 'exception':
                    raise ValueError(msg)
                elif misses == 'report':
                    print(msg)
                else:
                    pass
        return found
    
    # If verbose is not provided, look for an value in y first, then x
    # (Do this because 'verbose' kwarg is often inside one or both of x and y)
    if verbose is None:
        verbose = y.get('verbose', x.get('verbose', 1))

    z = x.copy()
    recurse(z, y, misses, verbose)
    return z