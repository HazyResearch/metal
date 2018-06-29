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
    assert((Y_h >= 1).all())
    N = Y_h.shape[0]
    Y_s = torch.zeros(N, k, dtype=torch.float)
    for i, j in enumerate(Y_h):
        Y_s[i, j-1] = 1.0
    return Y_s

def recursive_merge_dicts(x, y, misses='report', verbose=None):
    """
    Merge dictionary y into x, overwriting elements of x when there is a
    conflict, except if the element is a dictionary, in which case recurse.

    misses: what to do if a key in y is not in x
        'insert'    -> set x[key] = value
        'exception' -> raise an exception
        'report'    -> report the name of the missing key
        'ignore'    -> do nothing

    TODO: give example here (pull from tests)
    """
    def recurse(x, y, misses='report', verbose=True):
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
                    recursive_merge_dicts(x[k], v, misses, verbose)
                else:
                    if x[k] == v:
                        msg = f"Reaffirming {k}={x[k]}"
                    else:
                        msg = f"Overwriting {k}={x[k]} to {k}={v}"
                        x[k] = v
                    if verbose:
                        print(msg)
            else:
                for kx, vx in x.items():
                    if isinstance(vx, dict):
                        found = recursive_merge_dicts(vx, {k: v}, 
                            misses='ignore', verbose=verbose)
                    if found:
                        break
            if not found:
                msg = f'Could not find kwarg "{k}" in default config.'
                if misses == 'insert':
                    x[k] = v
                    if verbose: 
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
        verbose = y.get('verbose', x.get('verbose', True))

    recurse(x, y, misses, verbose)
    return x