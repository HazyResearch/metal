import numpy as np
from scipy.sparse import issparse, csr_matrix, hstack
import torch
from torch.utils.data import Dataset


class MetalDataset(Dataset):
    """A dataset that group each item in X with it label from Y
    
    Args:
        X: an n-dim iterable of items
        Y: a torch.Tensor of labels
            This may be hard labels [n] or soft labels [n, k]
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert(len(X) == len(Y))

    def __getitem__(self, index):
        return tuple([self.X[index], self.Y[index]])

    def __len__(self):
        return len(self.X)


class Checkpointer(object):
    def __init__(self, model_class, checkpoint_min=0, checkpoint_runway=0,
        verbose=True):
        """Saves checkpoints as applicable based on a reported metric.

        Args:
            checkpoint_min (float): the initial "best" score to beat
            checkpoint_runway (int): don't save any checkpoints for the first
                this many iterations
        """
        self.model_class = model_class
        self.best_model = None
        self.best_iteration = None
        self.best_score = checkpoint_min
        self.checkpoint_runway = checkpoint_runway
        self.verbose = verbose
        if checkpoint_runway and verbose:
            print(f"No checkpoints will be saved in the first "
                f"checkpoint_runway={checkpoint_runway} iterations.")
  
    def checkpoint(self, model, iteration, score):
        if iteration >= self.checkpoint_runway:
            is_best = score > self.best_score
            if is_best:
                if self.verbose:
                    print(f"Saving model at iteration {iteration} with best "
                        f"score {score}")
                self.best_model = model.state_dict()
                self.best_iteration = iteration
                self.best_score = score

    def restore(self, model):
        if self.best_model is None:
            raise Exception(f"Best model was never found. Best score = "
                f"{self.best_score}")
        if self.verbose:
            print(f"Restoring best model from iteration {self.best_iteration} "
                f"with score {self.best_score}")
            model.load_state_dict(self.best_model)
            return model
        

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
        Y_h: an [n], or [n,1] tensor of hard (int) labels between 0 and k 
            (inclusive), where 0 = abstain.
        k: the largest possible label in Y_h
    Returns:
        Y_s: a torch.FloatTensor of shape [n, k + 1] where Y_s[i,j] is the soft
            label for item i and class j.
    """
    Y_h = Y_h.clone()
    Y_h = Y_h.squeeze()
    assert(Y_h.dim() == 1)
    assert((Y_h >= 0).all())
    assert((Y_h <= k).all())
    n = Y_h.shape[0]
    Y_s = torch.zeros((n, k+1))
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
        msg = (f"Input of type {orig_type} could not be converted to 1d "
            "np.ndarray")
        raise ValueError(msg)
        
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

def convert_labels(Y, source, dest):
    """Convert a matrix from one label type to another
    
    Args:
        X: A np.ndarray or torch.Tensor of labels (ints)
        source: The convention the labels are currently expressed in
        dest: The convention to convert the labels to

    Conventions:
        'categorical': [0: abstain, 1: positive, 2: negative]
        'plusminus': [0: abstain, 1: positive, -1: negative]
        'onezero': [0: negative, 1: positive]

    Note that converting to 'onezero' will combine abstain and negative labels.
    """
    if Y is None: return Y
    Y = Y.copy()
    negative_map = {'categorical': 2, 'plusminus': -1, 'onezero': 0}
    Y[Y == negative_map[source]] = negative_map[dest]
    return Y

def plusminus_to_categorical(Y):
    return convert_labels(Y, 'plusminus', 'categorical')

def categorical_to_plusminus(Y):
    return convert_labels(Y, 'categorical', 'plusminus')

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
