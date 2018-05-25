from collections import Counter

import numpy as np
from scipy.sparse import issparse
import torch

def _to_array(array_like):
    """Convert a 1d array-like (e.g,. list, tensor, etc.) to an np.ndarray"""
    
    try:
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
            raise ValueError
        
        # Convert to ints
        array_like = array_like.astype(np.dtype(int))

        # Correct shape
        if (array_like.ndim > 1) and (1 in array_like.shape):
            array_like = array_like.flatten()
        
        # If unsuccessful, report it
        if not isinstance(array_like, np.ndarray) or array_like.ndim != 1:
            raise ValueError # To be caught by the next line
    except ValueError:
        msg = (f"Input of type {orig_type} could not be converted to 1d "
            "np.ndarray")
        raise ValueError(msg)

    return array_like


def _drop_ignored(gold, pred, ignore_in_gold, ignore_in_pred):
    """Remove from gold and pred all items with labels designated to ignore."""
    keepers = np.ones_like(gold).astype(bool)
    for x in ignore_in_gold:
        keepers *= np.where(gold != x, 1, 0).astype(bool)
    for x in ignore_in_pred:
        keepers *= np.where(pred != x, 1, 0).astype(bool)

    gold = gold[keepers]
    pred = pred[keepers]
    return gold, pred


def accuracy_score(gold, pred, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate accuracy.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.

    Returns:
        A float, the accuracy score
    """
    gold = _to_array(gold)
    pred = _to_array(pred)
    if ignore_in_gold or ignore_in_pred:
        gold, pred = _drop_ignored(gold, pred, ignore_in_gold, ignore_in_pred)

    if len(gold) and len(pred):
        acc = np.sum(gold == pred) / len(gold)
    else:
        acc = 0

    return acc


def metric_score(gold, pred, metric, **kwargs):
    if metric == 'accuracy':
        return accuracy_score(gold, pred, **kwargs)
    else:
        msg = f"The metric you provided ({metric}) is not supported."
        raise ValueError(msg)


def confusion_matrix(gold, pred, null_pred=False, null_gold=False):
    """A shortcut class for building a confusion matrix all at once.
    
    Args:
        gold: a torch.Tensor of gold labels (ints)
        pred: a torch.Tensor of predictions (ints)
    """    
    conf = ConfusionMatrix(null_pred=null_pred, null_gold=null_gold)
    conf.add(gold, pred)
    mat = conf.compile()
    return mat


class ConfusionMatrix(object):
    """
    An iteratively built confusion matrix
    Assumed axes are true label on top, predictions on the side
    """
    def __init__(self, null_pred=False, null_gold=False):
        """
        Args:
            null_pred: If True, show the row corresponding to null predictions
            null_gold: If True, show the col corresponding to null gold labels

        """
        self.counter = Counter()
        self.mat = None
        self.null_pred = null_pred
        self.null_gold = null_gold

    def __repr__(self):
        if self.mat is None:
            self.compile()
        return str(self.mat)

    def add(self, gold, pred):
        """
        Args:
            gold: a torch.Tensor of gold labels (ints)
            pred: a torch.Tensor of predictions (ints)
        """
        self.counter.update(zip(pred, gold))
    
    def compile(self, trim=True):
        k = max([max(tup) for tup in self.counter.keys()]) + 1  # include 0

        mat = np.zeros((k, k), dtype=int)
        for (p, y), v in self.counter.items():
            mat[p, y] = v
        
        if trim and not self.null_pred:
            mat = mat[1:, :]
        if trim and not self.null_gold:
            mat = mat[:, 1:]

        self.mat = mat
        return mat

    def display(self, counts=True, indent=0, spacing=2, decimals=3, 
        mark_diag=True):
        mat = self.compile(trim=False)
        m, n = mat.shape
        tab = ' ' * spacing
        margin = ' ' * indent

        # Print headers
        s = margin + ' ' * (5 + spacing)
        for j in range(n):
            if j == 0 and not self.null_gold:
                continue
            s += f" y={j} " + tab
        print(s)

        # Print data
        for i in range(m):
            # Skip null predictions row if necessary
            if i == 0 and not self.null_pred:
                continue
            s = margin + f" l={i} " + tab
            for j in range(n):
                # Skip null gold if necessary
                if j == 0 and not self.null_gold:
                    continue
                else:
                    if i == j and mark_diag and not counts:
                        s = s[:-1] + '*'
                    if counts:
                        s += f"{mat[i,j]:^5d}" + tab
                    else:
                        s += f"{mat[i,j]/sum(mat[i,1:]):>5.3f}" + tab
            print(s)        
