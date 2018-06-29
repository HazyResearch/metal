import numpy as np
from scipy.sparse import issparse
import torch

def metric_score(gold, pred, metric, **kwargs):
    if metric == 'accuracy':
        return accuracy_score(gold, pred, **kwargs)
    elif metric == 'coverage':
        return coverage_score(gold, pred, **kwargs)
    elif metric == 'precision':
        return precision_score(gold, pred, **kwargs)
    elif metric == 'recall':
        return recall_score(gold, pred, **kwargs)
    elif metric == 'f1':
        return f1_score(gold, pred, **kwargs)
    elif metric == 'fbeta':
        return fbeta_score(gold, pred, **kwargs)
    else:
        msg = f"The metric you provided ({metric}) is not supported."
        raise ValueError(msg)

def accuracy_score(gold, pred, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (micro) accuracy.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.

    Returns:
        A float, the (micro) accuracy score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    if len(gold) and len(pred):
        acc = np.sum(gold == pred) / len(gold)
    else:
        acc = 0

    return acc

def coverage_score(gold, pred, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (global) coverage.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.

    Returns:
        A float, the (global) coverage score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    return np.sum(pred != 0) / len(pred)

def precision_score(gold, pred, pos_label=1, ignore_in_gold=[], 
    ignore_in_pred=[]):
    """
    Calculate precision for a single class.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.
        pos_label: The class label to treat as positive for precision

    Returns:
        pre: The (float) precision score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    positives = np.where(pred == pos_label, 1, 0).astype(bool)
    trues = np.where(gold == pos_label, 1, 0).astype(bool)
    TP = np.sum(positives * trues)
    FP = np.sum(positives * np.logical_not(trues))

    if TP or FP:
        pre = TP / (TP + FP)
    else:
        pre = 0

    return pre

def recall_score(gold, pred, pos_label=1, ignore_in_gold=[], 
    ignore_in_pred=[]):
    """
    Calculate recall for a single class.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.
        pos_label: The class label to treat as positive for recall

    Returns:
        rec: The (float) recall score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    positives = np.where(pred == pos_label, 1, 0).astype(bool)
    trues = np.where(gold == pos_label, 1, 0).astype(bool)
    TP = np.sum(positives * trues)
    FN = np.sum(np.logical_not(positives) * trues)

    if TP or FN:
        rec = TP / (TP + FN)
    else:
        rec = 0

    return rec

def fbeta_score(gold, pred, pos_label=1, beta=1.0, ignore_in_gold=[], 
    ignore_in_pred=[]):
    """
    Calculate recall for a single class.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.
        pos_label: The class label to treat as positive for f-beta
        beta: The beta to use in the f-beta score calculation

    Returns:
        fbeta: The (float) f-beta score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)
    pre = precision_score(gold, pred, pos_label=pos_label)
    rec = recall_score(gold, pred, pos_label=pos_label)

    if pre or rec:
        fbeta = (1 + beta**2) * (pre * rec)/((beta**2 * pre) + rec)
    else:
        fbeta = 0

    return fbeta

def f1_score(gold, pred, **kwargs):
    return fbeta_score(gold, pred, beta=1.0, **kwargs)

def _to_array(array_like):
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
    
def _preprocess(gold, pred, ignore_in_gold, ignore_in_pred):
    gold = _to_array(gold)
    pred = _to_array(pred)
    if ignore_in_gold or ignore_in_pred:
        gold, pred = _drop_ignored(gold, pred, ignore_in_gold, ignore_in_pred)
    return gold, pred
