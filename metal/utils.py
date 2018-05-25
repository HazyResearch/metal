import numpy as np

def rargmax(x, axis=0, eps=1e-8):
    """Argmax with random tie-breaking
    
    Args:
        x: a 1-dim numpy array
        axis: the axis on which to find averages
    Returns:
        the argmax index
    """
    idxs = np.where(abs(x - np.max(x, axis=axis)) < eps)[0]
    return np.random.choice(idxs)