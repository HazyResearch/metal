import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

############################################################
# Label Matrix Plotting
############################################################


def view_label_matrix(L, colorbar=True):
    """Display an [n, m] matrix of labels"""
    L = L.todense() if sparse.issparse(L) else L
    plt.imshow(L, aspect="auto")
    plt.title("Label Matrix")
    if colorbar:
        labels = sorted(np.unique(np.asarray(L).reshape(-1, 1).squeeze()))
        boundaries = np.array(labels + [max(labels) + 1]) - 0.5
        plt.colorbar(boundaries=boundaries, ticks=labels)
    plt.show()


def view_overlaps(L, self_overlaps=False, normalize=True, colorbar=True):
    """Display an [m, m] matrix of overlaps"""
    L = L.todense() if sparse.issparse(L) else L
    G = _get_overlaps_matrix(L, normalize=normalize)
    if not self_overlaps:
        np.fill_diagonal(G, 0)  # Zero out self-overlaps
    plt.imshow(G, aspect="auto")
    plt.title("Overlaps")
    if colorbar:
        plt.colorbar()
    plt.show()


def view_conflicts(L, normalize=True, colorbar=True):
    """Display an [m, m] matrix of conflicts"""
    L = L.todense() if sparse.issparse(L) else L
    C = _get_conflicts_matrix(L, normalize=normalize)
    plt.imshow(C, aspect="auto")
    plt.title("Conflicts")
    if colorbar:
        plt.colorbar()
    plt.show()


def _get_overlaps_matrix(L, normalize=True):
    n, m = L.shape
    X = np.where(L != 0, 1, 0).T
    G = X @ X.T

    if normalize:
        G = G / n
    return G


def _get_conflicts_matrix(L, normalize=True):
    n, m = L.shape
    C = np.zeros((m, m))

    # Iterate over the pairs of LFs
    for i in range(m):
        for j in range(m):
            # Get the overlapping non-zero indices
            overlaps = list(
                set(np.where(L[:, i] != 0)[0]).intersection(np.where(L[:, j] != 0)[0])
            )
            C[i, j] = np.where(L[overlaps, i] != L[overlaps, j], 1, 0).sum()

    if normalize:
        C = C / n
    return C


############################################################
# Classifier Diagnostics
############################################################


def plot_probabilities_histogram(Y_p, title=None):
    """Plot a histogram from a numpy array of probabilities

    Args:
        Y_p: An [n] or [n, 1] np.ndarray of probabilities (floats in [0,1])
    """
    if Y_p.ndim > 1:
        msg = (
            f"Arg Y_p should be a 1-dimensional np.ndarray, not of shape "
            f"{Y_p.shape}."
        )
        raise ValueError(msg)
    plt.hist(Y_p, bins=20)
    plt.xlim((0, 1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    if isinstance(title, str):
        plt.title(title)
    plt.show()


def plot_predictions_histogram(Y_ph, Y, title=None):
    """Plot a histogram comparing int predictions vs true labels by class

    Args:
        Y_ph: An [n] or [n, 1] np.ndarray of predicted int labels
        Y: An [n] or [n, 1] np.ndarray of gold labels
    """
    labels = list(set(Y).union(set(Y_ph)))
    edges = [x - 0.5 for x in range(min(labels), max(labels) + 2)]

    plt.hist([Y_ph, Y], bins=edges, label=["Predicted", "Gold"])
    ax = plt.gca()
    ax.set_xticks(labels)
    plt.xlabel("Label")
    plt.ylabel("# Predictions")
    plt.legend(loc="upper right")
    if isinstance(title, str):
        plt.title(title)
    plt.show()
