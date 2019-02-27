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


def plot_probabilities_histogram(Y_probs, title=None):
    """Plot a histogram from a numpy array of probabilities

    Args:
        Y_probs: An [n] or [n, 1] np.ndarray of probabilities (floats in [0,1])
    """
    if Y_probs.ndim > 1:
        msg = (
            f"Arg Y_probs should be a 1-dimensional np.ndarray, not of shape "
            f"{Y_probs.shape}."
        )
        raise ValueError(msg)
    plt.hist(Y_probs, bins=20)
    plt.xlim((0, 1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    if isinstance(title, str):
        plt.title(title)
    plt.show()


def plot_predictions_histogram(Y_preds, Y_gold, title=None):
    """Plot a histogram comparing int predictions vs true labels by class

    Args:
        Y_gold: An [n] or [n, 1] np.ndarray of gold labels
        Y_preds: An [n] or [n, 1] np.ndarray of predicted int labels
    """
    labels = list(set(Y_gold).union(set(Y_preds)))
    edges = [x - 0.5 for x in range(min(labels), max(labels) + 2)]

    plt.hist([Y_preds, Y_gold], bins=edges, label=["Predicted", "Gold"])
    ax = plt.gca()
    ax.set_xticks(labels)
    plt.xlabel("Label")
    plt.ylabel("# Predictions")
    plt.legend(loc="upper right")
    if isinstance(title, str):
        plt.title(title)
    plt.show()


def plot_calibration_histogram(Y_probs, Y_gold, title=None, legend=True):
    """Plot a histogram of correct predictions from an array of probabilities

    Args:
        Y_probs: An [n] or [n, 1] np.ndarray of probabilities (floats in [0,1])
        Y_gold: An [n] or [n, 1] np.ndarray of gold labels
    """
    if Y_probs.ndim > 1:
        Y_preds = np.argmax(np.array(Y_probs), axis=1).astype(float) + 1.0
        Y_probs = Y_probs[:, 0]
    else:
        Y_preds = (np.sign(Y_probs - 0.5) + 1.0) / 2.0

    correct_idx = list(np.where(Y_preds == Y_gold)[0])
    print("Accuracy: ", len(correct_idx) / (1.0 * len(Y_probs)))

    Y_p_correct = Y_probs[correct_idx]
    plt.hist(Y_probs, bins=20, alpha=0.75)
    plt.hist(Y_p_correct, bins=20, alpha=0.75)
    plt.xlim((0, 1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    if isinstance(title, str):
        plt.title(title)
    if legend:
        plt.legend(["All", "Correct"])
