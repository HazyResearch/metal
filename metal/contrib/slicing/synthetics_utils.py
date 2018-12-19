from random import shuffle

import matplotlib.pyplot as plt
import numpy as np


def shuffle_matrices(matrices):
    """Shuffle each member of a list of matrices having the same first dimension
    (along first dimension) according o the same shuffling order."""

    N = matrices[0].shape[0]
    idxs = list(range(N))
    shuffle(idxs)
    out = []
    for M in matrices:
        if M.shape[0] != N:
            raise ValueError("All matrices must have same first dimension.")
        out.append(M[idxs])
    return out


def generate_multi_mode_data(n, mus, props, labels, plot=False):
    """Generate multi-mode data

    Args:
        - n: [int] Number of data points to generate
        - mus: [list of d-dim np.arrays] centers of the modes
        - props: [list of floats] proportion of data in each mode
        - labels: [list of ints] class label of each mode
        - plot: [bool] Whether to plot the data

    Returns:
        - X: [n x d-dim array] Data points
        - Y: [n-dim array] Data labels
        - C: [n-dim array] Index of the mode each data point belongs to
    """

    assert sum(props) == 1.0
    ns = [int(n * prop) for prop in props]
    d = mus[0].shape[0]
    I_d = np.diag(np.ones(d))

    # Generate data
    Xu = [
        np.random.multivariate_normal(mu, I_d, size=ni)
        for mu, ni in zip(mus, ns)
    ]
    Yu = [l * np.ones(ni) for ni, l in zip(ns, labels)]
    Cu = [i * np.ones(ni) for i, ni in enumerate(ns)]

    # Plot the data
    if plot:
        for Xi in Xu:
            plt.scatter(Xi[:, 0], Xi[:, 1])
        plt.show()

    # Generate labels and shuffle
    return shuffle_matrices([np.vstack(Xu), np.hstack(Yu), np.hstack(Cu)])


def generate_label_matrix(
    n, accs, covs, Y, C, overlap_portion=0.3, overlap_acc=1.0
):
    """Generate label matrix. We assume that the last LF is the head LF and the
    one before it is the torso LF it will interact with.

    Args:
        - n: [int] Number of data points
        - accs: [list of floats] accuracies of LFs
        #TODO: covs isn't the overall coverage, but coverage on the associated mode
        - covs: [list of floats] coverage for each LF for its mode
        - Y: [n-dim array] Data labels
        - C: [n-dim array] Index of the mode each data point belongs to
        - overlap_portion: [float] % of "head" LF that overlaps with "torso" LF
        TODO: Not using overlap_acc yet!
        - overlap_acc: [float] Accuracy of torso LF | head LF on overlap_portion

    Returns:
        - L: [n x d-dim array] Data points
        - overlap_idx: [n-dim array] Index of where head and torso LF overlap
    """
    m = np.shape(accs)[0]

    # Construct a label matrix with given accs and covs
    L = np.zeros((n, m))
    for i in range(n):
        j = int(C[i])
        if np.random.random() < covs[j]:
            if np.random.random() < accs[j]:
                L[i, j] = Y[i]
            else:
                L[i, j] = -Y[i]

    # Change labeling patterns of LF[-2] and LF[-1] so they have some overlap
    for i in range(n):
        j = int(C[i])
        if j == int(np.max(C)):
            if np.random.random() < overlap_portion:
                L[i, j - 1] = -Y[i]  # downvote LF1 on overlap
                L[i, j] = Y[i]  # upvote LF2 on overlap

    overlap_idx = [i for i in range(n) if (L[i, -2] != 0 and L[i, -1] != 0)]
    return L, overlap_idx
