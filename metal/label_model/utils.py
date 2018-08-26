import matplotlib
import numpy as np

# Avoids a potential error when using matplotlib in virtual envrionments
# https://stackoverflow.com/questions/34977388/matplotlib-runtimeerror-python-
# is-not-installed-as-a-framework
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402 # isort:skip


def compute_mu(L_aug, Y, k, p):
    """Given label matrix L_aug and labels Y, compute the true mu params.

    Args:
        L: (np.array {0,1}) [n, d] The augmented (indicator) label matrix
        Y: (np.array int) [n] The true labels in {1,...,k}
        k: (int) Cardinality
        p: (np.array float) [k] The class balance
    """
    n, d = L_aug.shape
    assert Y.shape[0] == n

    # Compute mu
    mu = np.zeros((d, k))
    for y in range(1, k + 1):
        L_y = L_aug[Y == y]
        mu[:, y - 1] = L_y.sum(axis=0) / L_y.shape[0]
    return mu


def compute_covariance(L_aug, Y, k, p):
    """Given label matrix L_aug and labels Y, compute the covariance.

    Args:
        L: (np.array {0,1}) [n, d] The augmented (indicator) label matrix
        Y: (np.array int) [n] The true labels in {1,...,k}
        k: (int) Cardinality
        p: (np.array float) [k] The class balance
    """
    n, d = L_aug.shape
    assert Y.shape[0] == n
    mu = compute_mu(L_aug, Y, k, p)
    return (L_aug.T @ L_aug) / n - mu @ np.diag(p) @ mu.T


def compute_inv_covariance(L_aug, Y, k, p):
    """Given label matrix L and labels Y, compute the covariance.

    Args:
        L: (np.array) [n, d] The augmented (indicator) label matrix
        Y: (np.array int) [n] The true labels in {1,...,k}
    """
    return np.linalg.inv(compute_covariance(L_aug, Y, k, p))


def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)


def visualize_matrix(X, fig_size=(10, 10)):
    plt.rcParams["figure.figsize"] = fig_size
    plt.imshow(X)
    plt.colorbar()
