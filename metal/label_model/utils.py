import matplotlib
import numpy as np
import copy
from math import exp
#import cvxpy as cp

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

#Utils for Dependency Learner
def get_deps_from_inverse_sig(J, thresh=0.2):
    deps = []
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if abs(J[i,j]) > thresh:
                deps.append((i,j))
    return deps

def find_largest(O,mu,dim,mask,thresh):
    prod = np.outer(mu, mu)
    J = np.linalg.pinv(np.linalg.inv(O) - prod)

    max_val = 0
    max_ind = (-1,-1)
    J_clean = copy.deepcopy(J)
    for i in range(dim):
        for j in range(dim):
            if abs(J[i,j]) <= thresh:
                J_clean[i,j] = 0
            if (i,j) not in mask and abs(J_clean[i,j]) > max_val:
                max_val = abs(J_clean[i,j])
                max_ind = (i,j)
    return max_val, max_ind, J_clean

def solveMatrixCompletion(O_inv, deps):
    #print("deps: ", deps)
    try: 
        set(deps)
    except:
        assert(0==1,"NOT HASHABLE")


    zeros_set = []
    for i in range(O_inv.shape[0]):
        for j in range(O_inv.shape[1]):
            zeros_set.append((i,j))
    zeros_set = set(zeros_set)
    zeros_set = zeros_set - set(deps)
    zeros_set = list(zeros_set)
    
    #form q
    q = np.zeros((len(zeros_set),),dtype=float)
    M = np.zeros((len(zeros_set),O_inv.shape[0]),dtype=float)
    for ix, z in enumerate(zeros_set):
        M[ix, z[0]] = 1
        M[ix, z[1]] = 1

    for ix, z in enumerate(zeros_set):
        q[ix] = np.log(O_inv[z[0],z[1]]**2)
        
    l = np.linalg.pinv(M) @ q
    return l

def solveMatrixCompletionWithMu(O_inv, deps):
    l = solveMatrixCompletion(O_inv,deps)
    l = np.exp(l)
    z_rec = np.sqrt(l)
    mu = calculate_empirical_mu(z_rec,np.linalg.inv(O_inv))
    return mu

def calculate_empirical_mu(z,O):
    c = 1 + z.dot(O.dot(z.T)) # check this
    mu = 1/np.sqrt(c)*O.dot(z.T)
    return mu

def samplegrid(w, h, n):
    sampled = []
    for i in range(w):
        sampled.append((i,i))
    return sampled
