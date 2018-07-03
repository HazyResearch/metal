from collections import defaultdict, Counter
from itertools import chain
import os

import numpy as np
import numpy.random as random
from scipy.sparse import csc_matrix, lil_matrix
import torch

from metal.metrics import accuracy_score, coverage_score


################################################################################
# Single-Task 
################################################################################

def logistic_fn(x):
    return 1 / (1 + np.exp(-x))

def choose_other_label(k, y):
    """Given a cardinality k and true label y, return random value in 
    {1,...,k} \ {y}."""
    return random.choice(list(set(range(1, k+1)) - set([y])))

def generate_single_task_unipolar(n, m, k=2, alpha_range=[0.6, 0.9],
    beta_range=[0.25, 0.5], class_balance=None, polarity_balance=None, 
    seed=None):
    """Generate a single task unipolar label matrix
    Args:
        n: Number of data points
        m: Number of LFs
        k: Cardinality
        alpha_range: Range of accuracy parameters to sample from uniformly
        beta_range: Range of labeling propensity ranges to sample from uniformly
        class_balance: Class balance of Y
        polarity_balance: Balance of LF polarities
    
    Note that $P(\lambda_i=y | Y=y) = \alpha_i * \beta_i$, which we return as
        the conditional_probs
    
    Generative process:
        - LF alphas, betas, and polarities are chosen randomly
        - For each data point:
            - A label is chosen randomly according to the class balance
            - For each LF j:
                - If Y = p_j, label wp beta[j] * alpha[j]
                - If Y != p_j, label wp beta[j] * (1-alpha[j])
                - Else, abstain
    
    Returns:
        L: Label matrix
        Y: True labels, in {1,...,k}
        metadata: Dictionary of metadata:
            - k: Cardinality
            - alphas: The LF accuracies
            - betas: LF labeling propensities
            - polarities: LF polarities p_i
            - coverage: P(\lambda_i == p_i)
            - cond_probs: P(\lambda_i == p_i | Y == p_i)
    """
    if seed is not None:
        random.seed(seed)

    # Choose LF accuracies and labeling propensities
    alphas = random.rand(m) * (max(alpha_range) - min(alpha_range)) \
        + min(alpha_range)
    betas = random.rand(m) * (max(beta_range) - min(beta_range)) \
        + min(beta_range)
    polarities = random.choice(range(1, k+1), size=m, p=polarity_balance)

    # Generate the true data point labels
    Y = random.choice(range(1, k+1), size=n, p=class_balance)

    # Generate the label matrix
    # Construct efficiently as LIL matrix, then convert to CSC
    L = lil_matrix((n, m), dtype=np.int)
    for i in range(n):
        for j in range(m):
            if random.random() < betas[j]:
                correct = random.random() < alphas[j]
                if (polarities[j] == Y[i] and correct) or \
                    (polarities[j] != Y[i] and not correct):
                    L[i,j] = polarities[j]
    
    # Return L, Y, and metadata dictionary
    metadata = {
        'k': k,
        'alphas': alphas,
        'betas': betas,
        'polarities': polarities,
        'coverages': np.where(L.todense() != 0, 1, 0).sum(axis=0) / n,
        'cond_probs': alphas * betas,
        'class_balance': class_balance or np.ones(k)/k,
    }
    return L.tocsc(), torch.tensor(Y, dtype=torch.short), metadata


def gaussian_bags_of_words(Y, vocab, sigma=1, bag_size=[25, 50]):
    """
    Generate Gaussian bags of words based on label assignments

    Args:
        Y: (Tensor) true labels
        sigma: (float) the standard deviation of the Gaussian distributions
        bag_size: (list) the min and max length of bags of words

    Returns:
        X: (Tensor) a tensor of indices representing tokens
        items: (list) a list of entences (strings)

    The sentences are conditionally independent, given a label.
    Note that technically we use a half-normal distribution here because we 
        take the absolute value of the normal distribution.

    Example:
        TBD

    """
    def make_distribution(sigma, num_words):
        p = abs(np.random.normal(0, sigma, num_words))
        return p / sum(p)
    
    Y = Y.numpy()
    num_words = len(vocab)
    word_dists = {y: make_distribution(sigma, num_words) for y in set(Y)}
    bag_sizes = np.random.choice(range(min(bag_size), max(bag_size)), len(Y))

    X = []
    items = []
    for i, (y, length) in enumerate(zip(Y, bag_sizes)):
        x = torch.from_numpy(
            np.random.choice(num_words, length, p=word_dists[y]))
        X.append(x)
        items.append(' '.join(vocab[j] for j in x))

    return X, items