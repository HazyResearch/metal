from collections import defaultdict, Counter
from itertools import chain
import os

import numpy as np
from numpy.random import random, rand, randint, choice
from scipy.sparse import csc_matrix
import torch

# from metal.structs import TaskTree, SingleTaskTree

def generate_single_task_unipolar(n, m, k=2, min_acc=0.6, max_acc=0.9, 
    beta=0.1, class_balance=None, lf_balance=None):
    """Generate a single task label matrix
    
    acc: of non-abstaining votes, what fraction are correct?
    beta: of items whose labels match my polarity, what fraction do I label?

    True labels take on values in {1,...,k}.

    Example:
        For a given LF of polarity 1 (the 0.3 class):
        n = 2000
        class_balance = [0.3, 0.7]
        acc = 0.6
        beta = 0.1

        There are 600 TPs  		        (n * balance)
        I label 60 of them 		        (n * balance * beta)
        I label 36 of them correctly 	(n * balance * beta * alpha)
        I label 24 of them incorrectly  (n * balance * beta * (1 - alpha))
    """
    Y = choice(range(1, k+1), size=n, p=class_balance)
    accs = rand(m) * (max_acc - min_acc) + min_acc
    if not class_balance:
        class_balance = np.full(shape=k+1, fill_value=1/k)
        class_balance[0] = 0

    betas = np.ones(m) * beta
    pols = choice(range(1, k+1), size=m, p=lf_balance)
    
    rows = []
    cols = []
    data = []

    for i in range(n):
        for j in range(m):
            if Y[i] == pols[j]:
                if random() < class_balance[Y[i]] * betas[j] * accs[j]:
                    rows.append(i)
                    cols.append(j)
                    data.append(pols[j])
            else:
                if random() < class_balance[Y[i]] * betas[j] * (1 - accs[j]):
                    rows.append(i)
                    cols.append(j)
                    data.append(pols[j])

    L = csc_matrix((data, (rows, cols)), shape=(n, m))
    validate_synthetic(L, Y, n, m, k, accs, betas, unipolar=True)

    Y = torch.tensor(Y, dtype=torch.short)
    return L, Y, accs


def validate_synthetic(L, Y, n, m, k, est_accs, est_betas, unipolar=False):
    # Check shape and ranges
    assert(L.shape[0] == n)
    assert(L.shape[1] == m)
    assert(L.min() >= 0)
    assert(L.max() <= k)

    # Check class balances
    # counter = Counter(Y)
    # true_class_balance






# ################################################################################
# # Single-Task 
# ################################################################################

# def generate_single_task(n, m, k=2, min_acc=0.6, max_acc=0.9, beta=0.5,
#     class_balance=None):
#     """Generate a single task label matrix with bipolar LFs
    
#     True labels take on values in {1,...,k}.
#     LFs are _not_ unipolar.
#     """
#     Y = choice(range(1, k+1), p=class_balance, size=n)
#     accs = rand(m) * (max_acc - min_acc) + min_acc
#     betas = np.ones(m) * beta
#     L = np.zeros((n, m))
#     for i in range(n):
#         for j in range(m):

#             # Label correctly with prob acc[j],
#             # else label incorrectly uniformly across the other categories
#             if random() < betas[j]:
#                 if random() < accs[j]:
#                     L[i,j] = Y[i]
#                 else:
#                     L[i,j] = choice(list(set(range(1, k+1)) - set([Y[i]])))

#     # Assemble the trivial task to LFs mapping and return
#     task_to_lfs = { 0 : list(range(m)) }

#     return L, task_to_lfs, accs, Y


# def generate_single_task_unipolar_lfs(n, m, k=2, min_acc=0.6, max_acc=0.9, 
#     beta=0.5, class_balance=None, p_neg=0):
#     """Generate a single task label matrix with all unipolar LFs
    
#     True labels take on values in {1,...,k}.

#     In this special case, however, LFs only ever vote one way or the other.
#     This reflects many cases in real usage, and seems to produce a spurious
#     minima (namely, just decide one class always right, all LFs are either
#     perfect or terrible).

#     If p_neg > 0, randomly vote a different way instead of abstaining with this
#     probability.
#     """
#     Y = choice(range(1, k+1), p=class_balance, size=n)
#     accs = rand(m) * (max_acc - min_acc) + min_acc
#     polarities = choice(range(1, k+1), size=m)
#     betas = np.ones(m) * beta
#     L = np.zeros((n, m))
#     print("NOTE: This may be using a faulty generative process. Use with caution!")
#     for i in range(n):
#         for j in range(m):
#             p = polarities[j]

#             # Note: The below generative process is a bit messed up, should
#             # rethink how to do this... reservoir sampling?

#             # Determine according to propensity whether or not to label at all
#             if random() < betas[j]:
#                 if p == Y[i]:
#                     if random() < accs[j]: 
#                         L[i,j] = Y[i]
#                 else:
#                     r = random()
#                     if r > accs[j]:
#                         L[i,j] = p
#                     elif r < p_neg:
#                         L[i,j] = choice(list(set(range(1, k+1)) - set([p])))

#     # Assemble the trivial task to LFs mapping and return
#     task_to_lfs = { 0 : list(range(m)) }
#     return L, task_to_lfs, accs, Y, polarities


# ################################################################################
# # Multi-Task 
# ################################################################################

# def generate_binary_tree(d, n, params):
#     """Generate a balanced binary tree of depth d, encoding 2^d classes.
#     Input is:
#         - `n`: Number of data points
#         - `d`: Depth of the tree; root node alone is considered depth 1
#         - `params`: A list of dictionaries of parameters for T tasks/nodes,
#             where the nodes are in depth-first order.
#     """
#     T = 2**d - 1

#     # Unpack params
#     if T != len(params):
#         raise ValueError(f'Params is length {len(params)}, while T={T}.')
#     m = [params[t]['m'] for t in range(T)]
#     acc_range = [(params[t]['min_acc'], params[t]['max_acc']) for t in range(T)]
#     beta = [params[t]['beta'] for t in range(T)]
#     null_acc = [params[t]['null_acc'] for t in range(T)]

#     # Generate edge set and maps
#     edges = add_children(0, d)

#     # Initialize helper data structures for tree traversal
#     children = defaultdict(list)
#     parents = defaultdict(list)
#     for s, t in edges:
#         children[s].append(t)
#         parents[t].append(s)
#     # Ensure that the child nodes are sorted by index
#     # Note that this means that value k of a node corresponds to child k-1
#     for t, cs in children.items():
#         children[t] = sorted(list(set(cs)))
#     # Get the ordered list of leaf task indices
#     leaf_nodes = [t for t in range(T) if all(e[0] != t for e in edges)]

#     # Initialize values
#     Y = randint(1, 2**d + 1, size=n)
#     Yt = np.zeros((T, n), dtype=int)
#     accs = [random(m[t]) * (acc_range[t][1] - acc_range[t][0]) + acc_range[t][0]
#         for t in range(T)]
#     L = np.zeros((n, sum(m)))
    
#     # Assemble the tasks -> LFs mapping
#     task_to_lfs = {}
#     m_offset = 0
#     for t in range(T):
#         task_to_lfs[t] = list(range(m_offset, m_offset + m[t]))
#         m_offset += m[t]

#     # For each data point:
#     for i in range(n):
        
#         # Get the active leaf node t and the node-level label yt
#         t = leaf_nodes[int(np.ceil(Y[i] / 2)) - 1]
#         Yt[t,i] = 2 if Y[i] % 2 == 0 else 1

#         # Set LF outputs on the path from active leaf node to root
#         active_nodes = set([t])
#         while True:

#             # Set LF labels for x_i at node t, given y is in this subtree
#             for ji, j in enumerate(task_to_lfs[t]):
#                 if random() < beta[t]:
#                     L[i,j] = Yt[t,i] if random() < accs[t][ji] else 3 - Yt[t,i]

#             # Traverse up a level
#             tp = t
#             if len(parents[tp]) == 0:
#                 break
#             t = parents[tp][0]
#             active_nodes.add(t)
#             Yt[t,i] = children[t].index(tp) + 1

#         # Set all LF outputs on nodes not on the active path
#         for t in set(range(T)) - active_nodes:
#             for j in task_to_lfs[t]:
#                 if random() > null_acc[t]:
#                     L[i,j] = choice([1,2])

#     return L, task_to_lfs, edges, accs, Y, Yt


#     def generate_uniform_binary_tree(d, n, m, min_acc=0.4, max_acc=0.8, beta=0.5, 
#         null_acc=0.85):
#         params = [
#             {
#                 'm' : m,
#                 'min_acc' : min_acc,
#                 'max_acc' : max_acc,
#                 'beta' : beta,
#                 'null_acc' : null_acc
#             } for _ in range(2**d - 1)
#         ]
#         return generate_binary_tree(d, n, params)