from collections import defaultdict, Counter
from itertools import chain, product
import os

import numpy as np
from numpy.random import random, choice
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import torch
import networkx as nx
import mpmath

from metal.metrics import accuracy_score, coverage_score
from metal.multitask.task_graph import TaskHierarchy
from metal.label_model import CliqueTree

from synthetic.words1k import vocab1k

def singletask_synthetic(n, m, k, **kwargs):
    data = SingleTaskTreeDepsGenerator(n, m, k, **kwargs)

    L = data.L
    Y = data.Y
    deps = data.E

    bags, D = gaussian_bags_of_words(Y, vocab1k, **kwargs) 
    X = bags_to_counts(bags, len(vocab1k))

    return D, L, X, Y, deps

############################# Generating Ls and Ys #############################

def indpm(x, y):
    """Plus-minus indicator function"""
    return 1 if x == y else -1

def indpm_0(x, y):
    """Plus-minus indicator function, special handling of zero values"""
    if x == 0 or y == 0:
        return 0
    else:
        return 1 if x == y else -1

def ind_0(x, y):
    """Indicator function, special handling of zero values"""
    return 1 if x == y and x != 0 and y != 0 else 0

###
### Dependencies Graphs
###
class DependenciesGraph(object):
    """Helper data structures for source dependencies graph"""
    def __init__(self, m):
        self.G = nx.Graph()
        self.edges = []
        self.parent = {}

class TreeDependencies(DependenciesGraph):
    """Generate a random tree-structured dependency graph based on a
    specified edge probability.
    """
    def __init__(self, m, edge_prob=1.0):
        self.G = nx.Graph()
        self.edges = []
        self.parent = {}
        self.children = defaultdict(set)
        for i in range(1, m):
            if random() < edge_prob:
                p_i = choice(i)
                self.edges.append((p_i, i))
                self.parent[i] = p_i
                self.children[p_i].add(i)
                self.G.add_edge(i, p_i)

class YPlusTreeDependencies(DependenciesGraph):
    """Generate a random tree-structured dependency graph based on a
    specified edge probability, but include Y and an edge between
    every node and Y
    """
    def __init__(self, m, edge_prob=1.0):
        self.G = nx.Graph()
        self.edges = []
        self.parent = {}
        self.children = defaultdict(set)

        # node m is Y:
        for i in range(1, m):
            if random() < edge_prob:
                p_i = choice(i)
                self.edges.append((p_i, i))
                self.edges.append((m, i))
                self.parent[i] = p_i
                self.children[p_i].add(i)
                self.G.add_edge(i, p_i)
                self.G.add_edge(m, i)

class ChainDependencies(DependenciesGraph):
    """Generate a chain-structured dependency graph."""
    def __init__(self, m, edge_prob=1.0):
        self.G = nx.Graph()
        self.edges = []
        self.parent = {}
        self.children = defaultdict(set)
        for i in range(1, m):
            p_i = i - 1
            self.edges.append((p_i, i))
            self.parent[i] = p_i
            self.children[p_i].add(i)
            self.G.add_edge(i, p_i)

###
### DATA GENERATORS
###
class DataGenerator(object):
    """Generates a synthetic single-task L and Y matrix with dependencies
    
    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        class_balance: (np.array) each class's percentage of the population
        deps_graph: (DependenciesGraph) A DependenciesGraph object
            specifying the dependencies structure of the sources
        param_ranges: (dict) A dictionary of ranges to draw the model parameters
            from:
            - theta_range: (tuple) The min and max possible values for theta, 
                the class conditional accuracy for each labeling source
            - theta_edge_range: The min and max possible values for theta_edge, 
                the strength of correlation between correlated sources
    
    The labeling functions have class-conditional accuracies, and 
    class-unconditional pairwise correlations forming a tree-structured graph.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains.
    """
    def __init__(self, n, m, k=2, class_balance='random', deps_graph=None,
        param_ranges={'theta_range': (0.1,1), 'theta_edge_range': (0.1,1)}, 
        **kwargs):
        self.n = n
        self.m = m
        self.k = k

        # Create a dictionary of cached probabilities and partition function
        self.probs_cached = {}
        self.Z_cached = {}
        self.SPA_msg_cache = {}

        # Dependencies graph
        self.deps_graph = DependenciesGraph(m) if deps_graph is None \
            else deps_graph
        self.edges = self.deps_graph.edges
        self.parent = self.deps_graph.parent
        self.children = self.deps_graph.children
        self.G = self.deps_graph.G
        self.c_tree = CliqueTree(m, k, self.edges, higher_order_cliques=True)
        
        # Generate class-conditional LF & edge parameters, stored in self.theta
        self.theta = self._generate_params(param_ranges)

        # Generate class balance self.p
        if class_balance is None:
            self.p = np.full(k, 1/k)
        elif class_balance == 'random':
            self.p = np.random.random(k)
            self.p /= self.p.sum()
        else:
            self.p = class_balance
        
        # Generate O, mu, Sigma, Sigma_inv
        Y = 1 # Note: we pick an arbitrary Y here, since assuming doesn't matter
        self.O = self._generate_O_Y(Y=Y)
        self.mu = self._generate_mu_Y(Y=Y)
        self.sig, self.sig_inv = self._generate_sigma(self.O, self.mu)

        # Generate the true labels self.Y and label matrix self.L
        self._generate_label_matrix()
    
    def _generate_params(self, param_ranges):
        """This function generates the parameters of the data generating model

        Note that along with the potential functions of the SPA algorithm, this
        essentially defines our model. This model is the most general form,
        where each marginal conditional probability for each clique C, 
        P(\lf_C | Y), is generated randomly.
        """
        theta = defaultdict(float)

        # Unary parameters
        theta_range = param_ranges['theta_range']
        t_min, t_max = min(theta_range), max(theta_range)
        for i in range(self.m):
            for y in range(1, self.k+1):
                theta[(i,y)] = (t_max - t_min) * random(self.k) + t_min

        # Choose random weights for the edges
        theta_edge_range = param_ranges['theta_edge_range']
        te_min, te_max = min(theta_edge_range), max(theta_edge_range)
        for (i,j) in self.edges:
            for y1 in range(0, self.k+1):
                for y2 in range(0, self.k+1):
                    w_ij = (te_max - te_min) * random(self.k) + te_min
                    theta[((i, j), y1, y2)] = w_ij
                    theta[((j, i), y2, y1)] = w_ij
        return theta
    
    def _node_factor(self, i, val_i, y):
        return np.exp(self.theta[(i, val_i)][y-1]) if val_i > 0 else 1
    
    def _edge_factor(self, i, j, val_i, val_j, y):
        if val_i > 0 and val_j > 0:
            return np.exp(self.theta[((i, j), val_i, val_j)][y-1])
        else:
            return 1
    
    def SPA(self, targets, y):
        """Returns the marginal probability of source labels in targets,
        conditioned on Y=y. Computes recursively and caches intermediates, using
        a tree structure, with the source indices being topologically ordered.

        NOTE: This function only works for trees / forests, i.e. triangulated
        graphs with largest maximal clique size = 2!
        TODO: Extend to use the CliqueTree for arbitrary graphs...

        Args:
            - targets: (dict) A dictionary of (source index, value) to return
                the marginal probability for
            - y: (int) Value of Y to condition on
        """
        Z = sum([self._SPA_unnormalized({0:l}, y) for l in range(self.k+1)])
        return self._SPA_unnormalized(targets, y) / Z

    def _SPA_unnormalized(self, targets, y):
        # Pick the first value in targets as the root
        i = sorted(list(targets.keys()))[0]
        val_i = targets[i]

        # Compute the local messages (unnormalized potential) recursively
        msg = 1
        msg *= self._node_factor(i, val_i, y)
        for c in self.G.neighbors(i):
            msg *= self._SPA_message(targets, y, c, i, val_i)
        return msg

    def _SPA_message(self, targets, y, i, j, val_j):
        """Computes the sum-product message from node i --> j"""
        # Form cache key and check cache
        # TODO: This is a bit clunky, clean up?
        cache_key = (
            tuple(targets.keys()), tuple(targets.values()), y, i, j, val_j
        )
        if cache_key in self.SPA_msg_cache:
            return self.SPA_msg_cache[cache_key]
        
        # Sum over the values of node i
        msg = 0
        vals_i = [targets[i]] if i in targets else range(self.k+1)
        for val_i in vals_i:
            msg_val_i = 1
        
            # Compute the local message for current node i
            msg_val_i *= self._node_factor(i, val_i, y)

            # Multiply the message from i --> j
            msg_val_i *= self._edge_factor(i, j, val_i, val_j, y)

            # Recursively compute the messages from children
            for c in set(self.G.neighbors(i)) - {j}:
                msg_val_i *= self._SPA_message(targets, y, c, i, val_i)
            msg += msg_val_i
        
        # Cache result then return it
        self.SPA_msg_cache[cache_key] = msg
        return msg
    
    def P_cond(self, lf_idxs, lf_vals, Y=None):
        """Returns the marginal probability of a set of LFs taking on a set of
        values, conditioned on a value of the true label Y.
        
        Computes lazily: If value has not yet been computed, computes using 
        naive SPA and caches the value.

        Args:
            - lf_idxs: An int or tuple of ints corresponding to LF indices, i.e.
                in {0,...,m-1}
            - lf_vals: An int or tuple of ints corresponding to LF values, i.e.
                in {0,1,...,k}
            - Y: An int in {1,...,k}, or None; if none, returns the 
                unconditional marginal probability of the LFs taking on lf_vals.
        """
        # TODO: Maybe expand this fn to take in two dicts, one for args, the 
        # other for conditioned on?s

        # Convert all inputs to a tuple of values
        if isinstance(lf_idxs, int):
            lf_idxs = (lf_idxs,)
        if isinstance(lf_vals, int):
            lf_vals = (lf_vals,)
        
        # TODO: Make sure lf_idxs are sorted
        sort_order = list(np.argsort(lf_idxs))
        lf_idxs = tuple(np.array(lf_idxs)[sort_order])
        lf_vals = tuple(np.array(lf_vals)[sort_order])
        
        # If Y = None, marginalize recursively
        if Y is None:
            return np.sum([ self.p[y-1] * self.P_cond(lf_idxs, lf_vals, Y=y) 
                for y in range(1, self.k + 1)])

        # Get or compute the probability
        vals = lf_idxs + lf_vals + (Y,)
        if vals not in self.probs_cached:
            targets = {i:li for i,li in zip(lf_idxs, lf_vals)}
            self.probs_cached[vals] = self.SPA(targets, Y)
        return self.probs_cached[vals]
    
    def P(self, lf_idxs, lf_vals, Y):
        """Returns the marginal probability of a set of LFs taking on a set of
        values *and* a value of the true label Y.
        
        Computes lazily: If value has not yet been computed, computes using 
        naive SPA and caches the value.

        Args:
            - lf_idxs: An int or tuple of ints corresponding to LF indices, i.e.
                in {0,...,m-1}
            - lf_vals: An int or tuple of ints corresponding to LF values, i.e.
                in {0,1,...,k}
            - Y: An int in {1,...,k}
        """
        return self.P_cond(lf_idxs, lf_vals, Y=Y) * self.p[Y-1]
    
    def _generate_O_Y(self, Y=None):
        """Generates the matrix O = E[\psi \psi^T | Y], where \psi is the set of
        generalized clique indicator statistics set by self.c_tree.

        Args:
            - Y: If Y is None, marginalizes out Y
        """
        O = np.zeros([self.c_tree.d, self.c_tree.d])
        for i, c1 in enumerate(self.c_tree.iter_index()):
            for j, c2 in enumerate(self.c_tree.iter_index()):

                # Check for inconsistent values of the same source
                consistent = True
                for idx in set(c1.keys()) & set(c2.keys()):
                    if c1[idx] != c2[idx]:
                        consistent = False
                if not consistent:
                    continue

                # Get the probability of the union of the clique values
                u = {**c1, **c2}
                O[i,j] = self.P_cond(tuple(u.keys()), tuple(u.values()), Y=Y)
        return O
    
    def _generate_O(self):
        """Generates the matrix O = E[\psi \psi^T], where \psi is the set of
        generalized clique indicator statistics set by self.c_tree, and now
        includes Y- i.e. these are the joint marginals as entries.
        """
        # TODO!
        raise NotImplementedError()
    
    def _generate_mu_Y(self, Y=None):
        """Generates the vector \mu = E[\psi | Y], where \psi is the set of
        generalized clique indicator statistics set by self.c_tree.

        Args:
            - Y: If Y is None, marginalizes out Y
        """
        mu = np.zeros((self.c_tree.d, 1))
        for i, c1 in enumerate(self.c_tree.iter_index()):
            mu[i, 0] = self.P_cond(tuple(c1.keys()), tuple(c1.values()), Y=Y)
        return mu
    
    def _generate_sigma(self, O, mu):
        sigma = O - mu @ mu.T
        sigma_inv = np.linalg.inv(sigma)
        return sigma, sigma_inv
        
    def _generate_label_matrix(self):
        """Generate an n x m label matrix with entries in {0,...,k}"""
        self.L = np.zeros((self.n, self.m))
        self.Y = np.zeros(self.n, dtype=np.int64)
        
        for i in range(self.n):
            y = choice(self.k, p=self.p) + 1  # Note that y \in {1,...,k}
            self.Y[i] = y
            for j in range(self.m):
                if j in self.parent:
                    p_j = self.parent[j]
                    l_p_j = int(self.L[i,p_j])
                    prob_y = self.P_cond((j, p_j), (y, l_p_j), y) \
                        / self.P_cond(p_j, l_p_j, y)
                    prob_0 = self.P_cond((j, p_j), (0, l_p_j), y) \
                        / self.P_cond(p_j, l_p_j, y)
                else:
                    prob_y = self.P_cond(j, y, y)
                    prob_0 = self.P_cond(j, 0, y)
                p = np.ones(self.k+1) * (1 - prob_y - prob_0) / (self.k - 1)
                p[0] = prob_0
                p[y] = prob_y
                self.L[i,j] = choice(self.k+1, p=p)
        
        # Correct output type
        self.L = csr_matrix(self.L, dtype=np.int)

class SimpleDataGenerator(DataGenerator):
    """Generates a synthetic single-task L and Y matrix with dependencies, using
    the simplified model of the Data Programming (NIPS 2016) paper, where:
        - Each LF chooses to label or abstain *independently*, determined by a
            single labeling propensity factor
        - Each LF has a singe accuracy parameter
        - Each (LF, LF) dependency has a single edge parameter
    
    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        class_balance: (np.array) each class's percentage of the population
        deps_graph: (DependenciesGraph) A DependenciesGraph object
            specifying the dependencies structure of the sources
        param_ranges: (dict) A dictionary of ranges to draw the model parameters
            from:
            - theta_lp_range: (tuple) The min and max possible values for 
                theta_lp, which controls the labeling propensity of each LF
            - theta_acc_range: (tuple) The min and max possible values for 
                theta_acc, which controls the accuracy of each LF
            - theta_edge_range: The min and max possible values for theta_edge, 
                the strength of correlation between correlated sources
    
    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains.
    """
    def __init__(self, n, m, k=2, class_balance='random', deps_graph=None,
        param_ranges={'theta_lp_range': (0.5,0.75), 'theta_acc_range': (0.1,1), 
        'theta_edge_range': (0.1,1)}, **kwargs):
        super().__init__(n, m, k=k, class_balance=class_balance,
            deps_graph=deps_graph, param_ranges=param_ranges, **kwargs)

    def _generate_params(self, param_ranges):
        """This function generates the parameters of the data generating model

        Note that along with the potential functions of the SPA algorithm, this
        essentially defines our model. This model is the simple form from the DP
        paper.
        """
        theta = defaultdict(float)

        # Unary parameters
        # Each entry is (labeling propensity, accuracy)
        theta_lp_range = param_ranges['theta_lp_range']
        t_lp_min, t_lp_max = min(theta_lp_range), max(theta_lp_range)
        theta_acc_range = param_ranges['theta_acc_range']
        t_acc_min, t_acc_max = min(theta_acc_range), max(theta_acc_range)
        for i in range(self.m):
            theta[i] = (
                (t_lp_max - t_lp_min) * random() + t_lp_min,
                (t_acc_max - t_acc_min) * random() + t_acc_min
            )

        # Choose random weights for the edges
        theta_edge_range = param_ranges['theta_edge_range']
        te_min, te_max = min(theta_edge_range), max(theta_edge_range)
        for (i,j) in self.edges:
            w_ij = (te_max - te_min) * random() + te_min
            theta[(i, j)] = w_ij
            theta[(j, i)] = w_ij
        return theta
    
    def _node_factor(self, i, val_i, y):
        # return np.exp(theta_lp * (val_i != 0) + theta_acc * (val_i == y))
        # Handle abstains in P_cond
        theta_lp, theta_acc = self.theta[i]
        return (val_i != 0) * np.exp(theta_acc * (val_i == y))

    def _edge_factor(self, i, j, val_i, val_j, y):
        # return np.exp(self.theta[(i,j)] * (val_i == val_j))
        return (val_i != 0 and val_j != 0) * \
            np.exp(self.theta[(i,j)] * (val_i == val_j))
    
    def P_cond(self, lf_idxs, lf_vals, Y=None):
        # Convert all inputs to a tuple of values
        if isinstance(lf_idxs, int):
            lf_idxs = (lf_idxs,)
        if isinstance(lf_vals, int):
            lf_vals = (lf_vals,)

        # Get the probabilities of the *non-zero* values
        lf_idxs_nnz = []
        lf_vals_nnz = []
        for i, val_i in zip(lf_idxs, lf_vals):
            if val_i != 0:
                lf_idxs_nnz.append(i)
                lf_vals_nnz.append(val_i)
        
        if len(lf_idxs_nnz) > 0:
            p = super().P_cond(lf_idxs_nnz, lf_vals_nnz, Y=Y)
        else:
            p = 1.0
        
        # Independent probabilities of labeling
        for i, val_i in zip(lf_idxs, lf_vals):
            p *= self.theta[i][0] if val_i != 0 else 1 - self.theta[i][0]
        return p

class HierarchicalMultiTaskDataGenerator(DataGenerator):
    def __init__(self, n, m, theta_range=(0.1, 1), 
        deps_graph=None, theta_edge_range=(0.1,1), **kwargs):
        super().__init__(n, m, k=4, theta_range=theta_range, 
            deps_graph=deps_graph, theta_edge_range=theta_edge_range)

        # Convert label matrix to tree task graph
        self.task_graph = TaskHierarchy(
            edges=[(0,1), (0,2)],
            cardinalities=[2,2,2]
        )
        L_mt = [np.zeros((self.n, self.m)) for _ in range(self.task_graph.t)]
        fs = list(self.task_graph.feasible_set())
        for i in range(self.n):
            for j in range(self.m):
                if self.L[i,j] > 0:
                    y = fs[int(self.L[i,j])-1]
                    for s in range(self.task_graph.t):
                        L_mt[s][i,j] = y[s]
        self.L = list(map(csr_matrix, L_mt))


############################# Generating Xs and Ds #############################

def gaussian_bags_of_words(Y, vocab=vocab1k, sigma=1, bag_size=[25, 50], 
    **kwargs):
    """
    Generate Gaussian bags of words based on label assignments

    Args:
        Y: np.array of true labels
        sigma: (float) the standard deviation of the Gaussian distributions
        bag_size: (list) the min and max length of bags of words

    Returns:
        X: (Tensor) a tensor of indices representing tokens
        D: (list) a list of sentences (strings)

    The sentences are conditionally independent, given a label.
    Note that technically we use a half-normal distribution here because we 
        take the absolute value of the normal distribution.

    Example:
        TBD

    """
    def make_distribution(sigma, num_words):
        p = abs(np.random.normal(0, sigma, num_words))
        return p / sum(p)
    
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

def bags_to_counts(bags, vocab_size):
    X = torch.zeros(len(bags), vocab_size, dtype=torch.float)
    for i, bag in enumerate(bags):
        for word in bag:
            X[i, word] += 1
    return X
        