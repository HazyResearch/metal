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
        for i in range(1, m):
            if random() < edge_prob:
                p_i = choice(i)
                self.edges.append((p_i, i))
                self.parent[i] = p_i
                self.G.add_edge(i, p_i)

class ChainDependencies(DependenciesGraph):
    """Generate a chain-structured dependency graph."""
    def __init__(self, m, edge_prob=1.0):
        self.G = nx.Graph()
        self.edges = []
        self.parent = {}
        for i in range(1, m):
            p_i = i - 1
            self.edges.append((p_i, i))
            self.parent[i] = p_i
            self.G.add_edge(i, p_i)

###
### Model Params
###
class ModelParameters(object):
    """Helper data structures for the source parameters; note that this object
    essentially defines our model.
    
    This model is the most general form, where each marginal conditional 
    probability for each clique C, P(\lf_C | Y), is generated randomly.

    Args:
        - m: (int) Number of sources
        - k: (int) Cardinality of the problem
        - theta_range: (tuple) The min and max possible values for theta, the
            class conditional accuracy for each labeling source
        - edges: (list) List of tuples of ints representing dependency edges
            between the sources
        - theta_edge_range: The min and max possible values for theta_edge, the
            strength of correlation between correlated sources
    """
    def __init__(self, m, k, theta_range=(0.1, 1), edges=[], 
        theta_edge_range=(0.1,1)):
        self.theta = defaultdict(float)
        for i in range(m):
            for y in range(1, k+1):
                t_min, t_max = min(theta_range), max(theta_range)
                self.theta[(i,y)] = (t_max - t_min) * random(k) + t_min

        # Choose random weights for the edges
        # Note: modifications to get the correct exponential model family
        #       formulation from the arxiv paper
        te_min, te_max = min(theta_edge_range), max(theta_edge_range)
        for (i,j) in edges:
            for y1 in range(0, k+1):
                for y2 in range(0, k+1):
                    w_ij = (te_max - te_min) * random(k) + te_min
                    self.theta[((i, j), y1, y2)] = w_ij

class SimpleModelParameters(object):
    """Helper data structures for the source parameters; note that this object
    essentially defines our model.
    
    This is the simplified model, where:
        - P(\lf_i=y | Y=y) is the same for all y
        - P(\lf_i=y' | Y=y) is the same for all y, y' != y
        - ...
    """
    # TODO
    pass

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
        model_params: (ModelParameters) A ModelParameters object
        deps_graph: (DependenciesGraph) A DependenciesGraph object
            specifying the dependencies structure of the sources
    
    The labeling functions have class-conditional accuracies, and 
    class-unconditional pairwise correlations forming a tree-structured graph.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains.
    """
    def __init__(self, n, m, k=2, class_balance='random', model_params=None, 
        deps_graph=None, **kwargs):
        self.n = n
        self.m = m
        self.k = k

        # Create a dictionary of cached probabilities and partition function
        self.probs_cached = {}
        self.Z_cached = {}

        # Dependencies graph
        self.deps_graph = DependenciesGraph(m) if deps_graph is None \
            else deps_graph
        self.edges = self.deps_graph.edges
        self.parent = self.deps_graph.parent
        self.G = self.deps_graph.G
        self.c_tree = CliqueTree(m, k, self.edges, higher_order_cliques=True)
        
        # Generate class-conditional LF & edge parameters, stored in self.theta
        self.model_params = ModelParameters(m, k, edges=self.edges) \
            if model_params is None else model_params
        self.theta = self.model_params.theta

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

    def naive_SPA(self, i, y, other_nodes=None, verbose=False):
        # this contains our nodes:
        G_i_set = nx.node_connected_component(self.G, i)
        G_i     = self.G.subgraph(G_i_set)
        G_i_bfs = nx.bfs_tree(G_i, i)

        if verbose: print("\n\nDoing a naive SPA for node ", i)
        if verbose: print("our graph ", G_i_bfs.edges())

        # build nodes at each depth of our tree
        # defines our elimination ordering
        dist_list = nx.shortest_path_length(G_i_bfs, source=i)
        depths = dict()
        max_depth = 0
        messages = dict()

        for node in dist_list:
            messages[node] = np.zeros(self.k+1)

            if dist_list[node] not in depths:
                depths[dist_list[node]] = [node]
                if dist_list[node] > max_depth:
                    max_depth = dist_list[node]
            else:
                depths[dist_list[node]].append(node)

        parents = nx.predecessor(G_i_bfs, i)

        if verbose: print("depths = ", depths)
        if verbose: print("parents = ", parents)

        # now we do sum product
        while max_depth > 0:
            for node in depths[max_depth]:
                # compute the message node->parent:
                # this is a function m(parent=val)

                if verbose: print("working on node ", node, " with parent ", parents[node][0])

                for val_p in range(self.k+1):
                    if verbose: print("For val_p = ", val_p)
                    mess = 0

                    if other_nodes is not None and node in other_nodes:
                        val_range = [other_nodes[node]]
                    else:
                        val_range = range(0, self.k+1)

                    for val in val_range:
                        mess_local = 1 # local unary term
                        if val > 0:
                            mess_local = np.exp(self.theta[(node, val)][y-1]) 

                        mess_edge = 1 # local edge term (node, parent)    
                        if val > 0 and val_p > 0:
                            # we need to figure out the exact order of these things, in the original graph:
                            if (parents[node][0], node) in self.edges:
                                mess_edge  = np.exp(self.theta[((parents[node][0], node), val_p, val)][y-1])
                            else:
                                mess_edge  = np.exp(self.theta[((node, parents[node][0]), val, val_p)][y-1])

                        mess_prod  = 1 # product of all incoming messages at node
                        for edge in nx.edges(G_i_bfs, node):
                            if edge[1] != parents[node][0]:
                                if verbose: print("incoming message along edge ", edge[1], " val = ", val, " mess =", messages[edge[1]][val])
                                mess_prod *= messages[edge[1]][val]

                        mess += mess_local * mess_edge * mess_prod # sum
                        if verbose: print("we added ", mess_local * mess_edge * mess_prod, "\n")
                                   
                    messages[node][val_p] = mess

                if verbose: print("finished message at node ", node)
                if verbose: print("message was ", messages[node], "\n")
            max_depth -= 1

        # now we're left with the messages just to i:
        if verbose: print("now at top, getting the final marginal for i = ", i)

        message_i = np.zeros(self.k+1)
        for val in range(0, self.k+1):
            mess_local = 1
            if val > 0:
                mess_local = np.exp(self.theta[(i, val)][y-1])
            mess_prod = 1
            for edge in nx.edges(G_i_bfs, i):
                mess_prod *= messages[edge[1]][val]
                if verbose: print("incoming message along edge ", edge[1], " val = ", val, " mess =", messages[edge[1]][val])
            message_i[val] = mess_local * mess_prod
            
        if verbose: print("Final marginal: ", message_i)
        if verbose: print("\n\n")
        return message_i
    
    def get_Z(self, y):
        """Get or compute the partition function"""
        if y in self.Z_cached:
            return self.Z_cached[y]
        else:
            Z = np.sum(self.naive_SPA(0,y))
            self.Z_cached[y] = Z
            return Z
    
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
        if vals in self.probs_cached:
            return self.probs_cached[vals]
        else:
            if len(lf_idxs) > 1:
                other_nodes = {i:li for i,li in zip(lf_idxs[1:], lf_vals[1:])}
            else:
                other_nodes = None
            # TODO: Cache this?
            Z = self.get_Z(Y)
            i, li = lf_idxs[0], lf_vals[0]
            p = self.naive_SPA(i, Y, other_nodes=other_nodes)[li] / Z
            self.probs_cached[vals] = p
            return p
    
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
        