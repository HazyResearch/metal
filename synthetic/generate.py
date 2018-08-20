from collections import defaultdict, Counter
from itertools import chain, product
import os

import numpy as np
from numpy.random import random, choice
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import torch
import networkx as nx

from metal.metrics import accuracy_score, coverage_score
from metal.multitask.task_graph import TaskHierarchy

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

class SingleTaskTreeDepsGenerator(object):
    """Generates a synthetic single-task L and Y matrix with dependencies
    
    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        class_balance: (np.array) each class's percentage of the population
        theta_range: (tuple) The min and max possible values for theta, the
            class conditional accuracy for each labeling source
        edge_prob: edge density in the graph of correlations between sources
        theta_edge_range: The min and max possible values for theta_edge, the
            strength of correlation between correlated sources

    The labeling functions have class-conditional accuracies, and 
    class-unconditional pairwise correlations forming a tree-structured graph.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains.
    """
    def __init__(self, n, m, k=2, class_balance='random', theta_range=(0.1, 1), 
        edge_prob=0.0, theta_edge_range=(0.1,1), **kwargs):
        self.n = n
        self.m = m
        self.k = k

        # Generate correlation structure: edges self.E, parents dict self.parent
        self._generate_edges(edge_prob)

        # Generate class-conditional LF & edge parameters, stored in self.theta
        self._generate_params(theta_range, theta_edge_range)

        # Generate class balance self.p
        if class_balance is None:
            self.p = np.full(k, 1/k)
        elif class_balance == 'random':
            self.p = np.random.random(k)
            self.p /= self.p.sum()
        else:
            self.p = class_balance

        # Generate the true labels self.Y and label matrix self.L
        self._generate_label_matrix()

        # Compute the conditional clique probabilities
        self._get_conditional_probs()

        # Correct output type
        self.L = csr_matrix(self.L, dtype=np.int)
    
    def _generate_edges(self, edge_prob):
        """Generate a random tree-structured dependency graph based on a
        specified edge probability.
    
        Also create helper data struct mapping child -> parent.
        """
        self.G = nx.Graph()
        self.E, self.parent = [], {}
        for i in range(self.m):
            if random() < edge_prob and i > 0:
                p_i = choice(i)
                self.E.append((p_i, i))
                self.parent[i] = p_i
                self.G.add_edge(i,p_i)

    def _generate_params(self, theta_range, theta_edge_range):
        self.theta = defaultdict(float)
        for i in range(self.m):
            for y in range(1, self.k+1):
                t_min, t_max = min(theta_range), max(theta_range)
                self.theta[(i,y)] = (t_max - t_min) * random(self.k) + t_min

        # Choose random weights for the edges
        # Note: modifications to get the correct exponential model family
        #       formulation from the arxiv paper
        te_min, te_max = min(theta_edge_range), max(theta_edge_range)
        for (i,j) in self.E:
            for y1 in range(1, self.k+1):
                for y2 in range(1, self.k+1):
                    #w_ij = (te_max - te_min) * random() + te_min
                    w_ij = (te_max - te_min) * random(self.k) + te_min
                    self.theta[((i, j), y1, y2)] = w_ij
                    #self.theta[((j, i), y1, y2)] = w_ij

        for key in self.theta:
            print(key, " ", self.theta[key])
    

    def get_Z(self, y):
        # verify all partition functions the same:
        '''for i in range(self.m):
            print(np.sum(self.naive_SPA(i,y)))'''

        return np.sum(self.naive_SPA(0,y))

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
                            if (parents[node][0], node) in self.E:
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

    # these are the real probabilities for each node:
    def P_vals_true(self):
        self.p_solo = defaultdict(float)

        for y in range(1, self.k+1):
            Z = self.get_Z(y)

            for i in range(self.m):
                print("Labeler = ", i)
                for val in range(self.k+1):
                    self.p_solo[(i,val,y)] = self.naive_SPA(i,y)[val] / Z
                    print("P(L=", val, ", Y=",y,") = ", self.p_solo[(i,val,y)])
                    
    # these are the real joint probabilities for each pair of nodes:
    def P_joints_true(self):
        self.p_joints = defaultdict(float)
        
        for y in range(1,self.k+1):
            Z = self.get_Z(y)

            for i in range(self.m):
                for j in range(i+1, self.m):
                    print("Labelers = ", (i,j))

                    for val1 in range(self.k+1):
                        for val2 in range(self.k+1):
                            other_nodes = dict()
                            other_nodes[j] = val2
                            self.p_joints[(i,val1,j,val2,y)] = self.naive_SPA(i,y,other_nodes=other_nodes)[val1] / Z
                            self.p_joints[(j,val2,i,val1,y)] = self.p_joints[(i,val1,j,val2,y)]
                            print("P(L_", i, "=", val1, ", L_", j, "=", val2, " | Y = ", y, ") = ", self.p_joints[(i,val1,j,val2,y)])

    def P_conditional(self, i, li, j, lj, y):
        """Compute the conditional probability 
            P_\theta(li | lj, y) 
            = 
            Z^{-1} exp( 
                theta_{i|y} \indpm{ \lambda_i = Y }
                + \theta_{i,j} \indpm{ \lambda_i = \lambda_j }
            )
        In other words, compute the conditional probability that LF i outputs
        li given that LF j output lj, and Y = y, parameterized by
            - a class-conditional LF accuracy parameter \theta_{i|y}
            - a symmetric LF correlation paramter \theta_{i,j}
        """
        return self.p_joints[(i,li,j,lj,y)] / self.p_solo[(j,lj,y)]

    def _generate_true_O(self):
        sz = self.m * (self.k)
        self.O_true = np.zeros([sz, sz])
        for i in range(self.m):
            for j in range(self.m):
                for val1 in range(1,self.k+1):
                    for val2 in range(1,self.k+1):
                        sm = 0
                        for y in range(1,self.k+1):
                            if i == j:
                                if val1 == val2:
                                    sm += self.p_solo[(i, val1, y)] * self.p[y-1]
                            else:
                                sm += self.p_joints[(i,val1,j,val2,y)] * self.p[y-1]
                        
                        self.O_true[i*(self.k)+val1-1, j*(self.k)+val2-1] = sm

    def _generate_true_mu(self):
        sz = self.m * (self.k)
        self.mu_true = np.zeros([sz, self.k])
        
        for i in range(self.m): 
            for val1 in range(1,self.k+1):
                for y in range(1, self.k+1):
                    self.mu_true[i*(self.k)+val1-1, y-1] = self.p_solo[(i,val1,y)]
        
    def _generate_label_matrix(self):
        """Generate an n x m label matrix with entries in {0,...,k}"""
        self.L = np.zeros((self.n, self.m))
        self.Y = np.zeros(self.n, dtype=np.int64)

        self.P_vals_true()
        self.P_joints_true()

        self._generate_true_mu()
        self._generate_true_O()

        print(self.O_true)
        print("\nCondition number = ", np.linalg.cond(self.O_true), "\n")
        print(self.mu_true)

        print(self.p)

        sig = self.O_true - self.mu_true @ np.diag(self.p) @ self.mu_true.T
        print("sig\n", sig)       
        print("\nCondition number = ", np.linalg.cond(sig), "\n")

        print("moment of truth!!!")
        self.sig_inv = np.linalg.inv(sig)
        print(self.sig_inv)

        for i in range(self.n):
            y = choice(self.k, p=self.p) + 1  # Note that y \in {1,...,k}
            self.Y[i] = y
            for j in range(self.m):
                if j in self.parent:
                    p_j = self.parent[j]
                    prob_y = self.P_conditional(j, y, p_j, int(self.L[i, p_j]), y)
                    prob_0 = self.P_conditional(j, 0, p_j, int(self.L[i, p_j]), y)
                else:
                    prob_y = self.p_solo[(j, y, y)]
                    prob_0 = self.p_solo[(j, 0, y)]
                p = np.ones(self.k+1) * (1 - prob_y - prob_0) / (self.k - 1)
                p[0] = prob_0
                p[y] = prob_y
                self.L[i,j] = choice(self.k+1, p=p)

    def _get_conditional_probs(self):
        """Compute the true clique conditional probabilities P(\lC | Y) by
        counting given L, Y; we'll use this as ground truth to compare to.

        Note that this generates an attribute, self.c_probs, that has the same
        definition as returned by `LabelModel.get_conditional_probs`.

        TODO: Can compute these exactly if we want to implement that.
        """
        # TODO: Extend to higher-order cliques again
        self.c_probs = np.zeros((self.m * (self.k+1), self.k))
        for y in range(1,self.k+1):
            Ly = self.L[self.Y == y]
            for ly in range(self.k+1):
                self.c_probs[ly::(self.k+1), y-1] = \
                    np.where(Ly == ly, 1, 0).sum(axis=0) / Ly.shape[0]


class HierarchicalMultiTaskTreeDepsGenerator(SingleTaskTreeDepsGenerator):
    def __init__(self, n, m, theta_range=(0, 1.5), edge_prob=0.0, 
        theta_edge_range=(-1,1)):
        super().__init__(n, m, k=4, theta_range=theta_range, 
            edge_prob=edge_prob, theta_edge_range=theta_edge_range)

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
        