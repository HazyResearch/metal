from itertools import product, chain, permutations
from collections import OrderedDict

import numpy as np
from scipy.sparse import issparse, csc_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mpmath

from metal.analysis import (
    plot_probabilities_histogram,
    lf_summary,
    confusion_matrix,
)
from metal.classifier import Classifier
from metal.label_model.lm_defaults import lm_default_config
from metal.label_model.graph_utils import get_clique_tree
from metal.utils import recursive_merge_dicts


class CliqueTree(object):
    """A data structure for representing a set of m labeling functions with
    cardinality k, and their dependencies, as a clique tree.

    Args:
        m: (int) Number of labeling functions
        k: (int) Cardinality of the classification problem
        edges: (list of tuples of ints) Edges (i,j) between labeling functions 
            indicating a conditional dependency between LF i and LF j
    """
    def __init__(self, m, k, edges, higher_order_cliques=True):
        self.m = m
        self.k = k
        self.edges = edges
        self.c_tree = get_clique_tree(range(self.m), self.edges)
        self._build_c_data(higher_order_cliques=higher_order_cliques)

    def _build_c_data(self, higher_order_cliques=True):
        """Create a helper data structure which maps cliques (as tuples of 
        member sources) --> {start_index, end_index, maximal_cliques}, where
        the last value is a set of indices in this data structure.
        """
        self.c_data = OrderedDict()

        # Add all the unary cliques
        for i in range(self.m):
            self.c_data[(i,)] = {
                'start_index': i * self.k,
                'end_index': (i+1) * self.k,
                'max_cliques': set([j for j in self.c_tree.nodes() 
                    if i in self.c_tree.node[j]['members']]),
                'size': 1,
                'members': {i}
            }
        
        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        start_index = self.m * self.k
        if higher_order_cliques:
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.node[item]
                    C_type = 'node'
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                    C_type = 'edge'
                else:
                    raise ValueError(item)
                
                # Important to sort here!!
                members = sorted(list(C['members']))
                nc = len(members)
                id = tuple(members)

                # Check if already added
                if id in self.c_data:
                    continue

                if nc > 1:
                    w = self.k ** nc
                    self.c_data[id] = {
                        'start_index': start_index,
                        'end_index': start_index + w,
                        'max_cliques': set([item]) if C_type == 'node' 
                            else set(item),
                        'size': nc,
                        'members': set(members)
                    }
                    start_index += w
        self.d = start_index
    
    def iter_index(self):
        """Iterates over the (clique_members, values) indices"""
        for c, c_data in self.c_data.items():
            for vals in product(range(1, self.k+1), repeat=c_data['size']):
                yield (c, vals)


class LabelModel(Classifier):
    """A LabelModel...TBD

    Args:
        k: (int) the cardinality of the classifier
        class_balance: (np.array) each class's percentage of the population
    """
    def __init__(self, k=2, class_balance=None, **kwargs):
        config = recursive_merge_dicts(lm_default_config, kwargs)
        super().__init__(k, config)

        self._set_class_balance(class_balance)

    def _set_class_balance(self, class_balance):
        # Class balance- assume uniform if not provided
        if class_balance is None:
            self.p = (1/self.k) * np.ones(self.k)
        else:
            self.p = class_balance
        self.P = torch.diag(torch.from_numpy(self.p)).float()
    
    def _create_L_ind(self, L):
        L_ind = np.zeros((self.n, self.m * self.k))
        for y in range(1, self.k+1):
            L_ind[:, y-1::self.k] = np.where(L == y, 1, 0)     
        return L_ind   

    def _get_augmented_label_matrix(self, L):
        """Returns an augmented version of L where each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.
        
        Args:
            - L: A dense n x m numpy array, where n is the number of data points
                and m is the number of sources, with values in {0,1,...,k}
        """
        # TODO: Be clear how to pass in the higher-order and all-cliques opts!
        L_aug = np.zeros((self.n, self.c_tree.d))

        # Create the basic indicator version of the source label matrix
        L_ind = self._create_L_ind(L)

        if self.config['all_unary_cliques']:
            L_aug[:, :self.m * self.k] = np.copy(L_ind)
        
        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if self.config['higher_order_cliques']:
            for c, c_data in self.c_data.items():
                si, ei = c_data['start_index'], c_data['end_index']
                nc = c_data['size']
                if nc > 1:
                    L_C = np.ones((self.n, self.k ** nc))
                    for i, vals in enumerate(product(range(self.k), repeat=nc)):
                        for j, v in enumerate(vals):
                            L_C[:,i] *= L_ind[:,c_data['members'][j]*self.k + v]
                    L_aug[:, si:ei] = L_C
        return L_aug
    
    def _generate_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        L_aug = self._get_augmented_label_matrix(L, offset=1)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy( L_aug.T @ L_aug / self.n ).float()
    
    def _generate_O_inv(self, L):
        """Form the *inverse* overlaps matrix"""
        self._generate_O(L)

        # If caching enabled, test to see if O_inv has already been computed
        if self.config['cache_O_inv'] and hasattr(self, 'O_inv') \
            and self.O_inv is not None and self.O_inv.shape == self.O.shape:
            err = torch.mean(torch.abs(self.O @ self.O_inv - torch.eye(self.d)))
            if err < self.config['cache_O_inv_thresh']:
                if self.config['verbose']:
                    print("Using cached O_inv...")
            else:
                self.O_inv = None
        else:
            self.O_inv = None

        # Print warning if O is poorly conditioned
        kappa_O = np.linalg.cond(self.O.numpy())
        if kappa_O > self.config['kappa_warning_thresh']:
            print(f"Warning: O is ill-conditioned: kappa(O) = {kappa_O:0.2f}.")

        # Use high-precision matrix operations starting with L.T @ L...
        if self.O_inv is None:
            if self.config['verbose']:
                print("Computing O^{-1}...")
            L_aug = self._get_augmented_label_matrix(L, offset=1)
            with mpmath.workdps(self.config['O_inv_prec']):
                O_unnorm = mpmath.matrix(L_aug.T @ L_aug)
                n = mpmath.mpf(self.n)
                O_inv = (O_unnorm / n) ** -1
                self.O_inv = torch.from_numpy(
                    np.array(O_inv.tolist(), dtype=float)).float()
        
        # self.O_inv = torch.from_numpy(np.linalg.inv(self.O.numpy())).float()

        # Trying the pseudoinverse, dropping singular values that are too small
        # eps = 1e-2
        # O = self.O.numpy()
        # U, s, V = np.linalg.svd(O)
        # S = np.diag(1/s)
        # S[1/S < eps] = 0
        # self.O_inv = torch.from_numpy(V.T @ S @ U.T).float()
        
    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint"""
        self.mask = torch.ones(self.d, self.d).byte()
        for members_i, ci in self.c_tree.c_data.items():
            si, ei = ci['start_index'], ci['end_index']
            for members_j, cj in self.c_tree.c_data.items():
                sj, ej = cj['start_index'], cj['end_index']

                # Check if ci and cj are part of the same maximal clique
                # If so, mask out their corresponding blocks in O^{-1}
                if len(ci['max_cliques'].intersection(cj['max_cliques'])) > 0:
                    self.mask[si:ei, sj:ej] = 0
                    self.mask[sj:ej, si:ei] = 0
                
                # Also try masking out any overlaps
                # mi = set(members_i) if isinstance(members_i, tuple) else \
                #     set([members_i])
                # mj = set(members_j) if isinstance(members_j, tuple) else \
                #     set([members_j])
                # if len(mi.intersection(mj)) > 0:
                #     self.mask[si:ei, sj:ej] = 0
                #     self.mask[sj:ej, si:ei] = 0
        
    def _init_params(self):
        """Initialize the learned params
        
        - \mu is the primary learned parameter, where each row corresponds to 
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:
        
            self.mu[i*self.k + j, y] = P(\lambda_i = j | Y = y)
        
        and similarly for higher-order cliques.
        - Z is the inverse form version of \mu.
        """
        # Initialize mu so as to break basic reflective symmetry
        self.mu_init = torch.randn(self.d, self.k)
        for members_i, ci in self.c_tree.c_data.items():
            si, ei = ci['start_index'], ci['end_index']

            # Unary cliques
            if ci['size'] == 1:
                self.mu_init[si:ei, :] = torch.eye(self.k) * np.random.random()

            # Higher-order cliques
            # TODO
        
        self.mu = nn.Parameter(self.mu_init.clone()).float()

        if self.inv_form:
            self.Z = nn.Parameter(torch.randn(self.d, self.k)).float()

        # Build the mask over O^{-1}
        # TODO: Put this elsewhere?
        self._build_mask()

    def predict_proba(self, L):
        """Returns the [n,k] matrix of label probabilities P(Y | \lambda)"""
        # TODO: Change internals to use sparse throughout and delete this:
        if issparse(L):
            L = L.todense()     

        self._set_constants(L)               
        
        L_aug = self._get_augmented_label_matrix(L, offset=1)     
        mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)

        # Create a "junction tree mask" over the columns of L_aug / mu
        if len(self.deps) > 0:
            jtm = np.zeros(L_aug.shape[1])

            # All maximal cliques are +1
            for i in self.c_tree.c_tree.nodes():
                node = self.c_tree.c_tree.node[i]
                jtm[node['start_index']:node['end_index']] = 1

            # All separator sets are -1
            for i, j in self.c_tree.c_tree.edges():
                edge = self.c_tree.c_tree[i][j]
                jtm[edge['start_index']:edge['end_index']] = 1
        else:
            jtm = np.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp( L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p) )
        Z = np.tile(X.sum(axis=1).reshape(-1,1), self.k)
        return X / Z

    def loss_inv_Z(self, l2=0.0):
        loss_1 = torch.norm((self.O_inv + self.Z @ self.Z.t())[self.mask])**2
        loss_l2 = torch.norm(self.Z)**2
        return loss_1 + l2 * loss_l2
    
    def get_Q(self):
        """Get the model's estimate of Q = \mu P \mu^T
        
        We can then separately extract \mu subject to additional constraints,
        e.g. \mu P 1 = diag(O).
        """
        Z = self.Z.detach().clone().numpy()
        O = self.O.numpy()
        I_k = np.eye(self.k)
        return O @ Z @ np.linalg.inv(I_k + Z.T @ O @ Z) @ Z.T @ O

    def loss_inv_mu(self, l2=0.0):
        #loss_1 = torch.norm(self.Q - self.mu @ self.P @ self.mu.t())**2
        loss_1 = torch.norm(self.Q - self.mu @ self.mu.t())**2
        #loss_2 = torch.norm(
        #    torch.sum(self.mu @ self.P, 1) - torch.diag(self.O))**2
        loss_2 = torch.norm(
            torch.sum(self.mu, 1) - torch.diag(self.O))**2     
        return loss_1 + loss_2
    
    def loss_mu(self, l2=0.0):
        loss_1 = torch.norm(
        #    (self.O - self.mu @ self.P @ self.mu.t())[self.mask])**2
            (self.O - self.mu @ self.mu.t())[self.mask])**2
        loss_2 = torch.norm(
        #    torch.sum(self.mu @ self.P, 1) - torch.diag(self.O))**2
            torch.sum(self.mu, 1) - torch.diag(self.O))**2
        # loss_l2 = torch.norm( self.mu - self.mu_init )**2
        loss_l2 = 0
        return loss_1 + loss_2 + l2 * loss_l2
    
    def _set_constants(self, L):
        self.n, self.m = L.shape
        self.t = 1

    def train(self, L, deps=[], O=None, c_tree=None, **kwargs):
        """Train the model (i.e. estimate mu) in one of two ways, depending on
        whether source dependencies are provided or not:
        
        (1) No dependencies (conditionally independent sources): Estimate mu
        subject to constraints:
            (1a) O_{B(i,j)} - (mu P mu.T)_{B(i,j)} = 0, for i != j, where B(i,j)
                is the block of entries corresponding to sources i,j
            (1b) np.sum( mu P, 1 ) = diag(O)
        
        (2) Source dependencies:
            - First, estimate Z subject to the inverse form
            constraint:
                (2a) O_\Omega + (ZZ.T)_\Omega = 0, \Omega is the deps mask
            - Then, compute Q = mu P mu.T
            - Finally, estimate mu subject to mu P mu.T = Q and (1b)
        """
        self.config = recursive_merge_dicts(self.config, kwargs, 
            misses='ignore')

        # TODO: Change internals to use sparse throughout and delete this:
        if issparse(L):
            L = L.todense()

        self._set_constants(L)
        self.c_tree = CliqueTree(self.m, self.k, deps) if c_tree is None else \
            c_tree
        self.d = self.c_tree.d

        # Whether to take the simple conditionally independent approach, or the
        # "inverse form" approach for handling dependencies
        # This flag allows us to eg test the latter even with no deps present
        self.inv_form = (len(self.c_tree.edges) > 0)

        if self.inv_form:
            # Compute O, O^{-1}, and initialize params
            if O is not None:
                self.O = torch.from_numpy(O).float()
                self.O_inv = torch.from_numpy(np.linalg.inv(O)).float()
                self.d = self.O_inv.shape[0]
            else:
                self._generate_O_inv(L)
            self._init_params()

            # Estimate Z, compute Q = \mu P \mu^T
            if self.config['verbose']:
                print("Estimating Z...")
            self._train(self.loss_inv_Z)
            self.Q = torch.from_numpy(self.get_Q()).float()

            # Estimate \mu
            if self.config['verbose']:
                print("Estimating \mu...")
            self._train(self.loss_inv_mu)
        else:
            # Compute O and initialize params
            if self.config['verbose']:
                print("Computing O...")
            self._generate_O(L)
            self._init_params()

            # Estimate \mu
            if self.config['verbose']:
                print("Estimating \mu...")
            self._train(self.loss_mu)
    
    def _train(self, loss_fn):
        """Train model (self.parameters()) by optimizing the provided loss fn"""
        # TODO: Merge this _train with Classifier._train

        train_config = self.config['train_config']

        # Set optimizer as SGD w/ momentum
        optimizer_config = self.config['train_config']['optimizer_config']
        optimizer = optim.SGD(
            self.parameters(),
            **optimizer_config['optimizer_common'],
            **optimizer_config['sgd_config']
        )

        # Train model
        for epoch in range(train_config['n_epochs']):
            optimizer.zero_grad()
            
            # Compute gradient and take a step
            # Note that since this uses all n training points this is an epoch!
            loss = loss_fn(l2=train_config['l2'])
            if torch.isnan(loss):
                raise Exception("Loss is NaN. Consider reducing learning rate.")

            loss.backward()
            optimizer.step()
            
            # Print loss every print_every steps
            if (self.config['verbose'] and 
                (epoch % train_config['print_every'] == 0 
                or epoch == train_config['n_epochs'] - 1)):
                msg = f"[Epoch {epoch}] Loss: {loss.item():0.6f}"
                print(msg)