from itertools import product, chain
from collections import OrderedDict
import random

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
    
    def _create_L_ind(self, L, km, offset):
        L_ind = np.zeros((self.n, self.m * km))
        for y in range(offset, self.k+1):
            L_ind[:, y-offset::km] = np.where(L == y, 1, 0)     
        return L_ind   

    def _get_augmented_label_matrix(self, L, offset=1):
        """Returns an augmented version of L where each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.
        
        Args:
            - L: A dense n x m numpy array, where n is the number of data points
                and m is the number of sources, with values in {0,1,...,k}
            - offset: Create indicators for values {offset,...,k}
        """
        km = self.k + 1 - offset

        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure
        self.c_data = {}
        
        # Create the basic indicator version of the source label matrix
        L_ind = self._create_L_ind(L, km, offset)

        if self.config['all_unary_cliques']:
            # Add all unary cliques
            for i in range(self.m):
                self.c_data[i] = {
                    'start_index': i*km,
                    'end_index': (i+1)*km,
                    'max_cliques': set([j for j in self.c_tree.nodes() 
                        if i in self.c_tree.node[j]['members']])
                }
            L_aug = np.copy(L_ind)
        else:
            L_aug = None
        
        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if self.config['higher_order_cliques']:
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.node[item]
                    C_type = 'node'
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                    C_type = 'edge'
                else:
                    raise ValueError(item)
                members = list(C['members'])
                nc = len(members)
                id = tuple(members) if len(members) > 1 else members[0]

                # Check if already added
                if id in self.c_data:
                    continue

                # If a unary maximal clique, just store its existing index
                if nc == 1 and self.config['all_unary_cliques']:
                    C['start_index'] = members[0] * km
                    C['end_index'] = (members[0]+1) * km

                # Else add one column for each possible value
                else:
                    L_C = np.ones((self.n, km ** nc))
                    for i, vals in enumerate(product(range(km), repeat=nc)):
                        for j, v in enumerate(vals):
                            L_C[:,i] *= L_ind[:, members[j]*km + v]

                    # Add to L_aug and store the indices
                    if L_aug is not None:
                        C['start_index'] = L_aug.shape[1]
                        C['end_index'] = L_aug.shape[1] + L_C.shape[1]
                        L_aug = np.hstack([L_aug, L_C])
                    else:
                        C['start_index'] = 0
                        C['end_index'] = L_C.shape[1]
                        L_aug = L_C
                    
                    # Add to self.c_data as well
                    self.c_data[id] = {
                        'start_index': C['start_index'],
                        'end_index': C['end_index'],
                        'max_cliques': set([item]) if C_type == 'node' 
                            else set(item)
                    }
            return L_aug
        else:
            return L_ind
    
    def _generate_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        L_aug = self._get_augmented_label_matrix(L, offset=1)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy( L_aug.T @ L_aug / self.n ).float()
    
    def _generate_O_inv(self, L, eps=1e-2, prec=1000, cond_thresh=500):
        """Form the *inverse* overlaps matrix"""
        self._generate_O(L)

        # Print warning if O is poorly conditioned
        kappa_O = np.linalg.cond(self.O.numpy())
        if kappa_O > cond_thresh:
            print(f"Warning: O is ill-conditioned: kappa(O) = {kappa_O:0.2f}.")
        
        # self.O_inv = torch.from_numpy(np.linalg.inv(self.O.numpy())).float()

        # Trying the pseudoinverse, dropping singular values that are too small
        # O = self.O.numpy()
        # U, s, V = np.linalg.svd(O)
        # S = np.diag(1/s)
        # S[1/S < eps] = 0
        # self.O_inv = torch.from_numpy(V.T @ S @ U.T).float()
        
        # Use high-precision matrix operations starting with L.T @ L...
        L_aug = self._get_augmented_label_matrix(L, offset=1)
        O_unnorm = mpmath.matrix(L_aug.T @ L_aug)
        n = mpmath.mpf(self.n)
        O_inv = (O_unnorm / n) ** -1
        with mpmath.workdps(prec):
            self.O_inv = torch.from_numpy(
                np.array(O_inv.tolist(), dtype=float)).float()
    
    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint"""
        self.mask = torch.ones(self.d, self.d).byte()
        for members_i, ci in self.c_data.items():
            si, ei = ci['start_index'], ci['end_index']
            for members_j, cj in self.c_data.items():
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
        for members_i, ci in self.c_data.items():
            si, ei = ci['start_index'], ci['end_index']

            # Unary cliques
            if isinstance(members_i, int) or len(members_i) == 1:
                self.mu_init[si:ei, :] = torch.eye(self.k) * np.random.random()

            # Higher-order cliques
            # TODO
        
        self.mu = nn.Parameter(self.mu_init.clone()).float()

        if self.inv_form:
            self.Z = nn.Parameter(torch.randn(self.d, self.k)).float()

        # Build the mask over O^{-1}
        # TODO: Put this elsewhere?
        self._build_mask()
    
    def get_conditional_probs(self, source=None):
        """Returns the full conditional probabilities table as a numpy array,
        where row i*(k+1) + ly is the conditional probabilities of source i 
        emmiting label ly (including abstains 0), conditioned on different 
        values of Y, i.e.:
        
            c_probs[i*(k+1) + ly, y] = P(\lambda_i = ly | Y = y)
        
        Note that this simply involves inferring the kth row by law of total
        probability and adding in to mu.
        
        If `source` is not None, returns only the corresponding block.
        """
        c_probs = np.zeros((self.m * (self.k+1), self.k))
        mu = self.mu.detach().clone().numpy()
        
        for i in range(self.m):
            # si = self.c_data[(i,)]['start_index']
            # ei = self.c_data[(i,)]['end_index']
            # mu_i = mu[si:ei, :]
            mu_i = mu[i*self.k:(i+1)*self.k, :]
            c_probs[i*(self.k+1) + 1:(i+1)*(self.k+1), :] = mu_i 
            
            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total prob
            c_probs[i*(self.k+1), :] = 1 - mu_i.sum(axis=0)
        c_probs = np.clip(c_probs, 0.01, 0.99)
    
        if source is not None:
            return c_probs[source*(self.k+1):(source+1)*(self.k+1)]
        else:
            return c_probs

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
            for i in self.c_tree.nodes():
                node = self.c_tree.node[i]
                jtm[node['start_index']:node['end_index']] = 1

            # All separator sets are -1
            for i, j in self.c_tree.edges():
                edge = self.c_tree[i][j]
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
        loss_1 = torch.norm(self.Q - self.mu @ self.P @ self.mu.t())**2
        loss_2 = torch.norm(
            torch.sum(self.mu @ self.P, 1) - torch.diag(self.O))**2
        return loss_1 + loss_2
    
    def loss_mu(self, l2=0.0):
        loss_1 = torch.norm(
            (self.O - self.mu @ self.P @ self.mu.t())[self.mask])**2
        loss_2 = torch.norm(
            torch.sum(self.mu @ self.P, 1) - torch.diag(self.O))**2
        # loss_l2 = torch.norm( self.mu - self.mu_init )**2
        loss_l2 = 0
        return loss_1 + loss_2 + l2 * loss_l2
    
    def _set_constants(self, L):
        self.n, self.m = L.shape
        self.t = 1

    def _set_dependencies(self, deps):
        nodes = range(self.m)
        self.deps = deps
        self.c_tree = get_clique_tree(nodes, deps)

    def train(self, L, deps=[], **kwargs):
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
        self._set_dependencies(deps)

        # Whether to take the simple conditionally independent approach, or the
        # "inverse form" approach for handling dependencies
        # This flag allows us to eg test the latter even with no deps present
        self.inv_form = (len(self.deps) > 0)

        if self.inv_form:
            # Compute O, O^{-1}, and initialize params
            if self.config['verbose']:
                print("Computing O^{-1}...")
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