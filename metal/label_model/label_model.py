from itertools import product

import numpy as np
from scipy.sparse import issparse, csc_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metal.analysis import (
    plot_probabilities_histogram,
    lf_summary,
    confusion_matrix,
)
from metal.classifier import Classifier
from metal.label_model.lm_defaults import lm_model_defaults
from metal.utils import recursive_merge_dicts
from metal.label_model.graph_utils import get_clique_tree


class LabelModel(Classifier):
    def __init__(self, m, k, p=None, deps=[], **kwargs):
        """
        Args:
            seed: int: Random state seed
            deps: list: A list of LF dependencies as tuples of LF indices
            kwargs: TBD
        """
        self.config = recursive_merge_dicts(lm_model_defaults, kwargs)
        super().__init__()

        self.k = k
        self.m = m
        self.d = self.m * self.k

        # Class balance- assume uniform if not provided
        if p is None:
            self.p = (1/self.k) * np.ones(self.k)
        else:
            self.p = p
        self.P = torch.diag(torch.from_numpy(self.p)).float()
        
        # Dependencies
        self.deps = deps
        self.c_tree = get_clique_tree(range(self.m), self.deps)

        # Whether to take the simple conditionally independent approach, or the
        # "inverse form" approach for handling dependencies
        # This flag allows us to eg test the latter even with no deps present
        self.inv_form = (len(self.deps) > 0)
    
    def config_set(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)
    
    def _get_augmented_label_matrix(self, L, offset=0):
        """Returns an augmented version of L where each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.
        
        Args:
            - L: A dense n x m numpy array, where n is the number of data points
                and m is the number of sources, with values in {0,1,...,k}
            - offset: Create indicators for values {offset,...,k}
        """
        n, m = L.shape
        km = self.k + 1 - offset
        L_aug = np.zeros((n, m * km))  # TODO: Extend for arbitrary cliques
        for y in range(offset, self.k+1):
            L_aug[:, y-offset::km] = np.where(L == y, 1, 0)
        return L_aug
    
    def _generate_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        self.n, self.m = L.shape
        L_aug = self._get_augmented_label_matrix(L, offset=1)
        self.O = torch.from_numpy( L_aug.T @ L_aug / self.n ).float()
    
    def _generate_O_inv(self, L):
        """Form the *inverse* overlaps matrix"""
        self._generate_O(L)
        self.O_inv = torch.from_numpy(np.linalg.inv(self.O.numpy())).float()
    
    def _init_params(self, init_precision=0.75, mu_init=0.25):
        """Initialize the learned params
        
        - \mu is the primary learned parameter, where each row corresponds to 
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:
        
            self.mu[i*self.k + j, y] = P(\lambda_i = j | Y = y)
        
        - Z is the inverse form version of \mu.
        - mask is the mask applied to O^{-1}, O for the matrix approx constraint

        Note that mu_init is used both to initialize and regularize mu (if L2 
        reg used). We initialize using:

            P(\lambda_i=y | Y=y) = P(\lambda_i) * P(Y=y | \lambda_i=y) / P(Y=y)
        
        where we use init_precision for P(Y=y | \lambda_i).
        """
        # Initialize mu on the right side of the core reflective symmetry
        # TODO: Better way to do this as (a) a constraint, or (b) a 
        # post-processing step?
        self.mu_init = torch.zeros(self.d, self.k)
        pl = torch.diag(self.O)
        for i in range(self.m):
            for y in range(self.k):
                idx = i*self.k + y
                # mu_init = min(0.25, pl[idx] * init_precision / self.p[y])
                self.mu_init[idx, y] += mu_init
        self.mu = nn.Parameter(self.mu_init.clone()).float()

        if self.inv_form:
            self.Z = nn.Parameter(torch.randn(self.d, self.k)).float()

        # Initialize the mask for the masked matrix approximation step; masks 
        # out the block diagonal of O and any dependencies
        self.mask = torch.ones(self.d, self.d).byte()
        for (i, j) in product(range(self.m), range(self.m)):
            if i == j or (i,j) in self.deps or (j,i) in self.deps:
                self.mask[i*self.k:(i+1)*self.k, j*self.k:(j+1)*self.k] = 0
    
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
            mu_i = mu[i*self.k:(i+1)*self.k, :]
            c_probs[i*(self.k+1) + 1:(i+1)*(self.k+1), :] = mu_i 
            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total prob
            c_probs[i*(self.k+1), :] = 1 - mu_i.sum(axis=0)
        
        # Clipping to be safe for now
        # TODO: Remove this
        c_probs = np.clip(c_probs, 0.01, 0.99)
    
        if source is not None:
            return c_probs[source*(self.k+1):(source+1)*(self.k+1)]
        else:
            return c_probs

    def get_label_probs(self, L):
        """Returns the n x k matrix of label probabilities P(Y | \lambda)"""
        if len(self.deps) > 0:
            raise NotImplementedError("Prediction for |G| > 0 not implemented.")
        
        mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)

        # Note: We omit abstains, effectively assuming uniform distribution here
        L_aug = self._get_augmented_label_matrix(L, offset=1)
        X = np.exp( L_aug @ np.log(mu) + np.log(self.p) )
        Z = np.tile(X.sum(axis=1).reshape(-1,1), self.k)
        return X / Z

    def loss_inv_Z(self, l2=0.0):
        return torch.norm((self.O_inv + self.Z @ self.Z.t())[self.mask])**2
    
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
        loss_l2 = torch.norm( self.mu - self.mu_init )**2
        return loss_1 + loss_2 + l2 * loss_l2
    
    def train(self, L, **kwargs):
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
            # Note that since this uses all N training points this is an epoch!
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