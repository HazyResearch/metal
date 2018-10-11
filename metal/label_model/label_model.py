from collections import Counter
from functools import partial
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
from torch.utils.data import DataLoader

from metal.classifier import Classifier
from metal.label_model.graph_utils import JunctionTree
from metal.label_model.lm_defaults import lm_default_config
from metal.utils import MetalDataset, recursive_merge_dicts


class LabelModel(Classifier):
    """A LabelModel...TBD

    Args:
        k: (int) the cardinality of the classifier
    """

    # This class variable is explained in the Classifier class
    implements_l2 = True

    def __init__(self, k=2, **kwargs):
        config = recursive_merge_dicts(lm_default_config, kwargs)
        super().__init__(k, config)

    # MODEL TRAINING
    def train(
        self,
        L_train=None,
        sigma_O=None,
        E_O=None,
        Y_dev=None,
        junction_tree=None,
        deps=[],
        class_balance=None,
        abstains=True,
        **kwargs,
    ):
        """Train the model (i.e. estimate mu) in one of two ways, depending on
        whether source dependencies are provided or not:

        Args:
            L_train: An [n,m] scipy.sparse matrix with values in {0,1,...,k}
                corresponding to labels from supervision sources on the
                training set. Either this or (sigma_O, E_O, junction_tree) must
                be provided.
            sigma_O: A [d,d] np.array representing Cov[\psi(O)], where O is the
                set of observable cliques of sources, \psi(O) is the vector of
                indicator random variables for O, and sigma_O = Cov[\psi(O)] is
                the generalized covariance for O. Either this
                (+ E_O, junction_tree) or L_train must be provided.
            E_O: A [d] np.array representing E[\psi(O)] (see above), i.e. the
                labeling rates for each source clique and label.
            Y_dev: Target labels for the dev set, for estimating class_balance
            junction_tree: A JunctionTree class representing the dependency
                structure of the LFs. If this is not provided, one is
                constructed based on any deps provided.
            deps: (list of tuples) known dependencies between supervision
                sources. If not provided, sources are assumed to be independent.
                TODO: add automatic dependency-learning code
            class_balance: (np.array) each class's percentage of the population
            abstains: (bool) Whether to include a 0 abstain value which the
                sources can output, but that is not in Y's range

        Note that to train the LabelModel, either (a) the [n, m] label matrix
        L_train or (b) (sigma_O, E_O, junction_tree) must be provided, where (b)
        is primarily for testing on non-noisey synthetic data.

        We learn the parameters mu (representing the marginal probabilities of
        the model over {Y, \lf_1, ..., \lf_}) by the following approach:
            - First, estimate Z subject to the inverse form
            constraint:
                (2a) O_\Omega + (ZZ.T)_\Omega = 0, \Omega is the deps mask
            - Then, compute mu using an eigendecomposition approach and sigma_H

        For further details, see:
        https://ajratner.github.io/assets/papers/mts-draft.pdf
        """
        self._init_train(
            L_train=L_train,
            sigma_O=sigma_O,
            E_O=E_O,
            Y_dev=Y_dev,
            junction_tree=junction_tree,
            deps=deps,
            class_balance=class_balance,
            abstains=abstains,
            **kwargs,
        )

        # Creating this faux dataset is necessary for now because the LabelModel
        # loss functions do not accept inputs, but Classifer._train() expects
        # training data to feed to the loss functions.
        dataset = MetalDataset([0], [0])
        train_loader = DataLoader(dataset)

        # Estimate Z, compute Q = \mu P \mu^T
        if self.config["verbose"]:
            print("Estimating Z...")
        self._train(train_loader, self.loss_inv_Z)

        # Compute and cache final mu from Z if inv_form=True, including breaking
        # sign symmetry here
        self.mu = self.get_mu()

    # MODEL PREDICTION
    def predict_proba(self, L):
        """Returns the [n,k] matrix of label probabilities P(Y | \lambda)

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        self._set_constants(L)

        # Only implemented for singleton separator sets right now
        if not self.jt.singleton_sep_sets:
            raise NotImplementedError("Predict for non-singleton sep sets.")

        L_aug = self.get_L_aug(L, offset=0)
        mask = self.jt.observed_maximal_mask(offset=0)

        # Get the extended version of mu that includes the non-minimal stats
        mu = self.get_extended_mu()

        n_s = len(self.jt.G.edges())
        X = np.exp(L_aug @ np.diag(mask) @ np.log(mu) - n_s * np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.k)
        return X / Z

    # MODEL TRAINING SUB-FUNCTIONS
    def _init_train(
        self,
        L_train=None,
        sigma_O=None,
        E_O=None,
        Y_dev=None,
        junction_tree=None,
        deps=[],
        class_balance=None,
        abstains=True,
        **kwargs,
    ):

        # Set config dictionaries
        self.config = recursive_merge_dicts(
            self.config, kwargs, misses="ignore"
        )

        # Set the class balance
        self._set_class_balance(class_balance, Y_dev)

        # Input is either (a) a label matrix, L_train, or else both (b.i) a
        # pre-computed sigma_O matrix and (b.ii) a JunctionTree object, where
        # (b) is primarily for synthetic testing
        if L_train is not None:
            self._check_L(L_train)
            self._set_constants(L_train)  # Sets self.m, self.t
            self.k0 = 0 if abstains else 1
        elif junction_tree is None or sigma_O is None or E_O is None:
            raise ValueError(
                "Must input L_train or (sigma_O, E_O, junction_tree)."
            )

        # We are either given an explicit JunctionTree object capturing the
        # dependency structure of the sources---*in which case the JunctionTree
        # params override any set already*--or a list of dependency edges
        self.jt = None
        if junction_tree is not None:
            self.jt = junction_tree
            self.m, self.t, self.k0 = self.jt.m, self.jt.t, self.jt.k0
        else:
            self.jt = JunctionTree(
                self.m, self.k, t=self.t, abstains=abstains, edges=deps
            )

        # Form basic data matrices
        if sigma_O is not None:
            self.E_O = E_O.reshape(-1, 1)
            self.sigma_O = torch.from_numpy(sigma_O).float()
            self.O = torch.from_numpy(sigma_O + self.E_O @ self.E_O.T).float()
        else:
            self.O = self.get_O(L_train)
            self.E_O = np.diag(self.O.numpy()).reshape(-1, 1)
            self.sigma_O = self.get_sigma_O(L_train)

        # Form Sigma_O^{-1}
        self.sigma_O_inv = torch.from_numpy(
            np.linalg.inv(self.sigma_O.numpy())
        ).float()

        # Initialize parameters
        self._init_params()

        # Build the mask over O^{-1}
        self._build_mask()

    def _set_class_balance(self, class_balance, Y_dev):
        """Set a prior for the class balance

        In order of preference:
        1) Use user-provided class_balance
        2) Estimate balance from Y_dev
        3) Assume uniform class distribution
        """
        if class_balance is not None:
            self.p = np.array(class_balance)
        elif Y_dev is not None:
            class_counts = Counter(Y_dev)
            sorted_counts = np.array(
                [v for k, v in sorted(class_counts.items())]
            )
            self.p = sorted_counts / sum(sorted_counts)
        else:
            self.p = (1 / self.k) * np.ones(self.k)
        self.P = torch.diag(torch.from_numpy(self.p)).float()

    def _check_L(self, L):
        """Run some basic checks on L."""
        # TODO: Take this out?
        if issparse(L):
            L = L.todense()

        # Check for correct values, e.g. warning if in {-1,0,1}
        if np.any(L < 0):
            raise ValueError("L must have values in {0,1,...,k}.")

    def _set_constants(self, L):
        self.m = L.shape[1]
        self.t = 1

    def get_sigma_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        O = self.get_O(L).numpy()
        l = np.diag(O).reshape(-1, 1)
        sigma_O = O - l @ l.T
        return torch.from_numpy(sigma_O).float()

    def get_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        n = L.shape[0]
        L_aug = self.get_L_aug(L)
        O = L_aug.T @ L_aug / n
        return torch.from_numpy(O).float()

    def get_L_aug(self, L, offset=1):
        """Returns the augmented version of L corresponding to the minimal set
        of indicators for each value of each clique of LFs.

        For example, three ind LFs with cardinality K=3, applied to a single
        data point, would go from:
            [1, 2, 3] --> [0, 0, 1, 0, 1, 1]
        """
        n = L.shape[0]
        d = self.jt.O_d if offset == 1 else self.jt.O_d_full
        L_aug = np.ones((n, d))

        # TODO: Update LabelModel to keep L variants as sparse matrices
        # throughout and remove this line.
        if issparse(L):
            L = L.todense()

        # Form matrix column-wise to be faster w.r.t. n
        for idx, vals in self.jt.iter_observed(offset=offset):
            for j, v in vals.items():
                L_aug[:, idx] *= L[:, j] == v
        return L_aug

    def _init_params(self):
        """Initialize the learned params Z"""
        self.Z = nn.Parameter(torch.randn(self.jt.O_d, self.k - 1)).float()

    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint; if
        an entry (i,j) corresponds to cliques C_i, C_j that belong to the same
        maximal clique, then mask out.
        """
        self.mask = torch.ones(self.jt.O_d, self.jt.O_d).byte()
        for ((i, vi), (j, vj)) in product(self.jt.iter_observed(), repeat=2):
            cids = set(vi.keys()).union(vj.keys())
            if len(self.jt.get_maximal_cliques(cids)) > 0:
                self.mask[i, j] = 0

    # These loss functions get all their data directly from the LabelModel
    # (for better or worse). The unused *args make these compatible with the
    # Classifer._train() method which expect loss functions to accept an input.

    def loss_inv_Z(self, *args):
        return (
            torch.norm((self.sigma_O_inv + self.Z @ self.Z.t())[self.mask]) ** 2
        )

    # MODEL PREDICTION SUB-FUNCTIONS
    def get_extended_mu(self):
        """Return mu, i.e. the matrix of marginal probabilities, for the full
        (non-minimal) set of values"""
        d = len(list(self.jt.iter_observed(offset=0)))
        mu = np.zeros((d, self.k))
        for y in range(1, self.k + 1):
            for idx, vals in self.jt.iter_observed(offset=0):
                mu[idx, y - 1] = self.P_marginal({**vals, self.m: y})
        return mu

    def _clip(self, p):
        return np.clip(p, 0.01, 0.99)

    def P_marginal(self, query):
        """Returns P(query), where query is a dictionary mapping a set of
        variable indices (where LFs are 0,...,self.m-1, Y is self.m) to values.

        Either looks up the value directly from the set of learned minimal
        statistics, mu, or infers from mu.
        """

        # Split the query into observed and Y
        query_O = dict([(k, v) for k, v in query.items() if k < self.m])
        Y = query[self.m]

        # If just the class balance, return directly
        if len(query_O) == 0:
            return self.p[Y - 1]

        # If the observed components in the minimal set of statistics, return
        elif query_O in self.jt.O_map:
            idx = self.jt.O_map.index(query_O)
            if Y > 1:
                return self.mu[idx, Y - 2]
            else:
                return self._clip(self.E_O[idx, 0] - self.mu[idx, :].sum())

        # Else, handle recursively
        else:
            # Remove one of the non-minimal values and recurse
            nm_idx = [
                i for i, v in query.items() if v == self.k0 and i < self.m
            ][0]
            query_r = dict([(k, v) for k, v in query.items() if k != nm_idx])
            p_o = sum(
                self.P_marginal({**query_r, nm_idx: v})
                for v in range(self.k0 + 1, self.k + 1)
            )
            return self._clip(self.P_marginal(query_r) - p_o)

    def _get_correct_sum(self, mu):
        """Sum all the elements of mu corresponding to P(\lf_i=Y); used as
        heuristic for breaking final sign symmetries"""
        sum_correct = 0.0
        for idx, vals in self.jt.iter_observed():
            if len(vals) == 1:
                val = list(vals.values())[0]
                if val > self.k0:
                    sum_correct += mu[idx, val - self.k0 - 1]
        return sum_correct

    def get_mu(self):
        """Recover mu from the low-rank matrix ZZ^T that we solve for."""
        # Test all ways of breaking col-wise sign symmetry, take one with
        # highest sum of correct accuracies P(\lf_i=y)
        mus = [
            self.get_sigma_OH(np.diag(c)) + self.E_O @ self.p[1:].reshape(1, -1)
            for c in product([-1, 1], repeat=(self.k - self.k0))
        ]
        idx = np.argmax([self._get_correct_sum(mu) for mu in mus])
        return self._clip(mus[idx])

    def get_sigma_OH(self, C):
        """Recover sigma_OH from the low-rank matrix ZZ^T that we solve for."""
        km = self.k - self.k0

        # Get Q = \Sigma_{OH} @ \Sigma_H^{-1} @ \Sigma_{OH}^T
        Z = self.Z.detach().clone().numpy()
        sigma_O = self.sigma_O.numpy()
        I_k = np.eye(km)
        Q = sigma_O @ Z @ np.linalg.inv(I_k + Z.T @ sigma_O @ Z) @ Z.T @ sigma_O

        # Take the eigendecomposition of Q and \Sigma_H^{-1}
        D1, V1 = np.linalg.eigh(Q)
        D2, V2 = np.linalg.eigh(np.linalg.inv(self.sigma_H))
        R = np.diag(1 / np.sqrt(D2[-km:] / D1[-km:]))

        # Recover \Sigma_{OH}, using C to break col-wise sign symmetry
        return V1[:, -km:] @ C @ R @ np.linalg.inv(V2)

    @property
    def sigma_H(self):
        if not self.jt.singleton_sep_sets:
            raise NotImplementedError("Sigma_H for non-singleton sep sets.")
        P = self.P[1:, 1:].numpy()
        p = np.diag(P).reshape(-1, 1)
        return P - p @ p.T


class LabelModelInd(LabelModel):
    """A LabelModel for the setting of independent LFs.

    Note: Something buggy with this right now... either fix or just get rid of.

    Args:
        k: (int) the cardinality of the classifier
    """

    # MODEL TRAINING
    def train(
        self,
        L_train=None,
        sigma_O=None,
        E_O=None,
        Y_dev=None,
        class_balance=None,
        abstains=True,
        **kwargs,
    ):
        """Train the model (i.e. estimate mu) in one of two ways, depending on
        whether source dependencies are provided or not:

        Args:
            L_train: An [n,m] scipy.sparse matrix with values in {0,1,...,k}
                corresponding to labels from supervision sources on the
                training set. Either this or (sigma_O, E_O, junction_tree) must
                be provided.
            sigma_O: A [d,d] np.array representing Cov[\psi(O)], where O is the
                set of observable cliques of sources, \psi(O) is the vector of
                indicator random variables for O, and sigma_O = Cov[\psi(O)] is
                the generalized covariance for O. Either this
                (+ E_O, junction_tree) or L_train must be provided.
            E_O: A [d] np.array representing E[\psi(O)] (see above), i.e. the
                labeling rates for each source clique and label.
            Y_dev: Target labels for the dev set, for estimating class_balance
            class_balance: (np.array) each class's percentage of the population
            abstains: (bool) Whether to include a 0 abstain value which the
                sources can output, but that is not in Y's range

        Note that to train the LabelModel, either (a) the [n, m] label matrix
        L_train or (b) (sigma_O, E_O, junction_tree) must be provided, where (b)
        is primarily for testing on non-noisey synthetic data.

        We learn the parameters mu (representing the marginal probabilities of
        the model over {Y, \lf_1, ..., \lf_}) assuming no dependencies
        (conditionally independent sources): we estimate mu subject to
        constraints:
            (1a) O_{B(i,j)} - (mu mu.T)_{B(i,j)} = 0, for i != j, where B(i,j)
                is the block of entries corresponding to sources i,j
            (1b) np.sum( mu, 1 ) = diag(O)
        """
        self._init_train(
            L_train=L_train,
            sigma_O=sigma_O,
            E_O=E_O,
            Y_dev=Y_dev,
            class_balance=class_balance,
            abstains=abstains,
            **kwargs,
        )
        train_config = self.config["train_config"]

        # Creating this faux dataset is necessary for now because the LabelModel
        # loss functions do not accept inputs, but Classifer._train() expects
        # training data to feed to the loss functions.
        dataset = MetalDataset([0], [0])
        train_loader = DataLoader(dataset)

        # Note that the LabelModel class implements its own (centered) L2 reg.
        l2 = train_config.get("l2", 0)

        # Estimate \mu
        # Note that self._mu is the learned Parameter here
        if self.config["verbose"]:
            print("Estimating \mu...")
        self._train(train_loader, partial(self.loss_mu, l2=l2))

        # Cache the numpy output learned param
        self.mu = self._mu.detach().numpy()

    def _init_params(self):
        """Initialize the learned params

        - \mu is the primary learned parameter, where each row corresponds to
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:

            self.mu[i*self.k + j, y] = P(\lambda_i = j | Y = y)

        and similarly for higher-order cliques.
        - Z is the inverse form version of \mu.
        """
        train_config = self.config["train_config"]

        # Initialize mu so as to break basic reflective symmetry
        # Note that we are given either a single or per-LF initial precision
        # value, prec_i = P(Y=y|\lf=y), and use:
        #   mu_init = P(\lf=y|Y=y) = P(\lf=y) * prec_i / P(Y=y)

        # Handle single or per-LF values
        if isinstance(train_config["prec_init"], (int, float)):
            prec_init = train_config["prec_init"] * torch.ones(self.m)
        else:
            prec_init = torch.from_numpy(train_config["prec_init"])
            if prec_init.shape[0] != self.m:
                raise ValueError(f"prec_init must have shape {self.m}.")

        # TODO: Update for higher-order cliques!
        # self.mu_init = torch.zeros(self.jt.O_d, self.k-1)
        # for i in range(self.m):
        #     for y in range(self.k-1):
        #         idx = i * self.k + y
        #         mu_init = torch.clamp(
        #           lps[idx] * prec_init[i] / self.p[y],0,1)
        #         self.mu_init[idx, y] += mu_init
        self._mu_init = torch.randn(self.jt.O_d, self.k)

        # Initialize randomly based on self.mu_init
        self._mu = nn.Parameter(
            self._mu_init.clone() * np.random.random() * 0.3
        ).float()

    # These loss functions get all their data directly from the LabelModel
    # (for better or worse). The unused *args make these compatible with the
    # Classifer._train() method which expect loss functions to accept an input.

    def loss_mu(self, *args, l2=0):
        loss_1 = torch.norm((self.O - self._mu @ self._mu.t())[self.mask]) ** 2
        loss_2 = torch.norm(torch.sum(self._mu, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self.loss_l2(l2=l2)

    def loss_l2(self, l2=0):
        """L2 loss centered around mu_init, scaled optionally per-source.

        In other words, diagonal Tikhonov regularization,
            ||D(\mu-\mu_{init})||_2^2
        where D is diagonal.

        Args:
            - l2: A float or np.array representing the per-source regularization
                strengths to use
        """
        if isinstance(l2, (int, float)):
            D = l2 * torch.eye(self.jt.O_d)
        else:
            D = torch.diag(torch.from_numpy(l2))

        # Note that mu is a matrix and this is the *Frobenius norm*
        return torch.norm(D @ (self._mu - self._mu_init)) ** 2
