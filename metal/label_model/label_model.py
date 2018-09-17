from collections import Counter
from functools import partial
from itertools import chain, product

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

    def _check_L(self, L):
        """Run some basic checks on L."""
        # TODO: Take this out?
        if issparse(L):
            L = L.todense()

        # Check for correct values, e.g. warning if in {-1,0,1}
        if np.any(L < 0):
            raise ValueError("L must have values in {0,1,...,k}.")

    def _create_L_ind(self, L):
        """Convert a label matrix with labels in 0...k to a one-hot format

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}

        Returns:
            L_ind: An [n,m*k] dense np.ndarray with values in {0,1}

        Note that no column is required for 0 (abstain) labels.
        """
        # TODO: Update LabelModel to keep L variants as sparse matrices
        # throughout and remove this line.
        if issparse(L):
            L = L.todense()

        L_ind = np.zeros((self.n, self.m * self.k))
        for y in range(1, self.k + 1):
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_ind[:, (y - 1) :: self.k] = np.where(L == y, 1, 0)
        return L_ind

    def _get_augmented_label_matrix(self, L, higher_order=False):
        """Returns an augmented version of L where each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure
        self.c_data = {}
        for i in range(self.m):
            self.c_data[i] = {
                "start_index": i * self.k,
                "end_index": (i + 1) * self.k,
                "max_cliques": set(
                    [
                        j
                        for j in self.c_tree.nodes()
                        if i in self.c_tree.node[j]["members"]
                    ]
                ),
            }

        L_ind = self._create_L_ind(L)

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if higher_order:
            L_aug = np.copy(L_ind)
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.node[item]
                    C_type = "node"
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                    C_type = "edge"
                else:
                    raise ValueError(item)
                members = list(C["members"])
                nc = len(members)

                # If a unary maximal clique, just store its existing index
                if nc == 1:
                    C["start_index"] = members[0] * self.k
                    C["end_index"] = (members[0] + 1) * self.k

                # Else add one column for each possible value
                else:
                    L_C = np.ones((self.n, self.k ** nc))
                    for i, vals in enumerate(product(range(self.k), repeat=nc)):
                        for j, v in enumerate(vals):
                            L_C[:, i] *= L_ind[:, members[j] * self.k + v]

                    # Add to L_aug and store the indices
                    if L_aug is not None:
                        C["start_index"] = L_aug.shape[1]
                        C["end_index"] = L_aug.shape[1] + L_C.shape[1]
                        L_aug = np.hstack([L_aug, L_C])
                    else:
                        C["start_index"] = 0
                        C["end_index"] = L_C.shape[1]
                        L_aug = L_C

                    # Add to self.c_data as well
                    id = tuple(members) if len(members) > 1 else members[0]
                    self.c_data[id] = {
                        "start_index": C["start_index"],
                        "end_index": C["end_index"],
                        "max_cliques": set([item])
                        if C_type == "node"
                        else set(item),
                    }
            return L_aug
        else:
            return L_ind

    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint"""
        self.mask = torch.ones(self.d, self.d).byte()

        # TODO: Where do we pass this from...?
        higher_order_cliques = True
        io = self.jt.iter_observed(higher_order_cliques=higher_order_cliques)
        for ((i, vals_i), (j, vals_j)) in product(io, repeat=2):

            # Check if ci and cj are part of the same maximal clique
            # If so, mask out their corresponding blocks in O^{-1}
            cids = set(vals_i.keys()).union(vals_j.keys())
            if len(self.jt._get_maximal_cliques(cids)) > 0:
                self.mask[i, j] = 0

    def _generate_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources

        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        L_aug = self._get_augmented_label_matrix(L)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy(L_aug.T @ L_aug / self.n).float()

    def _generate_O_inv(self, L):
        """Form the *inverse* overlaps matrix"""
        self._generate_O(L)
        self.O_inv = torch.from_numpy(np.linalg.inv(self.O.numpy())).float()

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

        # Get the per-value labeling propensities
        # Note that self.O must have been computed already!
        # lps = torch.diag(self.O).numpy()

        # Note: Params should be self.k - 1 now!

        # TODO: Update for higher-order cliques!
        # self.mu_init = torch.zeros(self.d, self.k-1)
        # for i in range(self.m):
        #     for y in range(self.k-1):
        #         idx = i * self.k + y
        #         mu_init = torch.clamp(lps[idx] * prec_init[i] / self.p[y],0,1)
        #         self.mu_init[idx, y] += mu_init
        self.mu_init = torch.randn(self.d, self.k - 1)

        # Initialize randomly based on self.mu_init
        self.mu = nn.Parameter(
            self.mu_init.clone() * np.random.random()
        ).float()

        if self.inv_form:
            self.Z = nn.Parameter(torch.randn(self.d, self.k - 1)).float()

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
        c_probs = np.zeros((self.m * (self.k + 1), self.k))
        mu = self.mu.detach().clone().numpy()

        for i in range(self.m):
            # si = self.c_data[(i,)]['start_index']
            # ei = self.c_data[(i,)]['end_index']
            # mu_i = mu[si:ei, :]
            mu_i = mu[i * self.k : (i + 1) * self.k, :]
            c_probs[i * (self.k + 1) + 1 : (i + 1) * (self.k + 1), :] = mu_i

            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total prob
            c_probs[i * (self.k + 1), :] = 1 - mu_i.sum(axis=0)
        c_probs = np.clip(c_probs, 0.01, 0.99)

        if source is not None:
            return c_probs[source * (self.k + 1) : (source + 1) * (self.k + 1)]
        else:
            return c_probs

    def predict_proba(self, L):
        """Returns the [n,k] matrix of label probabilities P(Y | \lambda)

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        self._set_constants(L)

        L_aug = self._get_augmented_label_matrix(L)
        mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)

        # Create a "junction tree mask" over the columns of L_aug / mu
        if len(self.deps) > 0:
            jtm = np.zeros(L_aug.shape[1])

            # All maximal cliques are +1
            for i in self.c_tree.nodes():
                node = self.c_tree.node[i]
                jtm[node["start_index"] : node["end_index"]] = 1

            # All separator sets are -1
            for i, j in self.c_tree.edges():
                edge = self.c_tree[i][j]
                jtm[edge["start_index"] : edge["end_index"]] = 1
        else:
            jtm = np.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.k)
        return X / Z

    def get_Q(self):
        """Get the model's estimate of
            Q = \Sigma_{OH} \Sigma_H^{-1} \Sigma_{OH}^T

        We can then separately extract \mu subject to additional constraints,
        e.g. \mu P 1 = diag(O).
        """
        Z = self.Z.detach().clone().numpy()
        # Note: shorthand for \Sigma_O
        O = self.O.numpy()
        # Note: Params should be self.k - 1 now!
        I_k = np.eye(self.k - 1)
        return O @ Z @ np.linalg.inv(I_k + Z.T @ O @ Z) @ Z.T @ O

    # These loss functions get all their data directly from the LabelModel
    # (for better or worse). The unused *args make these compatible with the
    # Classifer._train() method which expect loss functions to accept an input.

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
            D = l2 * torch.eye(self.d)
        else:
            D = torch.diag(torch.from_numpy(l2))

        # Note that mu is a matrix and this is the *Frobenius norm*
        return torch.norm(D @ (self.mu - self.mu_init)) ** 2

    def loss_inv_Z(self, *args):
        return torch.norm((self.O_inv + self.Z @ self.Z.t())[self.mask]) ** 2

    def loss_inv_mu(self, *args, l2=0):
        # loss_1 = torch.norm(self.Q - self.mu @ P @ self.mu.t()) ** 2
        # loss_2 = (
        #     torch.norm(torch.sum(self.mu @ P, 1) - torch.diag(self.O)) ** 2
        # )
        # return loss_1 + loss_2 + self.loss_l2(l2=l2)

        # Note: Params should be self.k - 1 now!
        P_inv = torch.from_numpy(np.linalg.inv(self.P[1:, 1:])).float()
        return torch.norm(self.Q - self.mu @ P_inv @ self.mu.t()) ** 2

    def loss_mu(self, *args, l2=0):
        loss_1 = (
            torch.norm((self.O - self.mu @ self.P @ self.mu.t())[self.mask])
            ** 2
        )
        loss_2 = (
            torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        )
        return loss_1 + loss_2 + self.loss_l2(l2=l2)

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

    def _set_constants(self, L):
        self.n, self.m = L.shape
        self.t = 1

    def train(
        self,
        L_train=None,
        sigma_O=None,
        Y_dev=None,
        junction_tree=None,
        deps=[],
        class_balance=None,
        **kwargs,
    ):
        """Train the model (i.e. estimate mu) in one of two ways, depending on
        whether source dependencies are provided or not:

        Args:
            L_train: An [n,m] scipy.sparse matrix with values in {0,1,...,k}
                corresponding to labels from supervision sources on the
                training set. Either this or sigma_O must be provided
            sigma_O: A [d,d] np.array representing the generalized covariance
                matrix for the observable variables (cliques). Either this or
                L_train must be provided
            Y_dev: Target labels for the dev set, for estimating class_balance
            junction_tree: A JunctionTree class representing the dependency
                structure of the LFs. If this is not provided, one is
                constructed based on any deps provided.
            deps: (list of tuples) known dependencies between supervision
                sources. If not provided, sources are assumed to be independent.
                TODO: add automatic dependency-learning code
            class_balance: (np.array) each class's percentage of the population

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

        # Set config dictionaries
        self.config = recursive_merge_dicts(
            self.config, kwargs, misses="ignore"
        )
        train_config = self.config["train_config"]

        # Set the class balance
        self._set_class_balance(class_balance, Y_dev)

        # TODO: Management of the different init options needs to be improved...
        # E.g. checks for consistency, readable error messages, etc.
        if L_train is not None:
            self._check_L(L_train)
            self._set_constants(L_train)  # Sets self.m, self.n, self.t
        elif junction_tree is not None and sigma_O is not None:
            self.m = junction_tree.m
        else:
            raise ValueError("Must input L_train or sigma_O and junction_tree.")

        # Set or create the JunctionTree that handles the dependency structure
        # of the LFs
        self.jt = None
        if junction_tree is not None:
            self.jt = junction_tree
        elif len(deps) > 0:
            self.jt = JunctionTree(
                self.m, self.k, edges=deps, higher_order_cliques=False
            )

        # Whether to take the simple conditionally independent approach, or the
        # "inverse form" approach for handling dependencies
        # This flag allows us to eg test the latter even with no deps present
        self.inv_form = len(deps) > 0 or self.jt is not None

        # Creating this faux dataset is necessary for now because the LabelModel
        # loss functions do not accept inputs, but Classifer._train() expects
        # training data to feed to the loss functions.
        dataset = MetalDataset([0], [0])
        train_loader = DataLoader(dataset)

        # Note that the LabelModel class implements its own (centered) L2 reg.
        l2 = train_config.get("l2", 0)

        if self.inv_form:
            # Compute O, O^{-1}, and initialize params
            if L_train is not None:
                if self.config["verbose"]:
                    print("Computing O^{-1}...")
                self._generate_O_inv(L_train)
            else:
                self.O = torch.from_numpy(sigma_O).float()
                self.O_inv = torch.from_numpy(np.linalg.inv(sigma_O)).float()
                self.d = self.O.shape[0]

            # Initialize parameters and mask
            self._init_params()

            # Estimate Z, compute Q = \mu P \mu^T
            if self.config["verbose"]:
                print("Estimating Z...")
            self._train(train_loader, self.loss_inv_Z)
            self.Q = torch.from_numpy(self.get_Q()).float()

            # Estimate \mu
            if self.config["verbose"]:
                print("Estimating \mu...")
            self._train(train_loader, partial(self.loss_inv_mu, l2=l2))
        else:
            # Compute O and initialize params
            if L_train is not None:
                if self.config["verbose"]:
                    print("Computing O...")
                self._generate_O(L_train)
            else:
                self.O = torch.from_numpy(sigma_O).float()
                self.d = self.O.shape[0]

            # Initialize parameters and mask
            self._init_params()

            # Estimate \mu
            if self.config["verbose"]:
                print("Estimating \mu...")
            self._train(train_loader, partial(self.loss_mu, l2=l2))
