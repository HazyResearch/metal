import numpy as np
from scipy.sparse import issparse, csc_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metal.analysis import (
    plot_probabilities_histogram,
    plot_predictions_histogram,
    confusion_matrix,
)
from metal.classifier import Classifier
from metal.label_model.lm_defaults import lm_model_defaults
from metal.utils import recursive_merge_dicts


class LabelModel(Classifier):
    def __init__(self, deps=[], **kwargs):
        """
        Args:
            cardinality: int: Classifier output dimension
            seed: int: Random state seed
            deps: list: A list of LF dependencies as tuples of LF indices
        """
        self.config = recursive_merge_dicts(lm_model_defaults, kwargs)
        super().__init__(self.config['cardinality'], self.config['seed'])
        self.deps = deps

        # TODO: This is temporary, need to update to handle categorical!
        if self.k > 2:
            raise NotImplementedError("Cardinaltiy > 2 not implemented.")
        
        # TODO: Extend to handl LF deps (merge with other branch...)
        if len(self.deps) > 0:
            raise NotImplementedError("Dependency handling not implemented.")
    
    def _check_L(self, L, init=False):
        """Check the format and content of the label tensor
        
        Args:
            L: An [n, m] scipy.sparse matrix of labels
            init: If True, initialize self.m; else set based on input L
        """
        # Accept single CSC-sparse matrix (for efficient column/LF slicing)
        if not issparse(L):
            raise Exception(f"L has type {type(L)}, but should be a"
                "scipy.sparse matrix.")
        if L.dtype != np.dtype(int):
            raise Exception(f"L has type {L.dtype}, should be int.")
        L = L.tocsc()

        # Check or set number of labeling functions
        # M should be the same for all tasks, since all LFs label all inputs
        n, m = L.shape
        self._check_or_set_attr('m', m, set_val=init)
        return L    
    
    def _infer_polarity(self, L, j):
        """Infer the polarity (labeled class) of LF j"""
        assert(isinstance(L, csc_matrix))
        vals = set(L.data[L.indptr[j]:L.indptr[j+1]])
        if len(vals) > 1:
            raise Exception(f"LF {j} is non-unipolar with values: {vals}.")
        elif len(vals) == 0:
            # If an LF doesn't label this task, set its polarity to 0 = abstain
            return 0
        else:
            return list(vals)[0]
    
    def _compute_overlaps_matrix(self, L):
        """Initializes the data structures for training task t"""
        # Check to make sure that L_t is unipolar, and infer polarities
        self.polarity = [self._infer_polarity(L, j) for j in range(self.m)]

        # Next, we form the empirical overlaps matrix O
        # In the unipolar categorical setting, this is just the empirical count
        # of non-zero overlaps; whether these were agreements or disagreements
        # is then just a function of the polarities p
        n, m = L.shape
        L_nz = L.copy()
        L_nz.data[:] = 1
        O = L_nz.T @ L_nz / n 
        self.O = torch.from_numpy(O.todense()).double()
    
    def _init_params(self, mu_init, learn_class_balance, class_balance_init):
        """Initialize the parameters for each LF on each task separately.
        
        Args:
            mu_init: float: Initial value & prior value for parameter mu, where
                mu is the rank-2 factorization of O. Here, one row represents
                (e.g. for k=2): 
                    $[P(\lambda_i=p_i|Y=1), P(\lambda_i=p_i|Y=2)]$
                where $p_i \in {1,2}$ is the polarity of $\lambda_i$.
            learn_class_balance: bool: If true, learn the class balance
                parameters $P(Y=y_k)$
            class_balance_init: array(float) or None: If None, set to be the
                uniform distribution across the classes. If 
                `learn_class_balance=True`, then this is the initial value & 
                prior value for the class balance; else, this is fixed as the
                class balance.
        """
        self.mu_init = mu_init
        self.mu = nn.Parameter(torch.randn(self.m, 2).double())

        # Class balance- can be fixed or learnable
        if learn_class_balance:
            self.class_balance = class_balance_init
        else:
            # TODO: Currently this is for binary- extend to categorical
            self.class_balance = nn.Parameter(torch.randn(1,).double())
        self.P = torch.diag(torch.DoubleTensor(
            [self.class_balance, 1-self.class_balance]))

        # Initialize mask
        self.mask = torch.ones(self.m, self.m).byte()
        for i in range(self.m):
            for j in range(self.m):
                if i == j or (i,j) in self.deps or (j,i) in self.deps:
                    self.mask[i,j] = 0
    
    def loss(self, l2=0.0):
        """Returns the loss.

        Note: A *centered* L2 loss term is incorporated here.
        The L2 term is centered around the self.mu_init property, which thus
        also serves as the value of a prior on the mus.
        """
        loss_1 = torch.norm( 
            (self.O - self.mu @ self.P @ self.mu.t())[self.mask] )**2
        loss_2 = torch.norm(
            torch.sum(self.mu @ self.P, 1) - torch.diag(self.O) )**2

        # L2 regularization, centered around mu_init
        loss_3 = torch.sum((self.mu - self.mu_init)**2)
        return loss_1 + loss_2 + l2 * loss_3
    
    def config_set(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        recursive_merge_dicts(self.config, update_dict)
    
    def accs(self):
        """The float numpy array of LF accuracies."""
        # Swap elements in each row so that column 1 corresponds to the polarity
        # of the LF---i.e. represents P(\lf_i = p_i | Y = p_i)---and column 2
        # is then P(\lf_i = p_i | Y != p_i)
        accs = self.mu.detach().data.numpy().copy()
        for j in range(self.m):
            # TODO: Update this to support categorical!
            if self.polarity[j] != 1:
                row = accs[j]
                accs[j] = row[::-1]
        return accs[:,0]
    
    def log_odds_accs(self):
        """The float numpy array of log-odds LF accuracies."""
        return torch.log(self.accs() / (1 - self.accs())).float()
    
    def predict_proba(self, L):
        """Get conditional probabilities P(Y | L) given the learned model

        Args:
            L: An [n, m] scipy.sparse matrix of labels 
        Returns:
            Y_p: An n x k numpy matrix of probabilistic labels (conditional
                probabilities), i.e. $Y_p[i,j] = P(Y_i=j|L)$.
        """
        # TODO: Check this whole method!
        L = self._check_L(L)
        # Note we cast to dense here
        L = torch.from_numpy(L.todense()).float()
        n = L.shape[0]

        # Here we iterate over the values of Y in {1,...,k}, forming
        # an [n, k] matrix of unnormalized predictions
        # Note in the unipolar setting:
        #   P(\lambda_j=k|y_t=k, \lambda_j != 0) = \alpha_i
        #   P(\lambda_j=k|y_t=l != k, \lambda_j != 0) = 1 - \alpha_i
        # So the computation is the same as in the binary case, except we
        # compute
        #   \theta^T \ind \{ \lambda_j != 0 \} \ind^{\pm} \{ \lambda_j = k \}
        Y_p = torch.zeros((n, self.k))
        for y_k in range(1, self.k):
            L_y = torch.where(
                (L != y_k) & (L != 0), torch.full((n, self.m), -1) , L)
            L_y = torch.where(L_y == y_k, torch.full((n, self.m), 1), L_y)
            Y_p[:, y_k-1] = L_y @ self.log_odds_accs()

        # Take the softmax to return an [n, k] numpy array
        return F.softmax(Y_p, dim=1).numpy()

    def get_accs_score(self, accs):
        """Returns the *averaged squared estimation error."""
        return np.linalg.norm(self.accs() - accs)**2 / self.m

    def train(self, L_train, L_dev=None, Y_dev=None, accs=None, **kwargs):
        """Learns the accuracies of the labeling functions from L_train

        Args:
            L_train:
            L_dev:
            Y_dev:
            accs: An M-length list of the true accuracies of the LFs if known

        Note that in this class, we learn this for each task separately by
        default, and store a separate accuracy for each LF in each task.
        """
        self.config = recursive_merge_dicts(self.config, kwargs)
        train_config = self.config['train_config']
        optimizer_config = train_config['optimizer_config']

        # Initialize class parameters based on L_train
        L_train = self._check_L(L_train, init=True)

        # Compute overlaps matrix self.O
        self._compute_overlaps_matrix(L_train)

        # Initialize params
        self._init_params(
            train_config['mu_init'],
            train_config['learn_class_balance'],
            train_config['class_balance_init'])

        # Set optimizer as SGD w/ momentum
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
            loss = self.loss(l2=train_config['l2'])
            loss.backward()
            optimizer.step()
            
            # Print loss every print_at steps
            if (self.config['verbose'] and 
                (epoch % train_config['print_at'] == 0 
                or epoch == train_config['n_epochs'] - 1)):
                msg = f"[Epoch {epoch}] Loss: {loss.item():0.6f}"
                if accs is not None:
                    accs_score = self.get_accs_score(accs)
                    msg += f"\tAccs mean sq. error = {accs_score}"
                print(msg)

        if self.config['verbose']:
            print('Finished Training')
            Y_p_train = self.predict_proba(L_train)
            Y_ph_dev = self.predict(L_dev)

            if self.config['show_plots']:
                if self.T == 1:
                    plot_probabilities_histogram(Y_p_train[:, 0].numpy(), 
                        title="Training Set Predictions")

                    plot_predictions_histogram(Y_ph_dev.numpy(), Y_dev[0].numpy(),
                        title="Dev Set Hard Predictions:")
                else:
                    raise NotImplementedError

            print("Confusion Matrix (Dev)")
            mat = confusion_matrix(Y_ph_dev, Y_dev[0], pretty=True)