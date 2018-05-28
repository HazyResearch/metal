import numpy as np
from scipy.sparse import issparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metal.classifier import Classifier
from metal.label_model.lm_config import DEFAULT_CONFIG

class LabelModelBase(Classifier):
    """An abstract class for a label model

    TODO: Add docstring
    """
    
    def __init__(self, config=DEFAULT_CONFIG, label_map=None):
        """
        Args:
            config:
            label_map: 
        """
        super().__init__()
        self.config = config
        self.label_map = label_map
    
    def _check_L(self, L, init=False):
        """Check the format and content of the label tensor
        
        Args:
            L: A T-legnth list of N x M scipy.sparse matrices.
            init: If True, initialize self.T, self.M, and self.label_map if 
                empty; else check against these.
        """
        # Accept single sparse matrix and make it a singleton list
        if not isinstance(L, list):
            L = [L]

        # Check or set number of tasks and labeling functions
        self._check_or_set_attr('T', len(L), set_val=init)
        n, m = L[0].shape
        self._check_or_set_attr('M', m, set_val=init)
        
        # Check the format and dimensions of the task label matrices
        for t, L_t in enumerate(L):
            n_t, m_t = L_t.shape
            self._check_or_set_attr('M', m_t)
            if n_t != n:
                raise Exception(f"L[{t}] has {n_t} rows, but should have {n}.")
            if not issparse(L_t):
                raise Exception(f"L[{t}] has type {type(L_t)}, but should be a"
                    "scipy.sparse matrix.")
            if L_t.dtype != np.dtype(int):
                raise Exception(f"L[{t}] has type {L_t.dtype}, should be int.")
            
            # Ensure is in CSC sparse format for efficient col (LF) slicing
            L_t = L_t.tocsc()

        # If no label_map was provided, assume labels are continuous integers
        # starting from 1
        if self.label_map is None and init:
            if self.T > 1:
                raise Exception('Initialization parameter "label_map" cannot '
                    'be inferred when T > 1')
            K = np.amax(L[0])
            self.label_map = [list(range(K))]

        # Set cardinalities of each task
        self.K_t = [len(labels) for labels in self.label_map]

        # Check for consistency with cardinalities list
        for t, L_t in enumerate(L):
            if np.amax(L_t) > self.K_t[t]:
                raise Exception(f"Task {t} has cardinality {self.K_t[t]}, but"
                    "L[{t}] has max value = {np.amax(L_t)}.")
        
        return L
    
    def train(self, X, **kwargs):
        raise NotImplementedError

    def predict_tasks_proba(self, L):
        """Returns a list of T [N, K_t] tensors of soft (float) predictions."""
        return [self.predict_proba(L, t=t) for t in range(self.T)]

    def predict_tasks(self, L, break_ties='random'):
        """Returns a list of T [N, K_t] tensors of hard (int) predictions."""
        return [self.predict(L, t=t, break_ties=break_ties) for t in range(self.T)]

    def predict_proba(self, L, t=0):
        """Returns an [N, K_t] tensor of soft (float) predictions for task t."""
        raise NotImplementedError

    def predict(self, L, t=0, break_ties='random'):
        """Returns an N-dim tensor of hard (int) predictions for task t."""
        Y_ts = self.predict_proba(L, t=t).numpy()

        N, k = Y_ts.shape
        Y_th = np.zeros(N)
        diffs = np.abs(Y_ts - Y_ts.max(axis=1).reshape((-1, 1)))

        TOL = 1e-5
        for i in range(N):
            max_idxs = np.where(diffs[i, :] < TOL)[0]
            if len(max_idxs) == 1:
                Y_th[i] = max_idxs[0] + 1
            # Deal with 'tie votes' according to the specified policy
            elif break_ties == 'random':
                Y_th[i] = np.random.choice(max_idxs) + 1
            elif break_ties == 'abstain':
                Y_th[i] = 0
            else:
                ValueError(f'break_ties={break_ties} policy not recognized.')
        
        return torch.tensor(Y_th, dtype=torch.short)


class LabelModel(LabelModelBase):
    def __init__(self, config={}, label_map=None, task_graph=None, deps=[]):
        """
        Args:
            config: dict: A dictionary of config settings
            label_map: T-dim list of lists: The label map for each task 
                t=0,...,T-1
            task_graph: TaskGraph: A task graph...TBD
            dependencies: list: A list of dependencies of the form...TBD
        """
        super().__init__(config)
        self.label_map = label_map
        self.task_graph = task_graph
        self.deps = deps
    
    def _infer_polarity(self, L_t, j):
        """Infer the polarity (labeled class) of LF j on task t"""
        # Note: We assume that L_t is in CSC format here!
        vals = set(L_t.data[L_t.indptr[j]:L_t.indptr[j+1]])
        if len(vals) > 1:
            raise Exception(f"LF {j} on task {t} is non-unipolar: {vals}.")
        elif len(vals) == 0:
            # If an LF doesn't label this task, set its polarity to 0 = abstain
            return 0
        else:
            return list(vals)[0]
    
    def _get_overlaps_matrix(self, L):
        """Initializes the data structures for training task t"""
        # Check to make sure that L is unipolar, and infer polarities
        p = [self._infer_polarity(L, j) for j in range(self.M)]

        # Next, we form the empirical overlaps matrix O
        # In the unipolar categorical setting, this is just the empirical count
        # of non-zero overlaps; whether these were agreements or disagreements
        # is just a function of the polarities p
        N = L.shape[0]
        L_nz = L.copy()
        L_nz.data[:] = 1
        O = L_nz.T @ L_nz / N
        O = O.todense()

        # Divide out the empirical labeling propensities
        beta = np.diag(O)
        B = np.diag(1 / np.diag(O))
        O = B @ O @ B

        # Correct the O matrix given the known polarities
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    c = 1 if p[i] == p[j] else -1
                    O[i,j] = c * (O[i,j] - 1)

        # Turn O in PyTorch Variable
        return torch.clamp(torch.from_numpy(O), min=-0.95, max=0.95).float()
    
    def _init_params(self, gamma_init):
        """Initialize the parameters for each LF on each task separately"""
        # Note: Need to break symmetries by initializing > 0
        self.gamma = nn.Parameter(gamma_init * torch.ones(self.T, self.M))
    
    def _task_loss(self, O_t, t, l2=0.0):
        """Returns the *scaled* loss (i.e. ~ loss / m^2).

        Note: A *centered* L2 loss term is incorporated here.
        The L2 term is centered around the self.gamma_init property, which thus
        also serves as the value of a prior on the gammas.
        """
        loss = 0.0
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    loss += (self.gamma[t,i] * self.gamma[t,j] - O_t[i,j])**2

        # Normalize loss
        loss /= (self.M**2 - self.M)

        # L2 regularization, centered around gamma_init
        if l2 > 0.0:
            loss += l2 * torch.sum((self.gamma[t] - self.gamma_init)**2)
        return loss
    
    @property
    def accs(self):
        """The float *Tensor* (not Variable) of LF accuracies."""
        return torch.clamp(0.5 * (self.gamma.data + 1), min=0.01, max=0.99)
    
    @property
    def log_odds_accs(self):
        """The float *Tensor* (not Variable) of log-odds LF accuracies."""
        return torch.log(self.accs / (1 - self.accs)).float()

    def predict_proba(self, L, t=0):
        """Get conditional probabilities P(y_t | L) given the learned LF accs
        
        Note: This implementation is for conditionally independent labeling 
        functions (given y_t); handling deps is next...
        """
        # Check L and convert L_t to torch
        # Note we cast to dense here
        L = self._check_L(L)
        L_t = torch.from_numpy(L[t].todense()).float()
        N = L_t.shape[0]

        # Here we iterate over the values of Y in {1,...,K_t}, forming
        # a N x max(K_t) matrix of unnormalized predictions
        # Note in the unipolar setting:
        #   P(\lambda_j=k|y_t=k, \lambda_j != 0) = \alpha_i
        #   P(\lambda_j=k|y_t=l != k, \lambda_j != 0) = 1 - \alpha_i
        # So the computation is the same as in the binary case, except we
        # compute
        #   \theta^T \ind \{ \lambda_j != 0 \} \ind^{\pm} \{ \lambda_j = k \}
        K = max(self.K_t)
        Yp = torch.zeros((N, K))
        for y_t in range(1, self.K_t[t] + 1):
            L_t_y = torch.where(
                (L_t != y_t) & (L_t != 0), torch.full((N, self.M), -1) , L_t)
            L_t_y = torch.where(L_t_y == y_t, torch.full((N, self.M), 1), L_t_y)
            Yp[:,y_t-1] = L_t_y @ self.log_odds_accs[t]

        # Now we  take the softmax returning an N x max(K_t) torch Tensor
        return F.softmax(Yp, dim=1)
    
    def get_accs_score(self, accs):
        """Returns the *averaged squared estimation error."""
        return np.linalg.norm(self.accs.numpy() - accs)**2 / self.M
    
    def train(self, L_train, gamma_init=0.5, n_epochs=100, lr=0.1,
        momentum=0.9, l2=0.0, print_at=10, accs=None):
        """Learns the accuracies of the labeling functions from L_train

        Note that in this class, we learn this for each task separately by
        default, and store a separate accuracy for each LF in each task.
        """
        L_train = self._check_L(L_train, init=True)

        # Get overlaps matrices for each task
        O = [self._get_overlaps_matrix(L_t) for L_t in L_train]

        # Init params
        self._init_params(gamma_init)

        # Set optimizer as SGD w/ momentum
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        
        # Train model
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sum over the task losses uniformly
            loss = 0.0
            for t, O_t in enumerate(O):
                loss += self._task_loss(O_t, t, l2=l2)
            
            # Compute gradient and take a step
            # Note that since this uses all N training points this is an epoch!
            loss.backward()
            optimizer.step()
            
            # Print loss every k steps
            if epoch % print_at == 0 or epoch == n_epochs - 1:
                msg = f"[Epoch {epoch}] Loss: {loss.item():0.6f}"
                if accs is not None:
                    accs_score = self.get_accs_score(accs)
                    msg += f"\tAccs mean sq. error = {accs_score}"
                print(msg)

        print('Finished Training')