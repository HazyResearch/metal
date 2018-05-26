import numpy as np
from scipy.sparse import issparse
import torch

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
    def __init__(self, config, label_map=None, task_graph=None, dependencies=[]):
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
        self.dependencies = dependencies
    
    def train(self, L_train):
        """Learns the accuracies of the labeling functions from L_train

        Note that in this class, we learn this for each task separately by
        default, and store a separate accuracy for each LF in each task.
        """
        self._check_L(L_train, init=True)

        # Train model for each task separately as default for now
        # TODO: Extend this given task graph!
        for t, L_t in enumerate(L_train):
            self._train_task(L_t, t)
    
    def _train_task(self, L_t, t):

        # TODO: Have separate trainer class, and implement PyTorch methods like
        # forward and get_loss

        # TODO: Additionally check to make sure label matrix is unipolar

        # TODO: Form overlaps matrix

    def predict_proba(self, L, t=0):
        raise NotImplementedError