import numpy as np
from scipy.sparse import issparse
import torch

from metal.classifier import Classifier

class LabelModelBase(Classifier):
    """
    An abstract class for a label model.

    TODO: Add docstring
    """
    
    def __init__(self, L_train, label_map=None):
        """
        Args:
            L_train: T-dim list of scipy.sparse: A list of scipy.sparse [N, M]
                matrices containing votes from M LFs on N examples for task t.
            label_map: T-dim list of lists: The label map for each task 
                t=0,...,T-1
        """
        # TODO: Accept single sparse matrix and make it a singleton list
        if not isinstance(L_train, list):
            L_train = [L_train]
        
        for i, x in enumerate(L_train):
            if not issparse(x):
                raise Exception(f"Element {i} of list L_train has type {type(x)}, "
                    "but be a scipy.sparse matrix.")

        self.L_train = L_train
        self.T = len(L_train)
        self.M = L_train[0].shape[1]

        if label_map is None:
            K = np.amax(L_train[0][:,:])
            label_map = [list(range(K))]
        self.label_map = label_map

        self.K_t = [len(labels) for labels in label_map]

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

    def train(self):
        raise NotImplementedError


class LabelModel(LabelModelBase):
    def __init__(self, L, label_map, task_graph=[], dependencies=[]):
        """
        task_graph: TaskGraph: A task graph...TBD
        dependencies: list: A list of dependencies of the form...TBD
        """
        super().__init__(L, label_map)
    
    def train(self):
        raise NotImplementedError