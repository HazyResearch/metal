import numpy as np
from scipy.sparse import issparse

from metal.label_model import LabelModel
from metal.multitask import MTClassifier, TaskGraph
from metal.utils import recursive_merge_dicts
from metal.label_model.lm_defaults import lm_default_config

class MTLabelModel(MTClassifier, LabelModel):
    def __init__(self, K=None, task_graph=None, **kwargs):
        """
        Args:
            K: A t-length list of task cardinalities (overrided by task_graph
                if task_graph is not None)
            task_graph: TaskGraph: A TaskGraph which defines a feasible set of
                task label vectors; overrides K if provided
        """
        config = recursive_merge_dicts(lm_default_config, kwargs)
        MTClassifier.__init__(self, K, config)

        if task_graph is None:
           task_graph = TaskGraph(K)
        self.task_graph = task_graph

        # Note: While K is a list of the cardinalities of the tasks, k is the 
        # cardinality of the feasible set. These are always the same for a 
        # single-task model, but rarely for a multi-task model.
        self.K = self.task_graph.K  
        self.k = self.task_graph.k

    def _set_constants(self, L):
        self.n, self.m = L[0].shape
        self.t = len(L)

    def _create_L_ind(self, L):
        """Convert T label matrices with labels in 0...K_t to a one-hot format

        An [n,m] matrix will become a [n,m*k] matrix. Note that no column is
        required for 0 (abstain) labels.

        Args:
            L: a T-length list of [n,m] scipy.sparse label matrices with values
                in {0,1,...,k}

        Returns:
            L_ind: An [n,m*k] dense np.ndarray with values in {0,1}
        
        Note that no column is required for 0 (abstain) labels.
        """
        # TODO: Update LabelModel to keep L, L_ind, L_aug as sparse matrices 
        # throughout and remove this line.
        if issparse(L[0]):
            L = [L_t.todense() for L_t in L]

        L_ind = np.ones((self.n, self.m * self.k))
        for yi, y in enumerate(self.task_graph.feasible_set()):
            for t in range(self.t):
                # A[x::y] slices A starting at x at intervals of y
                # e.g., np.arange(9)[0::3] == np.array([0,3,6])
                L_ind[:, yi::self.k] *= np.where(
                    np.logical_or(L[t] == y[t], L[t] == 0), 1, 0)

            # Set LFs that abstained on all feasible label vectors to all 0s
            L_ind[:, yi::self.k] *= np.where(sum(L) != 0, 1, 0)     

        return L_ind   