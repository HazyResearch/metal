import numpy as np
from scipy.sparse import issparse

from metal.label_model import LabelModel
from metal.label_model.lm_defaults import lm_default_config
from metal.multitask import MTClassifier
from metal.multitask.task_graph import TaskGraph
from metal.utils import recursive_merge_dicts


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
        # single-task model, but rarely the same for a multi-task model.
        self.k = self.task_graph.k

    def _set_constants(self, L):
        self.n, self.m = L[0].shape
        self.t = len(L)

    def _check_L(self, L):
        """Run some basic checks on L."""
        # TODO: Take this out?
        if issparse(L[0]):
            L = [L_t.todense() for L_t in L]

        # Check for correct values, e.g. warning if in {-1,0,1}
        for L_t in L:
            if np.any(L_t < 0):
                raise ValueError("L must have values in {0,1,...,k}.")

    def _create_L_ind(self, L):
        """Convert T label matrices with labels in 0...K_t to a one-hot format

        Here we can view e.g. the $(i,j)$ entries of the $T$ label matrices as
        a _label vector_ emitted by LF j for data point i.

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

        # Make sure converted to numpy here
        L = self._to_numpy(L)

        L_ind = np.ones((self.n, self.m * self.k))
        for yi, y in enumerate(self.task_graph.feasible_set()):
            for t in range(self.t):
                # A[x::y] slices A starting at x at intervals of y
                # e.g., np.arange(9)[0::3] == np.array([0,3,6])
                L_ind[:, yi :: self.k] *= np.where(
                    np.logical_or(L[t] == y[t], L[t] == 0), 1, 0
                )

            # Set LFs that abstained on all feasible label vectors to all 0s
            L_ind[:, yi :: self.k] *= np.where(sum(L) != 0, 1, 0)

        return L_ind

    def predict_proba(self, L):
        """Returns the task marginals estimated by the model: a t-length list of
        [n,k_t] matrices where the (i,j) entry of the sth matrix represents the
        estimated P((Y_i)_s | \lambda_j(x_i))

        Args:
            L: A t-length list of [n,m] scipy.sparse label matrices with values
                in {0,1,...,k}
        """
        # First, get the estimated probability distribution over the feasible
        # set defined by the TaskGraph
        # This is an [n,k] array, where k = |(feasible set)|
        Y_pf = LabelModel.predict_proba(self, L)
        n, k = Y_pf.shape

        # Now get the per-task marginals
        # TODO: Make this optional, versus just returning the above
        Y_p = [np.zeros((n, k_t)) for k_t in self.task_graph.K]
        for yi, y in enumerate(self.task_graph.feasible_set()):
            for t in range(self.t):
                k_t = int(y[t])
                Y_p[t][:, k_t - 1] += Y_pf[:, yi]
        return Y_p
