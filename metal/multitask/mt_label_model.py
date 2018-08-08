import numpy as np

from metal.label_model import LabelModel
from metal.multitask import MTClassifier
from metal.utils import recursive_merge_dicts
from metal.label_model.lm_defaults import lm_default_config

class MTLabelModel(MTClassifier, LabelModel):
    def __init__(self, K=None, task_graph=None, class_balance=None, deps=[], 
        **kwargs):
        """
        Args:
            K: A t-length list of task cardinalities (overrided by task_graph
                if task_graph is not None)
            task_graph: TaskGraph: A TaskGraph which defines a feasible set of
                task label vectors; overrides K
            class_balance: (np.array) each class's percentage of the population
            deps: list: A list of source dependencies as tuples of indices 
            kwargs:
                - seed: int: Random state seed
        """
        self.config = recursive_merge_dicts(lm_default_config, kwargs)
        super().__init__(K)

        # WARNING: Currently only storing 1 k for all tasks in self.k and
        # ignores self.K!
        print("WARNING: Currently assuming all tasks have same cardinality!")
        self.task_graph = task_graph
        if self.task_graph is None:
            self.k = K
        else:
            self.k = len(self.task_graph)

        self._set_class_balance(class_balance)

    def _set_constants(self, L):
        self.n, self.m = L[0].shape
        self.t = len(L)

    def _create_L_aug(self, L, km, offset):
        L_aug = np.ones((self.n, self.m * km))

        # TODO: By default, this will operate with offset = 1 by skipping
        # abstains; should fix this!
        for yi, y in enumerate(self.task_graph.feasible_set()):
            for s in range(self.t):
                # Note that we cast to dense here, and are creating a dense
                # matrix; can change to fully sparse operations if needed
                L_s = L[s].todense()
                L_aug[:, yi::km] *= np.where(
                    np.logical_or(L_s == y[s], L_s == 0), 1, 0)
            
            # Handle abstains- if all elements of the task label are 0
            L_aug[:, yi::km] *= np.where(
                sum(map(abs, L)).todense() != 0, 1, 0)     

        return L_aug   