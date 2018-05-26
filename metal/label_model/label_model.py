import numpy as np
from scipy.sparse import issparse
import torch

from metal.classifier import Classifier
from metal.label_model.lm_config import DEFAULT_CONFIG

class LabelModelBase(Classifier):
    """
    An abstract class for a label model.

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

    def train(self, L_train):
        """
        Args:
            L_train: T-dim list of scipy.sparse: A list of scipy.sparse [N, M]
                matrices containing votes from M LFs on N examples for task t.  
        """              
        # TODO: Accept single sparse matrix and make it a singleton list
        if not isinstance(L_train, list):
            L_train = [L_train]
        
        for i, L_t in enumerate(L_train):
            if not issparse(L_t):
                msg = (f"Element {i} of list L_train has type {type(L_t)}, but "
                    "should be a scipy.sparse matrix.")
                raise Exception(msg)

        self.T = len(L_train)
        self.N, self.M = L_train[0].shape

        # If no label_map was provided, assume labels are continuous integers
        # starting from 1
        if self.label_map is None:
            if self.T > 1:
                raise Exception('Initialization parameter "label_map" cannot '
                    'be inferred when T > 1')
            K = np.amax(L_train[0])
            self.label_map = [list(range(K))]

        self.K_t = [len(labels) for labels in self.label_map]

    def predict_proba(self, L, t=0):
        """Returns an [N, K_t] tensor of soft (float) predictions for task t."""
        raise NotImplementedError


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
        super().train(L_train)
        raise NotImplementedError

    def predict_proba(self, L, t=0):
        raise NotImplementedError