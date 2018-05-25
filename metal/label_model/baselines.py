from collections import Counter

import numpy as np
import torch

from metal.label_model.model import LabelModelBase
from metal.utils import rargmax

class RandomVoter(LabelModelBase):
    """
    A class that votes randomly among the available labels for each task
    """
    def train(self):
        "RandomVoter does not need to train. Skipping..."

    def predict_proba(self, L, t):
        N = L.shape[0]
        Y_ts = np.random.rand(N, self.K_t[t])
        Y_ts /= Y_ts.sum(axis=1).reshape(-1, 1)
        return torch.tensor(Y_ts, dtype=torch.float)


class MajorityLabelVoter(LabelModelBase):
    """
    A class that treats every task independently, placing all probability on 
    the majority label from all non-abstaining LFs for that task.

    Note that in the case of ties, non-integer probabilities are possible.
    """

    def train(self):
        "MajorityVoter does not need to train. Skipping..."

    def predict_proba(self, L, t):
        L = np.array(L.todense()).astype(int)
        N, M = L.shape
        K_t = self.K_t[t]
        Y_ts = np.zeros((N, K_t))
        for i in range(N):
            counts = np.zeros(K_t)
            for j in range(M):
                if L[i,j]:
                    counts[L[i,j] - 1] += 1
            Y_ts[i, :] = np.where(counts == max(counts), 1, 0)
        Y_ts /= Y_ts.sum(axis=1).reshape(-1, 1)
        return torch.tensor(Y_ts, dtype=torch.float)


class MajorityClassVoter(LabelModelBase):
    """
    A class that treats every task independently, placing all probability on
    the majority class based on class balance (and ignoring the label matrix).

    Note that in the case of ties, non-integer probabilities are possible.
    """
    def __init__(self, L_train, balances, **kwargs):
        """
        Args:
            balances: A list of T lists or np.arrays that each sum to 1, 
                corresponding to the (possibly estimated) class balances for 
                each task.
        """
        super().__init__(L_train, **kwargs)
        assert(isinstance(balances, list))
        assert(len(balances) == self.T)
        assert(all(isinstance(x, np.ndarray) for x in balances))
        self.balances = balances


    def train(self):
        "MajorityVoter does not need to train. Skipping..."

    def predict_proba(self, L, t):
        N = L.shape[0]
        Y_ts = np.zeros((N, self.K_t[t]))
        max_classes = np.where(self.balances[t] == max(self.balances[t]))
        for c in max_classes:
            Y_ts[:, c] = 1.0
        Y_ts /= Y_ts.sum(axis=1).reshape(-1, 1)
        return torch.tensor(Y_ts, dtype=torch.float)