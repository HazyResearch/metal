from collections import Counter

import numpy as np
from scipy.sparse import csc_matrix
import torch

from metal.utils import rargmax, multitask
from metal.label_model.label_model import LabelModelBase

class RandomVoter(LabelModelBase):
    """
    A class that votes randomly among the available labels for each task
    """
    def train(self, L, **kwargs):
        # Note that this also sets some class parameters which we need later
        _ = self._check_L(L, init=True)

    @multitask
    def predict_proba(self, L):
        """
        Args:
            L: An [N, M] scipy.sparse matrix of labels or a T-length list of 
                such matrices if self.multitask=True
        Returns:
            output: A T-length list of [N, K_t] soft predictions
        """
        L = self._check_L(L)
        Y_p = [self.predict_task_proba(L, t) for t in range(self.T)]
        return Y_p

    def predict_task_proba(self, L, t):
        L = self._check_L(L)
        N = L[t].shape[0]
        Y_t = np.random.rand(N, self.K_t[t])
        Y_t /= Y_t.sum(axis=1).reshape(-1, 1)
        return torch.tensor(Y_t, dtype=torch.float)


class MajorityClassVoter(RandomVoter):
    """
    A class that treats every task independently, placing all probability on
    the majority class based on class balance (and ignoring the label matrix).

    Note that in the case of ties, non-integer probabilities are possible.
    """
    def train(self, L, balances, **kwargs):
        """
        Args:
            balances: A list of T lists or np.arrays that each sum to 1, 
                corresponding to the (possibly estimated) class balances for 
                each task.
        """
        _ = self._check_L(L, init=True)
        assert(isinstance(balances, list))
        assert(len(balances) == self.T)
        assert(all(isinstance(x, np.ndarray) for x in balances))
        self.balances = balances
        
    def predict_task_proba(self, L, t):
        self._check_L(L)
        N = L[t].shape[0]
        Y_t = np.zeros((N, self.K_t[t]))
        max_classes = np.where(self.balances[t] == max(self.balances[t]))
        for c in max_classes:
            Y_t[:, c] = 1.0
        Y_t /= Y_t.sum(axis=1).reshape(-1, 1)
        return torch.tensor(Y_t, dtype=torch.float)


class MajorityLabelVoter(RandomVoter):
    """
    A class that treats every task independently, placing all probability on 
    the majority label from all non-abstaining LFs for that task.

    Note that in the case of ties, non-integer probabilities are possible.
    """
    def train(self, L, **kwargs):
        # Note that this also sets some class parameters which we need later
        _ = self._check_L(L, init=True)

    def predict_task_proba(self, L, t):
        L = self._check_L(L)
        L_t = np.array(L[t].todense()).astype(int)
        N, M = L_t.shape
        K_t = self.K_t[t]
        Y_t = np.zeros((N, K_t))
        for i in range(N):
            counts = np.zeros(K_t)
            for j in range(M):
                if L_t[i,j]:
                    counts[L_t[i,j] - 1] += 1
            Y_t[i, :] = np.where(counts == max(counts), 1, 0)
        Y_t /= Y_t.sum(axis=1).reshape(-1, 1)
        return torch.tensor(Y_t, dtype=torch.float)
