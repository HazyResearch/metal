from collections import Counter

import numpy as np
from scipy.sparse import csc_matrix
import torch

from metal.utils import rargmax
from metal.label_model.label_model import LabelModel

class RandomVoter(LabelModel):
    """
    A class that votes randomly among the available labels
    """
    def train(self, L, **kwargs):
        pass

    def predict_proba(self, L):
        """
        Args:
            L: An [N, M] scipy.sparse matrix of labels
        Returns:
            output: A [N, K_t] np.ndarray of soft predictions
        """
        N = L.shape[0]
        Y_p = np.random.rand(N, self.k)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityClassVoter(RandomVoter):
    """
    A class that places all probability on the majority class based on class 
    balance (and ignoring the label matrix).

    Note that in the case of ties, non-integer probabilities are possible.
    """
    def train(self, L, balance, **kwargs):
        """
        Args:
            balance: An np.arrays that sums to 1, corresponding to the
                (possibly estimated) class balance.
        """
        self.balance = np.array(balance)
        
    def predict_proba(self, L):
        N = L.shape[0]
        Y_p = np.zeros((N, self.k))
        max_classes = np.where(self.balance == max(self.balance))
        for c in max_classes:
            Y_p[:, c] = 1.0
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityLabelVoter(RandomVoter):
    """
    A class that places all probability on the majority label from all 
    non-abstaining LFs for that task.

    Note that in the case of ties, non-integer probabilities are possible.
    """
    def train(self, L, **kwargs):
        pass

    def predict_proba(self, L):
        L = np.array(L.todense()).astype(int)
        N, M = L.shape
        Y_p = np.zeros((N, self.k))
        for i in range(N):
            counts = np.zeros(self.k)
            for j in range(M):
                if L[i,j]:
                    counts[L[i,j] - 1] += 1
            Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p
