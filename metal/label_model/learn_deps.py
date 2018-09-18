import numpy as np
from metal.label_model.utils import *

class DependencyLearner():
    """A DependencyLearner TEMP VERSION

    Args:
        m: (int) the number of labeling functions
        Oinv: matrix of observed values (LL^T, Sigma_O_inverse)
        k: (int) the cardinality of the classifier
    """

    def __init__(self, m, Oinv, k=2, **kwargs):
        self.k = k #TODO: not used
        self.m = m
        self.Oinv = Oinv
        self.O = np.linalg.inv(self.Oinv)

    def edges_from_amc(self,thresh=0.2):
        _, deps_all = self._amc(thresh)
        deps = []
        for i,j in deps_all:
            if i < j:
                deps.append((i,j))
        return deps

    def _amc(self, thresh=0.2, nonzeros=3):
        iterative_deps_mask = samplegrid(self.m,self.m,nonzeros)
        
        while(True):
            mu = solveMatrixCompletionWithMu(self.Oinv, iterative_deps_mask)
            max_val, max_ind, J_clean = find_largest(self.O, mu,self.m, iterative_deps_mask, thresh)
            
            if max_val < 1e-6: 
                return mu, iterative_deps_mask
            iterative_deps_mask.append(max_ind)
        return mu, iterative_deps_mask


    