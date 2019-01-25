import numpy as np
import cvxpy as cp
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

    def _force_singleton(self,deps):
        """Ensure that learned dependencies for singleton separators.
        TODO: build checker for no complete graphs with these additions
        Args:
            deps: (list) List of tuples of LF pairs that are dependent
        Example:
            if (0,1) and (1,2) exist, add (0,2)
            if (0,1) and (0,2) exist, add (1,2)
            if (0,3) and (2,3) exist, add (0,2)
            if (1,3) and (0,1) exist, add (0,3)
        """
        deps_singleton = []
        for i,j in deps:
            if i < j:
                deps_singleton.append((i,j))

        for i,j in deps:
            for k,l in deps:
                if (i == k) and (j < l):
                    deps_singleton.append((j,l))
                if (j == l) and (i < k):
                    deps_singleton.append((i,k))
                if (j == k) and (i < l):
                    deps_singleton.append((i,l))
                if (i == l) and (j < k):
                    deps_singleton.append((j,k))
        return deps_singleton


    def edges_from_amc(self,thresh=0.2):
        _, deps_all = self._amc(thresh)
        deps = []
        for i,j in deps_all:
            if i < j:
                deps.append((i,j))
        return self._force_singleton(deps)
    
    def edges_from_rpca(self,thresh=0.2):
        deps_all = self._rpca(thresh)
        deps = []
        for i,j in deps_all:
            if i < j:
                deps.append((i,j))
        #return self._force_singleton(deps)
        return deps

    def NON_edges_from_rpca(self,thresh=1e-2):
        non_deps_all = self._rpca(thresh)
        non_deps = []
        for i,j in non_deps_all:
            if i < j:
                non_deps.append((i,j))
        #return self._force_singleton(deps)
        return non_deps
    
    def _rpca(self,thresh=1.0,delta=1e-5):
        lam = 1/np.sqrt(self.m)
        z_hat, J_hat, L_rec, opt_error = self._run_optimization(self.Oinv, delta, lam)
        
        #find dependencies from S_inv
        deps_hat = get_deps_from_inverse_sig(J_hat, thresh) 
        return deps_hat

    def _amc(self, thresh=0.2, nonzeros=3):
        iterative_deps_mask = samplegrid(self.m,self.m,nonzeros)
        
        while(True):
            mu = solveMatrixCompletionWithMu(self.Oinv, iterative_deps_mask)
            max_val, max_ind, J_clean = find_largest(self.O, mu,self.m, iterative_deps_mask, thresh)
            
            if max_val < 1e-6: 
                return mu, iterative_deps_mask
            iterative_deps_mask.append(max_ind)
        return mu, iterative_deps_mask

    def _run_optimization(self,O_inv=None, delta=1e-5, lam=0.01, O=None, obj_option='default', const_option='default'): 
        """ Runs Robust PCA using over matrix O_inv given hyperparameters
            delta: noise control
            lambda: sparsity control
            Can change the objective function and constraint with
            obj_option: 'default' and 'no_diag'
            const_option:  'default' and 'no_diag'
        """

        # low-rank matrix
        L = cp.Variable([self.m,self.m])

        # sparse matrix
        S = cp.Variable([self.m,self.m])

        # objective and constraint definition
        if obj_option == 'default':
            objective = cp.Minimize(cp.norm(L, "nuc") + lam*cp.pnorm(S,1))
        if obj_option == 'likelihood':
            objective = cp.Minimize(cp.norm(L, "nuc") + lam*cp.pnorm(S-cp.diag(cp.diag(S)),1))

        if const_option == 'default':
            constraints = [cp.norm(O_inv-L-S, "fro") <= delta]
        if const_option == 'likelihood':
            constraints = [cp.norm(O_inv-L-S-cp.diag(cp.diag(S)), "fro") <= delt]

        # solve cvxpy problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=False)
        opt_error = prob.value

        # recover L and S
        J_hat = S.value
        L_rec = L.value
        if L_rec is not None:
            L_rec = 0.5*(L_rec + L_rec.T)
        else:
            print("L NOT RECOVERED")
            return 0,0,0,0

        # recover z_hat
        ev, eg = np.linalg.eig(L_rec)
        if ev[0] >= 0:
            z_hat = np.sqrt(ev[0])*eg[:,0]
        else:
            z_hat = -1*np.sqrt(-ev[0])*eg[:,0]

        return z_hat, J_hat, L_rec, opt_error