import cvxpy as cp
import numpy as np

from metal.label_model.utils import *
from metal.label_model.learn_deps import DependencyLearner

class RPCADependencyLearner(DependencyLearner):
    """Dependency Learner based on Robust PCA Decomposition.

    

    Args:
        k: (int) The cardinality of the dataset
        k: (int) The cardinality of the dataset
    """

    def __init__(self, k=2):
        self.k = k
        super().__init__(k)


    def run_optimization(self, O_inv, delta, lam, obj_option='default', const_option='default'): 
        """ Runs Robust PCA using over matrix O_inv given hyperparameters.

        Args:
            delta: (float) noise control
            lambda: (float) sparsity control
        """

        # low-rank matrix and sparse matrix
        m = np.shape(O_inv)[0]
        L = cp.Variable([m,m])
        S = cp.Variable([m,m])

        # objective and constraint definition
        objective = cp.Minimize(cp.norm(L, "nuc") + lam*cp.pnorm(S,1))
        constraints = [cp.norm(O_inv-L-S, "fro") <= delta]
        
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

        return J_hat, L_rec, opt_error

    #LEARNING EDGES
    def find_edges(self, sigma_O,thresh=0.2):
        """Learn dependencies between the sources using robust PCA.

            Args:
                sigma_O: A [m,m] np.array representing Cov[\psi(O)], where O is the
                    set of observable cliques of sources, \psi(O) is the vector of
                    indicator random variables for O, and sigma_O = Cov[\psi(O)] is
                    the generalized covariance for O. Either this or L_train 
                    must be provided.
                thresh: (float) values above this value in the recovered 
                sparse matrix count as a dependency

            Note that the sigma_O matrix is always mxm, where m is the number
            of supervision sources. 

            We learn the dependencies between supervision sources by solving:
                \min_{L,S} \|L\|_* + \lambda \|S\|_1  
                subject to  \|M-L-S\|_F \leq \delta

            where M is Sigma_O^{-1}, L is a low-rank matrix, S is a sparse matrix with
            non-zero values at entries (i,j) is sources i,j are dependent.

            For further details, see:
            https://arxiv.org/pdf/1001.2363.pdf
        """

        lam=1/np.sqrt(float(np.shape(sigma_O)[0]))
        J_hat, _, _ = self.run_optimization(np.linalg.inv(sigma_O), delta=1e-5,lam=lam)

        deps_all = get_deps_from_inverse_sig(J_hat, thresh) 
        deps = self._force_singleton(deps_all)

        return deps