import numpy as np
from numpy.random import random
import math
import copy

class DataGenerator(object):
    """
    Generates synthetic data or distributions 
    """ 

    def __init__(self, m, n=None, accs=None, coverage=None):
        """ Initialize DataGenerator object
        m: size of matrix
        n: number of samples (None for distribution)
        accs: m-length array for accuracies (None for random generation)
        coverage: m-length array for coverage (None for 100% coverage)
        """

        self.m = m
        self.n = n
        self.coverage = coverage
        if accs is None:
            self.accs = [0.75 + (0.3 * random() - 0.15) for _ in range(self.m)]
        else:
            self.accs = accs
        self.mu = None
        self.deps = None
        self.deps_no_diag = None
        self.O = None
        self.Oinv = None
        self.sig = None
        self.L = None
        self.Y = None

    def generate_O(self, k=0):
        """
        Generate data (sampled or distribution) for given case
        Returns:
        O: observed disagreement matrix
        Oinv: inverse of observed disagreement matrix
        """

        #independent case
        if k == 0:
            if self.n is None:
                O, Oinv, sig = self.generate_independent()
                self.sig = sig
                self.mu = self.accs
            else:
                L,Y, mu_from_acc, accuracies = self.generate_independent_sampled(self.coverage)
                O = np.dot(L,L.T)/(self.n-1)
                Oinv = np.linalg.inv(O)
                sig = O - np.outer(mu_from_acc,mu_from_acc)
                self.sig = sig
                self.mu = copy.deepcopy(mu_from_acc)
                self.L = copy.deepcopy(L)
            
            self.deps = None

        #dependency case
        else:
            if self.n is None:
                O, Oinv, sig, deps = self.generate_block_matrix_dist(k)
                self.mu = self.accs
                
            else:
                L,Y, mu_from_acc, accuracies, deps = self.generate_block_matrix_sampled(k,self.coverage)
                O = np.dot(L,L.T)/(self.n-1)
                Oinv = np.linalg.inv(O)
                sig = O - np.outer(mu_from_acc,mu_from_acc)
                self.mu = copy.deepcopy(mu_from_acc)
                self.L = copy.deepcopy(L)
            
            self.deps = copy.deepcopy(deps)
            self.deps_no_diag = [dep for dep in self.deps if dep[0] != dep[1]]
        
        self.O = copy.deepcopy(O)
        self.Oinv = copy.deepcopy(Oinv)
        self.sig = copy.deepcopy(sig)
        self.Y = copy.deepcopy(Y)

    def generate_independent(self):
        self.mu = np.array(self.accs)
        var = np.multiply(self.mu,(1.0-self.mu))
        ratio = np.max(np.divide(self.mu,var))/(np.sum(np.divide(self.mu,var)))

        sig = np.diag(var)
        mu = np.reshape(self.mu,[self.m,1])
        O = sig + mu @ mu.T
        Oinv = np.linalg.inv(O)

        return O, Oinv, sig

    def generate_independent_sampled(self, coverage):
        # Generate the true class labels (balanced)
        #TODO: include class imbalance option
        Y = np.array([-1 if random() < 0.5 else 1 for i in range(self.n)])
        
        if coverage:
            betas = [beta for _ in range(self.m)]
        else:
            betas = [1.0 for _ in range(self.m)]

        #Generate Labeling Functions
        L = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                L[i,j] = Y[j] if random() < self.accs[i] else -Y[j]
        
        # calculate mu's here
        accuracies = np.array([np.mean(L[i,:] == Y) for i in range(self.m)])
        mu_from_acc = 2*accuracies - 1

        return L,Y, mu_from_acc, accuracies



    def generate_covariance(self, mu_x, mu_y):
        if 0.05 < (min(mu_x,mu_y) - (mu_x*mu_y)):
            cov = np.random.uniform(0.05, (min(mu_x,mu_y) - (mu_x*mu_y)))
        else: 
            #this happens when we have accuracies that are very different
            cov = 0.0

        Ex0y0 = 1 - mu_x - mu_y + (mu_x*mu_y) + cov
        Ex0y1 = 1 - mu_x - Ex0y0
        Ex1y1 = mu_y - Ex0y1
        Ex1y0 = mu_x - Ex1y1

        assert Ex0y0 + Ex0y1 + Ex1y1 + Ex1y0 == 1.
        assert (Ex0y0 > 0.) and (Ex0y1 > 0.) and (Ex1y0 > 0.) and (Ex1y1 > 0.)

        return cov

    def generate_block_matrix_sampled(self,k,coverage,beta=1):
        assert ( k <= math.floor(self.m/2))
        Y = [-1 if random() < 0.5 else 1 for i in range(self.n)]
        Y = np.array(Y)
        #Y = np.array([-1 if random() < 0.5 else 1 for i in range(self.n)])

        #Generate accuracies and coverage
        accs = [0.75 + (0.3 * random() - 0.15) for _ in range(self.m-k)]
        if coverage:
            betas = [beta for _ in range(self.m)]
        else:
            betas = [1.0 for _ in range(self.m)]
        # k lfs
        # 0,2,4,6 --> LFs
        # 1,3,5,7 --> features
        #deps = [(i,i) for i in range(self.m)]
        deps = []
        L = np.zeros(( self.m,self.n))
        for i in range(k):
            deps.append((2*i,2*i+1))
            #deps.append((2*i+1,2*i))
            for j in range(self.n):
                L[2*i,j] = Y[j] if random() < accs[i] else -Y[j]
                if ((random() < 0.75) and (L[2*i,j] == Y[j]) and (Y[j] == 1)) or ((random() < 0.25) and (L[2*i,j] != Y[j])  and (Y[j] == 1)):
                    L[2*i+1,j] = 1
                else:
                    L[2*i+1,j] = -1
        # the remaining LFs
        for i in range(2*k,self.m):
            for j in range(self.n):
                L[i,j] = Y[j] if random() < accs[i-k] else -Y[j]

        # calculate mu's here
        accuracies = np.array([np.mean(L[i,:] == Y) for i in range(self.m)])
        mu_from_acc = 2*accuracies - 1

        return L,Y, mu_from_acc, accuracies, deps

    def generate_block_matrix_dist(self,k):
        assert ( k <= math.floor(self.m/2))
        mu = np.array(self.accs)
        var = np.multiply(mu,(1.0-mu))
        sig = np.diag(var)
        deps = [(i,i) for i in range(self.m)]
        for i in range(k):
            deps.append((2*i, 2*i+1))
            deps.append((2*i+1, 2*i))
            cov = self.generate_covariance(mu[2*i],mu[2*i+1])
            sig[2*i,2*i+1] = cov
            sig[2*i+1,2*i] = cov
    
        O = sig + np.outer(mu,mu)
        Oinv = np.linalg.inv(O)
        return O, Oinv, sig, deps
