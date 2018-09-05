import numpy as np
from numpy.random import random
import copy
import time
from math import exp

def compare_deps(deps1,deps2):
    # expects two lists of tuples
    diff = set(deps1).symmetric_difference(set(deps2))
    return len(set(deps1) - set(deps2))

def get_deps_from_inverse_sig(J, thresh=0.2):
    deps = []
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if abs(J[i,j]) > thresh:
                deps.append((i,j))
    return deps

def gen_accs_high_acc(m, mu_normal=0.6):
    mu_high = 1 - 1/float(m-1)*5*(1-mu_normal)
    return mu_high

def gen_accs_high_acc_ratio(m, a, b, mu_normal=0.6):
    mu_high = 1 - float((b-a)*(1-mu_normal))/float(a*(m-1))
    return mu_high

def return_acc_vector(m, mu_normal, mu_high):
    mu = [mu_normal for _ in range(m)]
    mu[-1] = mu_high
    return mu

def find_largest(O,mu,dim,mask,thresh):
    prod = np.outer(mu, mu)
    #print("PROD SHAPE: ", prod.shape)
    #print("O SHAPE: ", O.shape)
    C = O - prod
    #print("C SHAPE: ", C.shape)
    try:
        J = np.linalg.pinv(C)
    except:
        print("Failed to invert C in find largest")
        #print("C: ", C)
        #print("O: ", O)
        #print("mu mu^T: ", prod)
        J = np.zeros((dim,dim),dtype=float)
        return 0, (-1,-1), J

    max_val = 0
    max_ind = (-1,-1)
    #print("J SHAPE")
    #print(J.shape)
    J_clean = copy.deepcopy(J)
    for i in range(dim):
        for j in range(dim):
            if abs(J[i,j]) <= thresh:
                J_clean[i,j] = 0
            if (i,j) not in mask and abs(J_clean[i,j]) > max_val:
                max_val = abs(J_clean[i,j])
                max_ind = (i,j)
    return max_val, max_ind, J_clean

def solveMatrixCompletion(O_inv, deps):
    zeros_set = []
    for i in range(O_inv.shape[0]):
        for j in range(O_inv.shape[1]):
            zeros_set.append((i,j))
    zeros_set = set(zeros_set)
    zeros_set = zeros_set - set(deps)
    zeros_set = list(zeros_set)
    
    #form q
    q = np.zeros((len(zeros_set),),dtype=float)
    M = np.zeros((len(zeros_set),O_inv.shape[0]),dtype=float)
    for ix, z in enumerate(zeros_set):
        M[ix, z[0]] = 1
        M[ix, z[1]] = 1
    #print(M)
    #print(M.shape)
    for ix, z in enumerate(zeros_set):
        #print(z)
        q[ix] = np.log(O_inv[z[0],z[1]]**2)
    #print(len(zeros_set))
        
    l = np.linalg.pinv(M) @ q
    #print(M@l)
    #print(q)
    return l

def calculate_empirical_mu(z,O):
    c = 1 + z.dot(O.dot(z.T)) # check this
    mu = 1/np.sqrt(c)*O.dot(z.T)
    return mu

def solveMatrixCompletionWithMu(O_inv, O, deps):
    l = solveMatrixCompletion(O_inv,deps)
    l = np.exp(l)
    z_rec = np.sqrt(l)
    mu = calculate_empirical_mu(z_rec,O)
    return mu

def samplegrid(w, h, n):
    sampled = []
    for i in range(w):
        sampled.append((i,i))
    return sampled