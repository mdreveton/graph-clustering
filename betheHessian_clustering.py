"""
This is the code from Lorenzo Dall'Amico
https://lorenzodallamico.github.io/codes/
corresponding to the paper
Dall'Amico, L., Couillet, R., & Tremblay, N. (2021). A unified framework for spectral clustering in sparse graphs. The Journal of Machine Learning Research, 22(1), 9859-9914.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import random
random.seed;
import time
import itertools
import networkx as nx
import sys
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix, bmat

import sys
#sys.path += ['/localdata/dallamil-admin/Scrivania/Work/A_Progetti/Community detection/Articolo/github_codes/Package/']  ### Specify the directory where the Package is
from betheHessian_generic_functions import *


######################################################################################################################################


def find_sol(S, M, r, eps):
    
    ''' Function that solves Equation 24 through dicotomy
    Use : 
        rp = find_sol(S, M, r)
    Input :
        S (array of size p x p) : diagonal matrix with the smallest eigenvalues of H_r
        M (array os size p x p) : X^T@D@X, where X is the n x p matrix containing the p smallest eigenvectors of H_r
        r (scalar) : value of r for which X and S are computed
    Output :
        rp (scalar) : value of r \in (1, r) solution to Equation 24
    '''
    
    r_small = 1 # r* > r_small
    r_large = r # r* < r_large
    err = 1
    r_old = r_large
    
    while err > eps:
            
        r_new = (r_small + r_large)/2 
        err = np.abs(r_old - r_new)
        
        v = max(np.linalg.eigvalsh(r_new*S + (r-r_new)*M)) # evaluate the largest eigenvalue in the midpoint
        
        if v > (r-r_new)*(1+r*r_new): # update the boundaries
            r_small = r_new
        else:
            
            r_large = r_new
            
        r_old = r_new
            
    return r_large # return the right edge


######################################################################################################################################

def find_rho_B(A):
    
    ''' Function that computes rho(B)
    Use : 
        rho = find_rho_B(A)
    Input :
        A (array of size n x n) : sparse representation of the adjacency matrix
    Output :
        rho (scalar) : leading eigenvalue of the non-backatracking matrix
    '''
    
    n = np.shape(A)[0] # size of the network
    d = np.array(np.sum(A, axis = 0))[0] # degree vector
    D = scipy.sparse.diags(d, offsets = 0) # degree matrix
    I = scipy.sparse.diags(np.ones(n), offsets = 0) # identity matrix
    M = scipy.sparse.bmat([[A, I - D], [I, None]], format='csr') # matrix B'
    vM = scipy.sparse.linalg.eigs(M, k=1, which='LM', return_eigenvectors=False) # find the largest eigenvalue of B'
    return max(vM.real)

######################################################################################################################################



def find_zeta(A, rho, n_clusters, eps):
    
    ''' Function that calculates the vector zeta on a connected network A given k as zeta_p = min_{r > 1} {r : s_p(H_r) = 0}
    Use : 
        zeta_v, Y = find_zeta(A, rho, n_clusters, eps)
    Input :
        A (sparse matrix of size n) : adjacency matrix of the network
        rho (scalar) : spectral radius of the non-backtracking matrix
        n_clusters (scalar) : number of clusters k
        eps (scalar) : precision of the estimate
    Output :
        zeta_v (array of size k) : vector containing the values of zeta_p for 1 \leq p \leq k
        Y (array of size n x k) : matrix containing the informative eigenvectors on which k-means whould be performed
    '''
    

    d = np.array(np.sum(A, axis = 0))[0] # degree vector
    n = len(d) # size of the network
    D = scipy.sparse.diags(d, offsets = 0) # degree matrix
    I = scipy.sparse.diags(np.ones(n), offsets = 0) # identity matrix
    zeta_v = np.ones(n_clusters)
    Y = np.zeros((n, n_clusters))
    r = np.sqrt(rho) # initialization of r = sqrt{rho(B)}
    i = n_clusters
    
    while i > 1:
        
        delta = 1
        OUT = 'Estimating zeta : ' + str(i).zfill(2)
        sys.stdout.write('\r%s' % OUT)
        
        while delta > eps: # iterate while r*-r is smaller than eps
        
            H = (r**2-1)*I + D - r*A # Bethe-Hessian
            v, X = scipy.sparse.linalg.eigsh(H, k = i, which = 'SA') # compute the i+1 smallest eigenvalues and eigenvectors
            idx = v.argsort()
            v = v[idx]
            X = X[:,idx]
            S = np.diag(v) 
            M = X.T@D@X
            rp = find_sol(S, M, r, eps) # iterative solution of Equation 24
            delta = np.abs(r - rp) # updated value of delta
            r = rp # r <- r*      
                   
        degeneracy = sum(np.abs(v[1:]-v[-1]) < eps) # calculate the degeneracy of the i-th smallest eigenvalue
        zeta_v[i-degeneracy:i] = r # store the last value of r* found
        Y[:,i-degeneracy:i] += X[:,i-degeneracy:i] # store the corresponding eigenvectors
        i = i-degeneracy
        
        
    return zeta_v, Y



######################################################################################################################################




class ReturnValue:
    def __init__(self, estimated_labels, n_clusters, ov, mod, zeta_p):
        self.estimated_labels = estimated_labels
        self.n_clusters = n_clusters
        self.overlap = ov
        self.modularity = mod
        self.zeta_v = zeta_p


def community_detection(A, *args, **kwargs):
    
    '''Function to perform community detection on a graph with n nodes and k communities according to Algorithm 2
    Use : 
        cluster = community_detection(A, **kwargs)
    Input :
        A (sparse matrix n x n) : adjacency matrix
        **kwargs:
            n_max (scalar) : maximal number of possible classes to look for during the estimation. If not specified set equal to 80
            real_classes (array of size n) : vector containing the true labels of the network. If not specified set to None
            n_clusters (scalar) : number of clusters k. If not specified it will estimate it
            eps (scalar) : precision rate. If not specified set to machine precision
            projection (True/False) : performs the projection on the unitary hypersphere in dimension k, before the k-means step. If not else specified, set to true
            
    Outup :
        cluster.estimated_labels (array of size n) : vector containing the estimated labels
        cluster.n_clusters (scalar) : estimated values of k used in the k-means step
        cluster.overlap (scalar) : overlap with respect to the known partition
        cluster.modularity (scalar) : modularity of the estimated partition
        cluster.zeta_v (array of size k) : vector containing the values of zeta_p
    
    '''
    
    n_max = kwargs.get('n_max', 80)
    real_classes = kwargs.get('real_classes', [None])
    n_clusters = kwargs.get('n_clusters', None)
    eps = kwargs.get('eps', np.finfo(float).eps)
    projection = kwargs.get('projection', True)
    
    d = np.array(np.sum(A,axis = 0))[0] # degree vector
    n = len(d) # size of the network
    rho = find_rho_B(A) # r = rho(B)
    
    if n_clusters == None: # it the number of clusters is not known, we estimate it  
        
        n_clusters = 1 
        D_rho_05 = scipy.sparse.diags((d + (rho -1)*np.ones(n))**(-1/2), offsets = 0) 
        L_rho = D_rho_05.dot(A).dot(D_rho_05) # symmetric reduced Laplacian at tau = rho(B)-1
        flag = 0
        while flag == 0:
            if n_clusters < n_max: # the algo will not find more than n_max clusters
                vrho = scipy.sparse.linalg.eigsh(L_rho, k = n_clusters + 1 , which='LA', return_eigenvectors=False) # largest eigenvalues of L_tau
                if min(vrho)> 1/np.sqrt(rho) + np.finfo(float).eps: #  if informative
                    n_clusters += 1
                    OUT = 'Number of clusters detected : ' + str(n_clusters)
                    sys.stdout.write('\r%s' % OUT)
                    sys.stdout.flush()
                else:
                    flag = 1
            else:
                flag = 1



    print('\n')
    zeta_p, X = find_zeta(A, rho, n_clusters, eps) # find the zeta vector and  the corresponding informative  matrix
    
    if projection == True:
    
        for i in range(n):
            X[i] = X[i]/np.sqrt(np.sum(X[i]**2)) # normalize the rows  of X

    kmeans = KMeans(n_clusters = n_clusters) # perform kmeans on the informative eigenvector
    kmeans.fit(X)
    estimated_labels = kmeans.predict(X)
    
    print('\nLabels estimated')

    if real_classes[0] == None:
        ov = 'Not available'
    else:
        ov = overlap(real_classes, estimated_labels) # compute the overlap
        
    mod =  find_modularity(A, estimated_labels) # compute the modularity

    return ReturnValue(estimated_labels, n_clusters, ov, mod, zeta_p)


