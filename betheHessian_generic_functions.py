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
from networkx.algorithms.community.quality import modularity


####################################################################################################################


def adj(C_matrix,c, label, theta):
    
    ''' Function that generates the adjacency matrix A with n nodes and k communities
    Use:
        A = adj(C_matrix,c, label, theta)
    Input:
        C_matrix (array of size k x k) : affinity matrix of the network C
        c (scalar) : average connectivity of the network
        label (array of size n) : vector containing the label of each node
        theta  (array of size n) : vector with the intrinsic probability connection of each node
    Output:
        A (sparse matrix of size n x n) : symmetric adjacency matrix
        '''

    k = len(np.unique(label)) # number of communities
    fs = list()
    ss = list()

    n = len(theta)
    c_v = C_matrix[label].T # (k x n) matrix where we store the value of the affinity wrt a given label for each node
    first = np.random.choice(n,int(n*c),p = theta/n) # we choose the nodes that should get connected wp = theta_i/n: the number of times the node appears equals to the number of connection it will have

    for i in range(k): 
        v = theta*c_v[i]
        first_selected = first[label[first] == i] # among the nodes of first, select those with label i
        fs.append(first_selected.tolist())
        second_selected = np.random.choice(n,len(first_selected), p = v/np.sum(v)) # choose the nodes to connect to the first_selected
        ss.append(second_selected.tolist())

    fs = list(itertools.chain(*fs))
    ss = list(itertools.chain(*ss))

    fs = np.array(fs)
    ss  = np.array(ss)

    edge_list = np.column_stack((fs,ss)) # create the edge list from the connection defined earlier

    edge_list = np.unique(edge_list, axis = 0) # remove edges appearing more then once
    edge_list = edge_list[edge_list[:,0] > edge_list[:,1]] # keep only the edges such that A_{ij} = 1 and i > j

    G = nx.Graph()
    G.add_edges_from(edge_list)
    A = nx.adjacency_matrix(G, nodelist = np.arange(n)) # this creates a symmetric sparse matrix

    return A


####################################################################################################################


def matrix_C(c_out, c,fluctuation, fraction):
    
    ''' Function that generates the matrix C
    Use :
        C_matrix = matrix_C(c_out, c,fluctuation, fraction)
    Input:
        c_out (scalar) : average value of the of diagonal terms
        c (scalar) : average connectivity of the desired network
        fluctuation (scalar) : the off diagonal terms will be distributed according to N(c_out, c_out*fluctuation)
        fraction  (array of size equal to the number of clusters - k -) : vector \pi containing the  fraction of nodes in each class
    Output:
        C_matrix (array of size k x k) : affinity matrix C
        '''
    
    n_clusters = len(fraction)
    C_matrix = np.abs(np.random.normal(c_out, c_out*fluctuation, (n_clusters,n_clusters))) # generate the  off diagonal terms
    C_matrix = (C_matrix + C_matrix.T)/2 # symmetrize the  matrix
    nn = np.arange(n_clusters) 
    for i in range(n_clusters):
        x = nn[nn != i]
        C_matrix[i][i] = (c - (C_matrix[:,x]@fraction[x])[i])/fraction[i] # imposing CPi1 = c1

    return C_matrix  


#########################################################################################################################


def spec_L(A):
    
    ''' Function computes the spectrum of the matrix L = D^{-1/2}AD^{-1/2}
    Use :
        eig, vec = spec_L(A)
    Input:
        A (n times n sparse matrix): sparse representation of the adjacency matrix
    Output:
        eig (array of size n) : vector containing all the eigenvalues of L
        vec (array of size n x 2) : array with the two dominant eigenvalues of L
        '''
    
    d = np.array(np.sum(A,axis = 0))[0] # degree vector
    D_05 = scipy.sparse.diags(d**(-1/2),offsets = 0) # matrix D^{-1/2}
    L = D_05.dot(A.dot(D_05)) # matrix L = D^{-1/2}AD^{-1/2}
    eig = np.linalg.eigvalsh(L.A) # all eigenvalues of L
    v, vec = scipy.sparse.linalg.eigsh(L, k = 2, which = 'LA') # two largest eigenvalues  and eigenvectors of L
    
    return eig, vec

#########################################################################################################################

def dominant_B(A, n_clusters):
    
    ''' Function that computes the k largest eigenvalues of the non-backtracking matrix
    Use : 
        nu = dominant_B(A, n_clusters)
    Input :
        A (array of size n x n) : sparse representation of the adjacency matrix
        n_clusters (scalar) : number of clusters k
    Output :
        zeta_v (array of size k) : vector containing the vlaues of zeta_p for 1 \leq p \leq k
    '''
    
    d = np.array(np.sum(A,axis = 0))[0] # degree vector
    n = len(d) # size of the network
    D = scipy.sparse.diags(d, offsets = 0) # degree matrix
    I = scipy.sparse.diags(np.ones(n), offsets = 0) # identity matrix
    M = scipy.sparse.bmat([[A, I - D], [I, None]], format='csr') # matrix B'
    
    v, vv = scipy.sparse.linalg.eigs(M, k = n_clusters, which = 'LR') # largest eigenvalues of B'
    v = v.real # the largest eigenvalues of Bp are real
    idx = v.argsort()[::-1]   # sort the eigenvalues
    v = v[idx]
    
    return v

	

#########################################################################################################################

def overlap(real_classes, classes):

    '''Computes the overlap in neworks (with n nodes) with more then two classes and find the good permutation of the labels

    Use : 
        ov = overlap(real_classees, classees)
    Input : 
        real_classes (array of size n) : vector with the true labels
        classes (array of size n) : vector of the estimated labels
    Output : 
        ov (scalar) : overlap

    '''
    values = max(len(np.unique(real_classes)),len(np.unique(classes))) # number of classes
    n = len(classes) # size of the network

    matrix = np.zeros((values,values))
    for i in range(n):
        matrix[classes[i]][real_classes[i]] += 1 # n_classes x n_classes confusion matrix. Each entry corresponds to how many time label i and label j appeared assigned to the same node

    positions = np.zeros(values)
    for i in range(values):
        positions[i] = np.argmax(matrix[i]) # find the good assignment

    dummy_classes = (classes+1)*100
    for i in range(values):
        classes[dummy_classes == (i+1)*100] = positions[i] # reassign the labels

    n_classes = len(np.unique(real_classes))
    
    ov = (sum(classes == real_classes)/n - 1/n_classes)/(1-1/n_classes) # compute the overlap

    return ov
    v = np.sort(v)[::-1] # sort them
    
    return v  

#########################################################################################################################

def find_modularity(A, estimated_labels):
    
    '''Function to compute the modularity of a given partition on a network with n nodes
    Use: 
        mod =  modularity(A, estimated_labels)
    Input:
        A (sparse matrix of size n x n) : adjacency matrix of the network
        estimated_labels (array of size n) : vector containing the assignment of the labels
    Output:
        mod (scalar) : modularity of the assignment
    '''
    
    d = np.array(np.sum(A, axis = 0))[0] # degree vector
    m = sum(d) # 2|E|
    n_clusters = len(np.unique(estimated_labels)) # number  of clusters
    mod = 0
    for i in range(n_clusters):
        I_i = (estimated_labels == i)*1 # indicator vector : the entry j equal to 1 if node j belongs to class i
        mod += I_i@A@I_i - (d@I_i)**2/m 
        
    return mod/m


