#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:00:15 2022

@author: dreveton
"""

import networkx as nx
import numpy as np
import scipy as sp

from sklearn.cluster import SpectralClustering, KMeans



def betheHessian( G, weighted = False):
    N = nx.number_of_nodes( G )
    A = nx.adjacency_matrix( G )
    
    degrees = A@np.ones( N )
    D = np.diag( degrees )
    D = sp.sparse.csr_matrix( D )
    
    r = np.sum( degrees**2 ) / np.sum( degrees ) - 1
    r = np.sqrt( r )
    
    if weighted:
        A = A.todense()
        H = np.zeros( (N,N) )
        for i in range( N ):
            neigh_i = [ u for u in G.neighbors( i ) ]
            H[i,i] = 1 + np.sum( [ A[i,j]**2 / (r**2 - A[i,j]**1) for j in neigh_i ] )
            for j in neigh_i:
                H[ i,j ] = -r * A[ i, j ] / ( r**2 - A[ i, j ] )
        H = (r**2-1) * sp.sparse.csr_matrix( H )
    
    else:
        H = (r**2-1) * sp.sparse.eye( N ) + D - r * A
    
    return H



def betheHessianClustering( G ):
    
    H = betheHessian( G, weighted = False )
    
    H = H.todense()

    vals, vecs = np.linalg.eigh( H )
    negativeEigenvaluesIndex = [ i for i in range( nx.number_of_nodes( G ) ) if vals[ i ] < 0 ]
    vecs = vecs[ :, negativeEigenvaluesIndex ]
    
    kmeans = KMeans( n_clusters = len( negativeEigenvaluesIndex ), random_state = 0 ).fit( vecs )
    labels_pred = kmeans.labels_ + np.ones( nx.number_of_nodes( G ), dtype = int )
    
    return labels_pred


