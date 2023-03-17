#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:06:12 2022

@author: dreveton
"""

import networkx as nx
import numpy as np
import scipy as sp


def computeLinkProbabilitiesBetweenClusters( G, labels_pred ):
    labels_pred = labels_pred.astype(int)
    K = len( set( labels_pred ) )

    communities = dict( )
    for k in range( K ):
        communities[ k ] = [ node for node in range( nx.number_of_nodes( G ) ) if labels_pred[ node ] == k+1 ]
    
    W = np.zeros( (K,K) )
    for (i,j) in G.edges():
        W[ labels_pred[ i ] - 1, labels_pred[ j ]-1 ] += 1
        W[ labels_pred[ j ] - 1, labels_pred[ i ]-1 ] += 1
    
    distances = np.zeros( K*(K-1)//2 )
    for k in range(K):
        for ell in range( k,K ):
            W[ k,ell ] = W[ k,ell ] / len( communities[ k ] ) / len( communities[ ell ] )
            W[ ell,k ] = W[ k,ell ]
            distances[ K * k + ell - ( (k+2)*(k+1) ) // 2 ] = 1-W[ k,ell ]
    
    return W 

def dasguptaCost( G, T, root = 0, cost_function = lambda a,b: a+b ):
    """
    This computes the cost as defined in Dasgupta 2006
    of a the hierarchical clustering of a graph G;
    the clustering is represented by a BINARY tree T, whose root is 0
    """
    C = 0
    A = nx.adjacency_matrix( G ).todense( )
    
    internal_nodes = [ node for node in T.nodes() if T.degree( node ) != 1 ]
    for node in internal_nodes:
        offsprings = [ u for u in T.neighbors( node ) if nx.shortest_path_length( T, u, root ) > nx.shortest_path_length( T, node, root ) ]
        C1 = findMegaCommunity( T, offsprings[ 0 ] )
        C2 = findMegaCommunity( T, offsprings[ 1 ] )
        W = A[ C1,: ]
        W= W[ :,C2 ]
        C += np.sum(W) * cost_function( len(C1), len(C2) )
    
    return C



def findMegaCommunity( T, node, root = 0):
    
    leaves = [ u for u in T.nodes() if T.degree( u ) == 1 ]
    
    if node in leaves:
        return T.nodes[ node ]['community']
    
    if node == root:
        megaCommunity = [ ]
        for leave in leaves:
            megaCommunity += T.nodes[ leave ]['community']
        return megaCommunity

    else:
        leave_from_node = [ u for u in leaves if nx.shortest_path_length( T, u, node ) < nx.shortest_path_length( T, u, root ) ]
        megaCommunity = [ ]
        for leave in leave_from_node:
            megaCommunity += T.nodes[ leave ]['community']
        return megaCommunity


def fromDendogramToTree( dendogram, N ):
    T = nx.Graph( )
    T.add_nodes_from( [ i for i in range( 2 * dendogram.shape[0] ) ] )
    
    u = N
    for i in range( dendogram.shape[0] ):
        T.add_edge( dendogram[i][0], u )
        T.add_edge( dendogram[i][1], u )
        u += 1
    
    root = u-1
    if T.degree[6]!=2:
        print( 'There is a problem in the root index' )
    
    for i in range( N ):
        T.nodes[ i ]['community'] = i
    
    edges = nx.bfs_edges(T, root)
    nodes = [root] + [v for u, v in edges]
    
    mapping = dict( )
    dummy = 0
    for i in range( nx.number_of_nodes( T ) ):
        mapping[ int( nodes[ i ] ) ] = dummy
        dummy += 1
    
    return nx.relabel_nodes(T, mapping)

