#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:46:06 2022

@author: dreveton
"""

import networkx as nx
import numpy as np
import scipy as sp

from sklearn.cluster import SpectralClustering, KMeans

import random
from itertools import chain, combinations

from scipy.cluster.hierarchy import linkage
import graphClustering_flat as clustering
import auxiliary as auxilliary


# =============================================================================
# BOTTOM-UP CLUSTERING
# =============================================================================

def computeDistanceBetweenClusters( G, labels_pred ):
    K = len( set( labels_pred ) )
    W = auxilliary.computeLinkProbabilitiesBetweenClusters( G, labels_pred )
    
    distances = np.zeros( K*(K-1)//2 )
    for k in range(K):
        for ell in range( k+1,K ):
            distances[ K * k + ell - ( (k+2)*(k+1) ) // 2 ] = 1-W[ k,ell ]

    return distances

    

def bottomUpClustering( G, clusteringAlgo = clustering.betheHessianClustering, 
                       linakgeAlgo = linkage, linkageMethod = 'single' ):
    
    labels_pred = clusteringAlgo( G )

    distances = computeDistanceBetweenClusters( G, labels_pred )
    dendogram = linkage( distances, method = linkageMethod )
    
    return labels_pred, dendogram




# =============================================================================
# RECURSIVE BI-PARTITIONNING
# =============================================================================

def selectionRule( G ):
    H = clustering.betheHessian( G, weighted = False )
    #vals, vecs = sp.sparse.linalg.eigsh( H.asfptype() , k = 2, which = 'SA' )
    vals, vecs = sp.linalg.eigh( H.toarray( ) )
    
    if vals[1] < 0:
        return True
    else:
        return False
    
def biPartitioning( G ):
    n = nx.number_of_nodes( G )
        
    L = nx.normalized_laplacian_matrix( G )
    vals, vecs = sp.sparse.linalg.eigsh( L.asfptype() , k = 2, which = 'SM' )
    
    kmeans = KMeans( n_clusters = 2, random_state=0 ).fit( vecs )
    labels_pred_spec = kmeans.labels_ + np.ones( nx.number_of_nodes(G) )
    C1 = [ i for i in range( n ) if labels_pred_spec[ i ] == 1 ]
    C2 = [ i for i in range( n ) if labels_pred_spec[ i ] == 2 ]
    
    nodes = np.array( G.nodes() )
    
    return nx.induced_subgraph( G, nodes[C1] ), nx.induced_subgraph( G, nodes[C2] )


def _makeTree( sequence, add_nodes_to_the_leaf = True ):
    """Recursively creates a tree from the given sequence of nested tuples.
        This function employs the :func:`~networkx.tree.join` function
        to recursively join subtrees into a larger tree.
        """
        # The empty sequence represents the empty tree, which is the
        # (unique) graph with a single node. We mark the single node
        # with an attribute that indicates that it is the root of the
        # graph.
    if isinstance( sequence, np.int64):
        G = nx.empty_graph( 1 )
        nx.set_node_attributes(G, sequence, "community")
        return G
    
    if len(sequence) == 0:
        return nx.empty_graph( 1 )
    #For a non-empty sequence that is not a tuple, it means we arrived at the leaf of the tree,
    #and hence sequence corresponds to the list of nodes that are attached to this leaf and form a community.
    if not isinstance(sequence, tuple ) and add_nodes_to_the_leaf:
        G = nx.empty_graph( 1 )
        nx.set_node_attributes(G, sequence, "community")
        return G
        # For a nonempty sequence, get the subtrees for each child
        # sequence and join all the subtrees at their roots. After
        # joining the subtrees, the root is node 0.
    return nx.tree.join( [ ( _makeTree( child ), 0) for child in sequence ] )

def makeTree( sequence, sensible_relabeling = True ):
    
    T = _makeTree( sequence )
    
    if sensible_relabeling:
        return relabelTreeAccordingToBFS( T, root = 0 )
    else:
        return T
    
def relabelTreeAccordingToBFS( T, root = 0 ):
    # Relabel the nodes according to their breadth-first search
    # order, starting from the root node (that is, the node 0).
    bfs_nodes = chain([0], (v for u, v in nx.bfs_edges(T, 0)))
    labels = {v: i for i, v in enumerate(bfs_nodes)}
    # We would like to use `copy=False`, but `relabel_nodes` doesn't
    # allow a relabel mapping that can't be topologically sorted.
    T = nx.relabel_nodes(T, labels)
        
    # The following is to make sure that the node ordering is correct (sorted)
    # If not, problem when we will work with the adjacency matrix
    H = nx.Graph( )
    H.add_nodes_from( sorted( T.nodes( data=True ) ) )
    H.add_edges_from( T.edges(data=True) )
    
    return H



def recursiveBiParitioning( G ):
    
    def _topDownMainProcedure( G ):    
        
        if( selectionRule( G ) ):
            print('Running')
            G_right, G_left = biPartitioning( G )
            return _topDownMainProcedure( G_right ), _topDownMainProcedure( G_left )
        else:
            return list( G.nodes() )

    communities = _topDownMainProcedure( G )
    
    T_predicted = makeTree( communities )
    
    labels_pred = np.zeros( nx.number_of_nodes( G ), dtype = int )
    
    leaves = [ x for x in T_predicted.nodes() if T_predicted.degree( x ) == 1 ]
    dummy = 1
    for leave in leaves:
        for node in T_predicted.nodes[ leave ][ 'community' ]:
            labels_pred[ node ] = dummy
        dummy += 1
    
    return T_predicted, labels_pred




# =============================================================================
# 
# =============================================================================

# Select the clustering after k merges
def select_clustering(D, k):
    n = np.shape(D)[0] + 1
    k = min(k,n - 1)
    cluster = {i:[i] for i in range(n)}
    for t in range(k):
        cluster[n + t] = cluster.pop(int(D[t][0])) + cluster.pop(int(D[t][1]))
    return sorted(cluster.values(), key = len, reverse = True)


def _linkagePlusPlus( G, K ):
    A = nx.adjacency_matrix( G )
    
    randomNumber = np.random.randint( K+1, nx.number_of_nodes(G)-1 )
    chosenColumns = random.choices( [ i for i in range( nx.number_of_nodes(G) ) ] , k = randomNumber )
    Ahat = A[ :, chosenColumns ]
    
    left_vecs, vals, right_vecs = sp.sparse.linalg.svds( Ahat.asfptype( ), k = K , return_singular_vectors = 'u' )
    
    dendogram = linkage( left_vecs, method = 'single' )
    
    level = dendogram.shape[0] - (K-2)
    communities = select_clustering(dendogram, level-1 )
    
    labels_pred = np.zeros( nx.number_of_nodes(( G ) ), dtype = int )
    for k in range(K):
        for i in communities[ k ]:
            labels_pred[ i ] = k+1
    
    distances = computeDistanceBetweenClusters( G, labels_pred )
    dendogram = linkage( distances, method = 'single' )

    return labels_pred, dendogram


def linkagePlusPlusKnowK( G, K ):
    N = nx.number_of_nodes( G )
    results = [ ]
    cost = [ ]
    for k in range( int( 2 * K * np.log( N ) ) + 1 ):
        labels_pred, dendogram = _linkagePlusPlus( G, K )
        results.append( ( labels_pred, dendogram ) )
        
        T = auxilliary.fromDendogramToTree( dendogram, K )
        communities = dict( )
        for k in range( K ):
            communities[ k ] = [ i for i in range( N ) if labels_pred[ i ] == k+1 ]
            
        leaves = [ node for node in T.nodes() if T.degree[ node ] == 1 ]
        for leave in leaves:
            T.nodes[ leave ][ 'community' ] = communities[ T.nodes[leave][ 'community' ] ]
        
        cost.append( auxilliary.dasguptaCost(G, T) )
    
    return results[ np.argmin( cost ) ][ 0] 



