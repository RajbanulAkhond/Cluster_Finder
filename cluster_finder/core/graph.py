"""
Graph-related functions for cluster_finder package.

This module contains functions for creating and manipulating graphs.
"""

import numpy as np
import networkx as nx

def structure_to_graph(connectivity_matrix):
    """
    Convert a connectivity matrix to a networkx graph.
    
    Parameters:
        connectivity_matrix (numpy.ndarray): Connectivity matrix
        
    Returns:
        networkx.Graph: Graph representation of connectivity
    """
    G = nx.Graph()
    for i in range(len(connectivity_matrix)):
        for j in range(len(connectivity_matrix)):
            if connectivity_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G


def create_connectivity_matrix(structure, transition_metals, cutoff=3.5):
    """
    Create a connectivity matrix for transition metal atoms in a structure.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        transition_metals (list): List of transition metal element symbols
        cutoff (float): Maximum distance for connectivity
        
    Returns:
        tuple: (connectivity matrix, transition metal indices)
    """
    tm_indices = [i for i, site in enumerate(structure) 
                 if hasattr(site.specie, 'symbol') and site.specie.symbol in transition_metals]
    
    n = len(tm_indices)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i+1, n):
            if structure[tm_indices[i]].distance(structure[tm_indices[j]]) <= cutoff:
                matrix[i, j] = matrix[j, i] = 1
                
    return matrix, tm_indices 