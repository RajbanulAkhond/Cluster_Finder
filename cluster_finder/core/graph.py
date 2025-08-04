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


def create_connectivity_matrix(structure, transition_metals, cutoff=3.0):
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
    
    # Use vectorized calculation for efficiency
    if n > 0:
        # Extract coordinates for all transition metal sites
        coords = np.array([structure[idx].coords for idx in tm_indices])
        
        # Get all pairs of indices (upper triangle of matrix)
        i, j = np.triu_indices(n, k=1)
        
        # Calculate distances for all pairs at once
        site_i_coords = coords[i]
        site_j_coords = coords[j]
        distances = np.linalg.norm(site_i_coords - site_j_coords, axis=1)
        
        # Find connections based on cutoff
        connections = distances <= cutoff
        
        # Set matrix values for connections
        matrix[i[connections], j[connections]] = 1
        matrix[j[connections], i[connections]] = 1  # Mirror the matrix
                
    return matrix, tm_indices