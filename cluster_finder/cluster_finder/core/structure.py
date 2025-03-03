"""
Structure related functions for cluster_finder package.

This module contains functions for structure analysis, manipulation, and inspection.
"""

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import networkx as nx

def calculate_centroid(sites, lattice=None):
    """
    Calculate the centroid of a group of sites.
    
    Parameters:
        sites (list): List of pymatgen Site objects
        lattice (Lattice, optional): Lattice object for periodic boundary conditions
        
    Returns:
        numpy.ndarray: Centroid coordinates
    """
    coords = np.array([site.coords for site in sites])
    if lattice is not None:
        # Apply periodic boundary conditions if lattice is provided
        frac_coords = lattice.get_fractional_coords(coords)
        # Wrap to [0, 1)
        frac_coords = frac_coords % 1.0
        # Convert back to cartesian
        coords = lattice.get_cartesian_coords(frac_coords)
    return np.mean(coords, axis=0)

def find_non_equivalent_positions(structure, transition_metals):
    """
    Find non-equivalent transition metal atom positions in a structure.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        transition_metals (list): List of transition metal element symbols
        
    Returns:
        list: Unique transition metal sites
    """
    unique_sites = []
    seen = set()  # Set to store unique (symbol, fractional coordinates) tuples

    for site in structure:
        if hasattr(site.specie, 'symbol') and site.specie.symbol in transition_metals:
            site_key = (site.specie.symbol, tuple(site.frac_coords))
            if site_key not in seen:
                seen.add(site_key)
                unique_sites.append(site)

    return unique_sites


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


def generate_supercell(structure, supercell_matrix=(2, 2, 2)):
    """
    Generate a supercell from a structure.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        supercell_matrix (tuple): Supercell matrix dimensions
        
    Returns:
        Structure: Supercell structure
    """
    return structure.make_supercell(supercell_matrix) 