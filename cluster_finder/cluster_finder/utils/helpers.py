"""
Helper functions for cluster_finder package.

This module contains utility functions used across the package.
"""

import numpy as np
from mp_api.client import MPRester


def calculate_metal_distances(metal_list, api_key):
    """
    Calculate the minimum distance between atoms in pure metallic elements.
    
    Parameters:
        metal_list (list): List of metal element symbols
        api_key (str): Materials Project API key
        
    Returns:
        dict: Dictionary of metals and their minimum interatomic distances
    """
    distances = {}
    with MPRester(api_key) as mpr:
        for metal in metal_list:
            results = mpr.materials.summary.search(
                elements=[metal], 
                energy_above_hull=(0, 0), 
                num_elements=(1, 1), 
                fields=["structure"]
            )
            
            if results:
                from pymatgen.core.structure import Structure
                structure = Structure.from_dict(results[0].structure.as_dict())
                metal_sites = list(structure)
                
                # Calculate minimum distance between sites
                if len(metal_sites) > 1:
                    distances[metal] = min(
                        site.distance(other_site) 
                        for i, site in enumerate(metal_sites) 
                        for other_site in metal_sites[i+1:]
                    )
                else:
                    # If only one atom in unit cell, use lattice parameter
                    distances[metal] = structure.lattice.a
            else:
                distances[metal] = None
                
    return distances


def get_transition_metals():
    """
    Get a list of common transition metal elements.
    
    Returns:
        list: List of transition metal element symbols
    """
    return [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"
    ]


def search_transition_metal_compounds(transition_metals, api_key, min_elements=2, max_elements=4, 
                                      min_magnetization=0.001, include_fields=None):
    """
    Search for transition metal compounds meeting specific criteria.
    
    Parameters:
        transition_metals (list): List of transition metal elements to include
        api_key (str): Materials Project API key
        min_elements (int): Minimum number of elements in compound
        max_elements (int): Maximum number of elements in compound
        min_magnetization (float): Minimum magnetization value
        include_fields (list, optional): Additional fields to include in results
        
    Returns:
        list: List of matching entries
    """
    # Define default fields to retrieve
    default_fields = ["material_id", "formula_pretty", "structure", "total_magnetization"]
    
    # Combine with additional fields if provided
    if include_fields:
        fields = list(set(default_fields + include_fields))
    else:
        fields = default_fields
    
    # Initialize the MPRester
    with MPRester(api_key) as mpr:
        entries = mpr.materials.summary.search(
            elements=transition_metals,
            num_elements=(min_elements, max_elements),
            total_magnetization=(min_magnetization, None),
            fields=fields
        )
    
    return entries


def generate_rotation_matrix(axis=None, angle=None):
    """
    Generate a rotation matrix for 3D visualization.
    
    Parameters:
        axis (numpy.ndarray, optional): Rotation axis (normalized)
        angle (float, optional): Rotation angle in radians
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    # Default to a common viewing angle if none provided
    if axis is None:
        axis = np.array([0, 1, 0])  # y-axis
    if angle is None:
        angle = np.pi / 6  # 30 degrees
    
    # Normalize axis if not already normalized
    axis = axis / np.linalg.norm(axis)
    
    # Create rotation matrix (Rodrigues' rotation formula)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = (
        np.eye(3) + 
        np.sin(angle) * K + 
        (1 - np.cos(angle)) * np.dot(K, K)
    )
    
    return rotation_matrix 