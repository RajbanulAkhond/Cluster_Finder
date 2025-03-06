"""
Helper functions for cluster_finder package.

This module contains utility functions used across the package.
"""

import numpy as np
from mp_api.client import MPRester


def calculate_metal_distances(structure, sites):
    """Calculate distances between metal sites.
    
    Args:
        structure (Structure): Pymatgen structure object
        sites (list): List of sites to calculate distances between
        
    Returns:
        list: List of distances between sites
    """
    distances = []
    for i in range(len(sites)):
        for j in range(i+1, len(sites)):
            distance = structure.get_distance(sites[i], sites[j])
            distances.append(distance)
            
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


def find_trimers(structure, sites, max_distance):
    """Find trimers in a list of sites.
    
    Args:
        structure (Structure): Pymatgen structure object
        sites (list): List of sites to search for trimers
        max_distance (float): Maximum distance between sites to be considered connected
        
    Returns:
        list: List of trimers, where each trimer is a list of 3 sites
    """
    trimers = []
    n_sites = len(sites)
    
    # Need at least 3 sites to form a trimer
    if n_sites < 3:
        return trimers
        
    # Check all possible combinations of 3 sites
    for i in range(n_sites):
        for j in range(i+1, n_sites):
            for k in range(j+1, n_sites):
                # Calculate distances between all pairs
                d_ij = structure.get_distance(sites[i], sites[j])
                d_jk = structure.get_distance(sites[j], sites[k])
                d_ik = structure.get_distance(sites[i], sites[k])
                
                # Check if all distances are within max_distance
                if all(d <= max_distance for d in [d_ij, d_jk, d_ik]):
                    trimers.append([sites[i], sites[j], sites[k]])
                    
    return trimers


def search_and_analyze_trimers(structure, elements, max_distance):
    """Search for and analyze trimers in a structure.
    
    Args:
        structure (Structure): Pymatgen structure object
        elements (list): List of elements to consider
        max_distance (float): Maximum distance between sites to be considered connected
        
    Returns:
        dict: Dictionary containing formula and trimer information
    """
    # Get metal sites
    metal_sites = [i for i, site in enumerate(structure) 
                  if str(site.specie) in elements]
    
    # Find trimers
    trimers = find_trimers(structure, metal_sites, max_distance)
    
    # Analyze trimers
    trimer_info = []
    for trimer in trimers:
        # Calculate average distance
        distances = calculate_metal_distances(structure, trimer)
        avg_distance = sum(distances) / len(distances)
        
        # Get elements
        elements = [structure[site].specie.symbol for site in trimer]
        
        trimer_info.append({
            "sites": trimer,
            "average_distance": avg_distance,
            "elements": elements
        })
    
    return {
        "formula": structure.composition.reduced_formula,
        "trimers": trimer_info
    }


def get_mp_property(material_id, property_name, api_key=None):
    """
    Retrieve a specific property for a material from the Materials Project database.
    
    Parameters:
        material_id (str): Materials Project ID (e.g., 'mp-149')
        property_name (str): Name of the property to retrieve (e.g., 'energy_above_hull')
        api_key (str, optional): Materials Project API key. If None, will use the API key
                                 set in the MAPI_KEY environment variable.
    
    Returns:
        The requested property value or None if the property or material is not found.
    
    Examples:
        >>> e_hull = get_mp_property('mp-149', 'energy_above_hull')
        >>> formation_energy = get_mp_property('mp-149', 'formation_energy_per_atom')
    """
    try:
        with MPRester(api_key) as mpr:
            summary = mpr.materials.summary.search(
                material_ids=[material_id],
                fields=[property_name]
            )
            
            if not summary or len(summary) == 0:
                return None
                
            # Get the first (and should be only) result
            return getattr(summary[0], property_name, None)
    except Exception as e:
        print(f"Error retrieving property from Materials Project: {e}")
        return None 