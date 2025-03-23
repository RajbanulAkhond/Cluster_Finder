"""
Helper functions for cluster_finder package.

This module contains utility functions used across the package.
"""

import numpy as np
import re
import requests
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


def search_transition_metal_compounds(elements, api_key, min_elements=2, max_elements=4, 
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
            elements=elements,
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
        
    Raises:
        ValueError: If the API request fails or the property cannot be retrieved
    """
    # Clean up material_id to ensure it's in the correct format
    if material_id is None:
        raise ValueError("Material ID cannot be None")
        
    # Extract the standard MP ID format (mp-XXXXX)
    mp_id_match = re.search(r'(mp-\d+)', str(material_id))
    if mp_id_match:
        clean_material_id = mp_id_match.group(1)
    else:
        # If no mp-XXXXX pattern found, use as is, but remove any trailing text
        clean_material_id = str(material_id).split('.')[0].strip()
    
    print(f"Attempting to retrieve {property_name} for {clean_material_id}")
    
    # Using the MP API v0.41.1 compatible methods
    try:
        with MPRester(api_key) as mpr:
            # First try the new summary API method
            try:
                docs = mpr.summary.search(material_ids=[clean_material_id])
                if docs and len(docs) > 0:
                    for doc in docs:
                        if hasattr(doc, property_name):
                            return getattr(doc, property_name)
                    print(f"Property {property_name} not found in summary")
            except Exception as e1:
                print(f"Error with summary API call: {e1}")
                
            # Then try the materials API method
            try:
                docs = mpr.materials.search(material_ids=[clean_material_id])
                if docs and len(docs) > 0:
                    for doc in docs:
                        if hasattr(doc, property_name):
                            return getattr(doc, property_name)
                    print(f"Property {property_name} not found in materials")
            except Exception as e2:
                print(f"Error with materials API call: {e2}")
                
            # Try the direct documents endpoint as a fallback
            try:
                # Try the direct query method for the property
                results = mpr.thermo.search(material_ids=[clean_material_id])
                if results and len(results) > 0:
                    for result in results:
                        if hasattr(result, property_name):
                            return getattr(result, property_name)
                    print(f"Property {property_name} not found in thermo")
            except Exception as e3:
                print(f"Error with thermo API call: {e3}")
                
            # One more fallback for older API versions
            try:
                # Use the older get_data method
                data = mpr.get_data(clean_material_id)
                if data and len(data) > 0:
                    if property_name in data[0]:
                        return data[0][property_name]
                    print(f"Property {property_name} not found in get_data")
            except Exception as e4:
                print(f"Error with get_data call: {e4}")
                
    except Exception as e:
        print(f"Error initializing MPRester: {e}")
    
    # Direct HTTP request as a last resort
    try:
        print("Trying direct HTTP request...")
        headers = {"X-API-KEY": api_key}
        
        # Try the v2 API endpoint
        v2_url = f"https://api.materialsproject.org/materials/{clean_material_id}"
        response = requests.get(v2_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            # Try to find the property in different locations in the response
            if property_name in data:
                return data[property_name]
            elif "data" in data and property_name in data["data"]:
                return data["data"][property_name]
            elif "data" in data and len(data["data"]) > 0 and property_name in data["data"][0]:
                return data["data"][0][property_name]
            print(f"Property {property_name} not found in HTTP response")
        else:
            print(f"HTTP request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error with direct HTTP request: {e}")
        
    # If all attempts fail, raise an error
    raise ValueError(f"Could not retrieve {property_name} for material {material_id}")