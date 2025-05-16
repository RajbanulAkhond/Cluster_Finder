"""
Helper functions for cluster_finder package.

This module contains utility functions used across the package.
"""

import numpy as np
import re
import requests
import os
import sys
import signal
import subprocess
from typing import List, Dict, Any, Optional, Union
from mp_api.client import MPRester

# Import our async utilities
from cluster_finder.utils.async_utils import get_properties_batch, search_compounds_batch, get_api_key


def calculate_metal_distances(metals, api_key=None):
    """
    Calculate nearest-neighbor distances in pure elemental metals.
    
    This function retrieves structures of pure elemental metals from the 
    Materials Project database and calculates the nearest-neighbor distances.
    
    Parameters:
        metals (list): List of metal element symbols
        api_key (str, optional): Materials Project API key or None to use environment variable
    
    Returns:
        dict: Dictionary mapping metal symbols to nearest-neighbor distances
    """
    # Get API key from argument or environment variable
    api_key = get_api_key(api_key)
    
    from pymatgen.core.periodic_table import Element
    
    # Ensure metals are valid elements
    valid_metals = []
    metal_ids = []
    metal_to_id = {}
    
    # Get material IDs for each metal
    for metal in metals:
        try:
            element = Element(metal)
            valid_metals.append(metal)
            
            # Material ID pattern for pure elements: mp-xxx
            # We'll look up the actual IDs from the API
            metal_to_id[metal] = None
        except:
            print(f"Warning: {metal} is not a valid element symbol")
    
    # Get element material IDs using batch request
    print(f"Retrieving data for {len(valid_metals)} metals using batch requests...")
    
    results = {}
    try:
        with MPRester(api_key) as mpr:
            # Query materials with exact element compositions
            for metal in valid_metals:
                # For pure elements, we search by formula
                docs = mpr.summary.search(
                    chemsys=[metal],
                    fields=["material_id", "structure"]
                )
                
                if docs and len(docs) > 0:
                    # Get the most stable structure (usually the first result)
                    doc = docs[0]
                    metal_to_id[metal] = doc.material_id
                    
                    # Calculate the nearest-neighbor distance
                    structure = doc.structure
                    metal_sites = [i for i, site in enumerate(structure) if site.species_string == metal]
                    
                    if len(metal_sites) > 1:
                        min_dist = float('inf')
                        for i in range(len(metal_sites)):
                            for j in range(i+1, len(metal_sites)):
                                dist = structure.get_distance(metal_sites[i], metal_sites[j])
                                min_dist = min(min_dist, dist)
                        
                        results[metal] = min_dist
                    else:
                        results[metal] = None
                        print(f"Warning: Could not find multiple {metal} sites in structure")
                else:
                    results[metal] = None
                    print(f"Warning: No structures found for pure {metal}")
    except Exception as e:
        print(f"Error retrieving metal structures: {str(e)}")
    
    return results


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
                                      min_magnetization=0.001, max_magnetization=3, include_fields=None):
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
    # Use the batched version from async_utils
    return search_compounds_batch(
        elements=elements,
        api_key=api_key,
        min_elements=min_elements,
        max_elements=max_elements,
        min_magnetization=min_magnetization,
        max_magnetization=max_magnetization,
        include_fields=include_fields
    )


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
    # Get API key from argument or environment variable
    api_key = get_api_key(api_key)
    
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
        # Try to add hyphen if missing
        if clean_material_id.startswith('mp') and not clean_material_id.startswith('mp-'):
            clean_material_id = f"mp-{clean_material_id.lstrip('mp')}"
    
    # Use the batched property retrieval for a single material
    results = get_properties_batch(
        material_ids=[clean_material_id],
        properties=[property_name],
        api_key=api_key
    )
    
    # Check if we got results
    if clean_material_id in results and property_name in results[clean_material_id]:
        return results[clean_material_id][property_name]
    
    # If we couldn't get the property via the batch method, fall back to the original implementation
    print(f"Attempting to retrieve {property_name} for {clean_material_id} using fallback methods")
    
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


def get_mp_properties_batch(material_ids: List[str], properties: List[str], api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve multiple properties for multiple materials in a single batch request.
    
    Parameters:
        material_ids (List[str]): List of Materials Project IDs (e.g., ['mp-149', 'mp-13'])
        properties (List[str]): List of property names to retrieve
        api_key (str, optional): Materials Project API key. If None, will use the API key
                                set in the MAPI_KEY environment variable.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping material IDs to dictionaries of properties
        
    Example:
        >>> properties = get_mp_properties_batch(['mp-149', 'mp-13'], ['energy_above_hull', 'band_gap'])
        >>> print(properties['mp-149']['band_gap'])
    """
    # Get API key from argument or environment variable
    api_key = get_api_key(api_key)
    
    # Clean up material IDs
    clean_material_ids = []
    
    for mid in material_ids:
        if mid is None:
            continue
            
        mp_id_match = re.search(r'(mp-\d+)', str(mid))
        if mp_id_match:
            clean_material_ids.append(mp_id_match.group(1))
        else:
            # If no mp-XXXXX pattern found, use as is, but remove any trailing text
            clean_mid = str(mid).split('.')[0].strip()
            # Try to add hyphen if missing
            if clean_mid.startswith('mp') and not clean_mid.startswith('mp-'):
                clean_mid = f"mp-{clean_mid.lstrip('mp')}"
            clean_material_ids.append(clean_mid)
    
    # Use the batch property retrieval function
    return get_properties_batch(clean_material_ids, properties, api_key)


def kill_resource_tracker():
    """
    Clean up multiprocessing resources including loky backend temporary files.
    This prevents resource tracker warnings about leaked semaphore objects.
    """
    import sys
    import os
    import glob
    import tempfile
    import shutil
    import multiprocessing

    try:
        # Clean up loky backend temporary files
        tmp_pattern = os.path.join(tempfile.gettempdir(), "loky-*")
        for folder in glob.glob(tmp_pattern):
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder, ignore_errors=True)
            except:
                pass

        # Python version-dependent cleanup
        if sys.version_info >= (3, 8):
            try:
                # Force cleanup of resource tracker process
                from multiprocessing import resource_tracker
                if hasattr(resource_tracker, "_resource_tracker"):
                    resource_tracker._resource_tracker = None
            except:
                pass

        # Force cleanup of any remaining semaphores
        if hasattr(multiprocessing, "_cleanup"):
            multiprocessing._cleanup()

    except:
        pass  # Silently ignore any cleanup errors