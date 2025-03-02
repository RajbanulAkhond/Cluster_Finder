"""
Analysis functions for cluster_finder package.

This module contains functions for analyzing, classifying, and ranking clusters.
"""

import numpy as np
import pandas as pd
import ast
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ..core.structure import calculate_centroid

# Define mappings for symmetry analysis
# Point group order mapping (simplified for common point groups)
point_group_order_mapping = {
    "1": 1, "-1": 2, "2": 2, "m": 2, "2/m": 4,
    "222": 4, "mm2": 4, "mmm": 8,
    "4": 4, "-4": 4, "4/m": 8, "422": 8, "4mm": 8, "-42m": 8, "4/mmm": 16,
    "3": 3, "-3": 6, "32": 6, "3m": 6, "-3m": 12,
    "6": 6, "-6": 6, "6/m": 12, "622": 12, "6mm": 12, "-62m": 12, "6/mmm": 24,
    "23": 12, "m-3": 24, "432": 24, "-43m": 24, "m-3m": 48
}

# Space group number mapping
space_group_number_mapping = {f"P{i}": i for i in range(1, 231)}
space_group_number_mapping.update({f"I{i}": i for i in range(1, 231)})
space_group_number_mapping.update({f"F{i}": i for i in range(1, 231)})
space_group_number_mapping.update({f"A{i}": i for i in range(1, 231)})
space_group_number_mapping.update({f"B{i}": i for i in range(1, 231)})
space_group_number_mapping.update({f"C{i}": i for i in range(1, 231)})
space_group_number_mapping.update({f"R{i}": i for i in range(1, 231)})


def get_point_group_order(point_group_symbol):
    """
    Get the order of a point group using the predefined mapping.
    
    Parameters:
        point_group_symbol (str): Point group symbol
        
    Returns:
        int: Order of the point group
    """
    return point_group_order_mapping.get(point_group_symbol, 0)


def get_space_group_order(space_group_symbol):
    """
    Get the order of a space group using the predefined mapping.
    
    Parameters:
        space_group_symbol (str): Space group symbol
        
    Returns:
        int: Order of the space group
    """
    return space_group_number_mapping.get(space_group_symbol, 0)


def classify_dimensionality(structure, distance_cutoff=3.5):
    """
    Classify the dimensionality of a structure based on connectivity.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        distance_cutoff (float): Cutoff distance for connectivity
        
    Returns:
        str: Dimensionality classification ('0D', '1D', '2D', '3D')
    """
    # Create connectivity matrix
    num_sites = len(structure)
    connectivity = np.zeros((num_sites, num_sites), dtype=bool)
    
    # Populate connectivity matrix
    for i in range(num_sites):
        for j in range(i+1, num_sites):
            if structure[i].distance(structure[j]) <= distance_cutoff:
                connectivity[i, j] = connectivity[j, i] = True
    
    # Check periodicity along each lattice vector
    dim_count = 0
    
    # Check each lattice direction
    for dim in range(3):
        # Create translation vector in this dimension
        translation = np.zeros(3)
        translation[dim] = 1.0
        
        # Check if any site connects to its periodic image
        for i in range(num_sites):
            site_i = structure[i]
            # Create fractional coordinates for translated site
            frac_coords = site_i.frac_coords + translation
            
            # Find if it connects to any site in the actual structure
            for j in range(num_sites):
                site_j = structure[j]
                # Check distance between site_i and periodic image of site_j
                if site_i.distance_from_point(structure.lattice.get_cartesian_coords(frac_coords)) <= distance_cutoff:
                    dim_count += 1
                    break
            
            # If we found connectivity in this dimension, move to next dimension
            if dim_count > dim:
                break
    
    # Return dimensionality classification
    if dim_count == 0:
        return "0D"
    elif dim_count == 1:
        return "1D"
    elif dim_count == 2:
        return "2D"
    else:
        return "3D"


def cluster_compounds_dataframe(compounds_with_clusters, compound_system=None):
    """
    Create a dataframe with cluster information for all compounds.
    
    Parameters:
        compounds_with_clusters (list): List of compound dictionaries with clusters
        compound_system (str, optional): Name of the compound system
        
    Returns:
        pandas.DataFrame: Dataframe with cluster information
    """
    records = []
    
    for compound in compounds_with_clusters:
        if not compound["clusters"]:
            continue  # Skip compounds without clusters
            
        material_id = compound["material_id"]
        formula = compound["formula"]
        total_magnetization = compound.get("total_magnetization", 0)
        structure = compound["structure"]
        
        # Analyze space group
        try:
            analyzer = SpacegroupAnalyzer(structure)
            space_group = analyzer.get_space_group_symbol()
            point_group = analyzer.get_point_group_symbol()
        except Exception:
            space_group = "Symmetry Not Determined"
            point_group = "Symmetry Not Determined"
            
        # Group clusters by size
        clusters_by_size = {}
        for cluster in compound["clusters"]:
            size = cluster["size"]
            if size not in clusters_by_size:
                clusters_by_size[size] = []
            clusters_by_size[size].append(cluster)
            
        # Create record
        record = {
            "material_id": material_id,
            "formula": formula,
            "total_magnetization": total_magnetization,
            "space_group": space_group,
            "point_group": point_group,
            "cluster_sizes": list(clusters_by_size.keys()),
            "avg_cluster_size": sum(clusters_by_size.keys()) / len(clusters_by_size) if clusters_by_size else 0,
            "num_clusters": len(compound["clusters"]),
            "average_distance": [cluster["average_distance"] for cluster in compound["clusters"]],
            "point_groups": {}, # Will be filled by symmetry analysis later
            "dimensionality": classify_dimensionality(structure),
            "compound_system": compound_system,
        }
        
        records.append(record)
        
    # Create DataFrame
    df = pd.DataFrame(records)
    return df


def rank_clusters(df):
    """
    Rank clusters based on average distance, point group order, and space group order.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing cluster information
        
    Returns:
        pandas.DataFrame: Sorted DataFrame with ranking information
    """
    # Filter out rows where symmetry was not determined or cluster size > 6
    df_filtered = df[
        (df["space_group"] != "Symmetry Not Determined") &
        (df["cluster_sizes"].apply(lambda x: all(int(size) <= 6 for size in ast.literal_eval(x) if isinstance(x, str) else x)))
    ].copy()
    
    # Process average_distance column
    if "average_distance" in df_filtered.columns:
        # Convert string representation to list if needed
        if df_filtered["average_distance"].dtype == object:
            df_filtered["average_distance"] = df_filtered["average_distance"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        # Create a new column for the minimum average distance
        df_filtered["min_avg_distance"] = df_filtered["average_distance"].apply(
            lambda x: min(x) if isinstance(x, list) and len(x) > 0 else None
        )
    
    # Create new column for point group ranking
    df_filtered["point_group_order"] = df_filtered["point_group"].apply(
        lambda pg: get_point_group_order(pg) if pg != "Symmetry Not Determined" else 0
    )
    
    # Create new column for space group ranking
    df_filtered["space_group_order"] = df_filtered["space_group"].apply(
        lambda sg: get_space_group_order(sg) if sg != "Symmetry Not Determined" else 0
    )
    
    # Sort the dataframe
    df_sorted = df_filtered.sort_values(
        by=["min_avg_distance", "point_group_order", "space_group_order"],
        ascending=[True, False, False]
    )
    
    return df_sorted 