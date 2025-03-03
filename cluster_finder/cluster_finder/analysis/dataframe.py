"""
Dataframe creation and manipulation functions for cluster_finder package.

This module contains functions for creating and manipulating dataframes from cluster data.
"""

import pandas as pd
import os
from ..analysis.postprocess import get_point_group_order, get_space_group_order

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
        # Skip compounds with no clusters
        clusters = compound.get("clusters", [])
        if not clusters:
            continue

        material_id = compound.get("material_id")
        formula = compound.get("formula")
        magnetization = compound.get("total_magnetization")  # sometimes stored as total_magnetization key
        # For the pymatgen structure, we assume there's a 'structure' key or it can be added later.
        # In the main workflow, structure may be obtained separately; adjust accordingly.
        structure = compound.get("structure", None)

        compound_system = compound_system
        num_clusters = len(clusters)
        cluster_sizes = [cluster.get("size") for cluster in clusters]
        avg_distances = [cluster.get("average_distance") for cluster in clusters]
        cluster_sites = [[site.as_dict() for site in cluster.get("sites")] for cluster in clusters]

        record = {
            "compound_system": compound_system,
            "material_id": material_id,
            "formula": formula,
            "magnetization": magnetization,
            "num_clusters": num_clusters,
            "cluster_sizes": cluster_sizes,
            "average_distance": avg_distances,
            "cluster_sites": cluster_sites,
            "structure": structure.to(fmt="json"),  # pymatgen structure for this compound
        }
        records.append(record)

    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(records)
    return df 

def postprocessed_clusters_dataframe(data_source):
    """
    Post-process cluster data from a CSV file or pandas DataFrame.
    
    This function:
    1. Reads the data (if a CSV file path is provided)
    2. Converts string representations of lists to actual lists
    3. Calculates derived properties like min_avg_distance, point_group_order, and space_group_order
    
    Parameters:
        data_source (str or pandas.DataFrame): Path to a CSV file or a pandas DataFrame
        
    Returns:
        pandas.DataFrame: Processed DataFrame with additional calculated columns
        
    Examples:
        >>> # From a CSV file
        >>> df = postprocessed_clusters_dataframe("path/to/clusters.csv")
        >>> 
        >>> # From an existing DataFrame
        >>> raw_df = pd.read_csv("path/to/clusters.csv")
        >>> processed_df = postprocessed_clusters_dataframe(raw_df)
    """
    # Handle input: either read from CSV or use provided DataFrame
    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()  # Create a copy to avoid modifying the original
    else:
        raise TypeError("data_source must be either a file path (str) or a pandas DataFrame")
    
    # Convert string lists to actual lists
    for col in ["cluster_sizes", "average_distance"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Calculate min_avg_distance for each compound
    if "average_distance" in df.columns:
        df["min_avg_distance"] = df["average_distance"].apply(
            lambda x: min(x) if isinstance(x, list) and len(x) > 0 else None
        )
    
    # Calculate point group order
    if "point_group" in df.columns:
        df["point_group_order"] = df["point_group"].apply(get_point_group_order)
    
    # Calculate space group order
    if "space_group" in df.columns:
        df["space_group_order"] = df["space_group"].apply(get_space_group_order)
    
    return df

# For backward compatibility
def postprocess_clusters(csv_filename):
    """
    Legacy function for backward compatibility.
    
    This function is deprecated and will be removed in a future version.
    Please use postprocessed_clusters_dataframe() instead.
    
    Parameters:
        csv_filename (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    import warnings
    warnings.warn(
        "postprocess_clusters is deprecated and will be removed in a future version. "
        "Please use postprocessed_clusters_dataframe instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return postprocessed_clusters_dataframe(csv_filename)