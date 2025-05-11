"""
Dataframe creation and manipulation functions for cluster_finder package.

This module contains functions for creating and manipulating dataframes from cluster data.
"""

import pandas as pd
import os
import ast
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from ..core.structure import generate_lattice_with_clusters, generate_supercell
from ..analysis.postprocess import get_point_group_order, get_space_group_order, classify_dimensionality

def cluster_compounds_dataframe(compounds_with_clusters, compound_system=None, verbose=False):
    """
    Create a dataframe with cluster information for all compounds.
    
    Parameters:
        compounds_with_clusters (list): List of compound dictionaries with clusters
        compound_system (str, optional): Name of the compound system
        verbose (bool, optional): If True, include all unique keys from compounds as columns
        
    Returns:
        pandas.DataFrame: Dataframe with cluster information
    """
    records = []

    for compound in compounds_with_clusters:
        material_id = compound.get("material_id")
        formula = compound.get("formula")
        magnetization = compound.get("total_magnetization")  # sometimes stored as total_magnetization key
        structure = compound.get("structure")

        # Convert structure to dictionary if it's a Structure object
        if isinstance(structure, Structure):
            structure = structure.as_dict()

        # Get clusters or use empty list if none found
        clusters = compound.get("clusters", [])
        num_clusters = len(clusters)
        cluster_sizes = [cluster.get("size") for cluster in clusters] if clusters else []
        avg_distances = [cluster.get("average_distance") for cluster in clusters] if clusters else []
        cluster_sites = [[site.as_dict() for site in cluster.get("sites")] for cluster in clusters] if clusters else []

        # Create basic record with default columns
        record = {
            "compound_system": compound_system,
            "material_id": material_id,
            "formula": formula,
            "magnetization": magnetization,
            "num_clusters": num_clusters,
            "cluster_sizes": cluster_sizes,
            "average_distance": avg_distances,
            "cluster_sites": cluster_sites,
            "structure": structure,  # Already a dictionary
        }
        
        # If verbose, add all unique keys from the compound
        if verbose:
            for key, value in compound.items():
                if key not in record:  # Only add if not already in record
                    # Handle objects that need conversion
                    if isinstance(value, Structure):
                        value = value.as_dict()
                    record[key] = value
        
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
    
    # Vectorized string-to-literal conversion for list columns
    list_columns = ["cluster_sizes", "average_distance"]
    for col in list_columns:
        if col in df.columns:
            # Use pandas.Series.apply with vectorized conditional
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    # Process structure data in a more efficient way
    def process_structure(structure_data):
        if isinstance(structure_data, str):
            try:
                return ast.literal_eval(structure_data)
            except:
                return Structure.from_str(structure_data, fmt="json")
        return structure_data
    
    # Process structure column
    if "structure" in df.columns:
        df["structure_processed"] = df["structure"].apply(process_structure)
        
    # Process cluster_sites data
    if "cluster_sites" in df.columns:
        df["cluster_sites_processed"] = df["cluster_sites"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    # Unfortunately, the subsequent operations with generating lattices and classifying
    # dimensionality need to be done per-row since they involve complex PyMatGen operations.
    # Here we'll still use a row-wise operation but optimize the inner work.
    records = []
    
    for _, row in df.iterrows():
        # Get basic data
        compound_system = row.get('compound_system')
        material_id = row.get('material_id')
        formula = row.get('formula')
        magnetization = row.get('magnetization')
        num_clusters = row.get('num_clusters')
        
        # Get processed data
        cluster_sizes = row.get('cluster_sizes')
        average_distance = row.get('average_distance')
        structure_data = row.get('structure_processed', row.get('structure'))
        cluster_sites_data = row.get('cluster_sites_processed', row.get('cluster_sites'))
        
        # Convert structure to Structure object if it's a dict
        if isinstance(structure_data, dict):
            structure = Structure.from_dict(structure_data)
        else:
            structure = structure_data
        
        # Generate clusters with optimized site creation
        clusters = []
        for i, (size, avg_dist, sites) in enumerate(zip(cluster_sizes, average_distance, cluster_sites_data)):
            # Create all site objects at once
            sites_objects = [PeriodicSite.from_dict(site) for site in sites]
            
            clusters.append({
                'size': size,
                'average_distance': avg_dist,
                'sites': sites_objects,
                'label': f'X{i}'  # Add a unique label for each cluster
            })
        
        # These operations can't be easily vectorized due to PyMatGen dependencies
        conventional_structure, space_group, point_groups = generate_lattice_with_clusters(structure, clusters)
        supercell_structure = generate_supercell(conventional_structure, (20, 20, 20))
        predicted_dimentionality, norm_svals = classify_dimensionality(supercell_structure)
        
        record = {
            "compound_system": compound_system,
            "material_id": material_id,
            "formula": formula,
            "magnetization": magnetization,
            "num_clusters": num_clusters,
            "cluster_sizes": cluster_sizes,
            "average_distance": average_distance,
            "space_group": space_group,
            "point_groups": point_groups,
            "predicted_dimentionality": predicted_dimentionality,
            "norm_svals": norm_svals,
            "conventional_cluster_lattice": conventional_structure.to(fmt="json"),
            "cluster_sites": cluster_sites_data,
        }
        records.append(record)
    
    # Drop temporary columns
    if "structure_processed" in df.columns:
        df = df.drop(columns=["structure_processed"])
    if "cluster_sites_processed" in df.columns:
        df = df.drop(columns=["cluster_sites_processed"])
    
    new_df = pd.DataFrame(records)
    return new_df