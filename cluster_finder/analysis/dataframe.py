"""
Dataframe creation and manipulation functions for cluster_finder package.

This module contains functions for creating and manipulating dataframes from cluster data.
"""

import pandas as pd
import os
import ast
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from ..core.structure import generate_lattice_with_clusters, generate_supercell
from ..analysis.postprocess import get_point_group_order, get_space_group_order, classify_dimensionality

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
    
    records = []
    for _, row in df.iterrows():
        material_id = row['material_id']
        formula = row['formula']
        magnetization = row['magnetization']
        num_clusters = row['num_clusters']
        
        # Handle both string and list inputs for cluster_sizes and average_distance
        cluster_sizes = row['cluster_sizes']
        average_distance = row['average_distance']
        if isinstance(cluster_sizes, str):
            cluster_sizes = ast.literal_eval(cluster_sizes)
        if isinstance(average_distance, str):
            average_distance = ast.literal_eval(average_distance)
        
        # Handle both string and list inputs for structure and cluster_sites
        structure_data = row['structure']
        if isinstance(structure_data, str):
            structure = Structure.from_str(structure_data, fmt="json")
        else:
            structure = structure_data
        
        cluster_sites_data = row['cluster_sites']
        if isinstance(cluster_sites_data, str):
            cluster_sites = ast.literal_eval(cluster_sites_data)
        else:
            cluster_sites = cluster_sites_data
        
        clusters = []
        for i, (size, avg_dist, sites) in enumerate(zip(cluster_sizes, average_distance, cluster_sites)):
            sites_objects = [PeriodicSite.from_dict(site) for site in sites]
            clusters.append({
                'size': size,
                'average_distance': avg_dist,
                'sites': sites_objects,
                'label': f'X{i}'  # Add a unique label for each cluster
            })
        conventional_structure, space_group, point_groups = generate_lattice_with_clusters(structure, clusters)
        supercell_structure = generate_supercell(conventional_structure, (20, 20, 20))
        predicted_dimentionality, norm_svals = classify_dimensionality(supercell_structure)
        record = {
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
            "cluster_sites": cluster_sites,
        }
        records.append(record)
    new_df = pd.DataFrame(records)
    return new_df