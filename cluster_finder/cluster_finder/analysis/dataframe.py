"""
Dataframe creation and manipulation functions for cluster_finder package.

This module contains functions for creating and manipulating dataframes from cluster data.
"""

import pandas as pd

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