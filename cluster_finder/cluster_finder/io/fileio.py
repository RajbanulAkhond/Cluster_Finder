"""
I/O functions for cluster_finder package.

This module contains functions for importing and exporting structures and data.
"""

import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import os


def export_structure_to_cif(structure, filename):
    """Export structure to CIF file.
    
    Args:
        structure (Structure): Pymatgen structure object
        filename (str): Output filename
        
    Returns:
        str: Path to the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Write structure to CIF file without symmetry analysis
    writer = CifWriter(structure, symprec=None)
    writer.write_file(filename)
    
    return filename


def generate_lattice_with_clusters(structure, clusters, tolerance=1e-5):
    """
    Generate a structure with cluster centroids as sites.
    
    Parameters:
        structure (Structure): Original pymatgen Structure
        clusters (list): List of cluster dictionaries
        tolerance (float): Tolerance for fractional coordinate comparison
        
    Returns:
        Structure: New structure with cluster centroids as sites
    """
    # Get original lattice
    lattice = structure.lattice
    
    # Extract centroids
    centroids = [cluster["centroid"] for cluster in clusters]
    
    # Convert centroids to fractional coordinates
    frac_coords = [lattice.get_fractional_coords(centroid) for centroid in centroids]
    
    # Remove duplicates (due to periodic boundary conditions)
    unique_frac_coords = []
    unique_indices = []
    
    for i, coord in enumerate(frac_coords):
        # Normalize fractional coordinates to [0, 1)
        normalized_coord = np.mod(coord, 1.0)
        
        # Check if this is a duplicate
        is_duplicate = False
        for existing_coord in unique_frac_coords:
            if np.allclose(normalized_coord, existing_coord, atol=tolerance):
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_frac_coords.append(normalized_coord)
            unique_indices.append(i)
    
    # Create a new structure with centroids as dummy atoms
    cluster_structure = Structure(
        lattice=lattice,
        species=["X"] * len(unique_frac_coords),  # Use "X" as dummy element
        coords=unique_frac_coords,
        coords_are_cartesian=False
    )
    
    return cluster_structure


def import_csv_data(filename):
    """
    Import data from a CSV file.
    
    Parameters:
        filename (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the data
    """
    return pd.read_csv(filename)


def export_csv_data(df, filename):
    """
    Export data to a CSV file.
    
    Parameters:
        df (pandas.DataFrame): DataFrame to export
        filename (str): Output filename
        
    Returns:
        str: Path to the saved file
    """
    df.to_csv(filename, index=False)
    return filename


def postprocess_clusters(csv_filename):
    """
    Post-process cluster data from a CSV file for further analysis.
    
    Parameters:
        csv_filename (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    # Load data
    df = pd.read_csv(csv_filename)
    
    # Process cluster_sizes column
    if 'cluster_sizes' in df.columns:
        df['cluster_sizes'] = df['cluster_sizes'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Process average_distance column
    if 'average_distance' in df.columns:
        df['average_distance'] = df['average_distance'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Process point_groups column if it exists
    if 'point_groups' in df.columns and df['point_groups'].dtype == object:
        df['point_groups'] = df['point_groups'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Calculate derived statistics
    if 'cluster_sizes' in df.columns:
        df['max_cluster_size'] = df['cluster_sizes'].apply(lambda x: max(x) if x else 0)
        df['min_cluster_size'] = df['cluster_sizes'].apply(lambda x: min(x) if x else 0)
        df['has_small_clusters'] = df['min_cluster_size'].apply(lambda x: x <= 3)
        df['has_large_clusters'] = df['max_cluster_size'].apply(lambda x: x >= 4)
    
    # Calculate minimum average distance
    if 'average_distance' in df.columns:
        df['min_avg_distance'] = df['average_distance'].apply(
            lambda x: min(x) if isinstance(x, list) and len(x) > 0 else None
        )
    
    return df 