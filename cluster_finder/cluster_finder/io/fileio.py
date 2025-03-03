"""
I/O functions for cluster_finder package.

This module contains functions for reading and writing files.
"""

import os
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

# Import at the top level to avoid circular imports
from ..analysis.postprocess import get_point_group_order, get_space_group_order

def export_structure_to_cif(structure, filename):
    """
    Export a pymatgen Structure to a CIF file.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        filename (str): Path to the output CIF file
        
    Returns:
        str: Path to the output CIF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Write structure to CIF file without symmetry analysis
    # and without magnetic moments to avoid occupancy issues
    writer = CifWriter(structure, symprec=None, write_magmoms=False)
    writer.write_file(filename)
    
    return filename


def import_csv_data(filename):
    """
    Import cluster data from a CSV file.
    
    Parameters:
        filename (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the imported data
    """
    return pd.read_csv(filename)


def export_csv_data(df, filename):
    """
    Export cluster data to a CSV file.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the data to export
        filename (str): Path to the output CSV file
        
    Returns:
        str: Path to the output CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Export to CSV
    df.to_csv(filename, index=False)
    
    return filename


def postprocess_clusters(csv_filename):
    """
    Post-process cluster data from CSV file.
    
    This function:
    1. Reads the CSV file
    2. Converts string representations of lists to actual lists
    3. Calculates derived properties
    
    Args:
        csv_filename (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    # Read CSV file
    df = pd.read_csv(csv_filename)
    
    # Convert string lists to actual lists
    for col in ["cluster_sizes", "average_distance"]:
        if col in df.columns:
            df[col] = df[col].apply(eval)
    
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