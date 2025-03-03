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


