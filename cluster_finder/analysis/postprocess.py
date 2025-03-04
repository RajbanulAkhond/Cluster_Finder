"""
Postprocessing functions for cluster_finder package.

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

# A comprehensive mapping for space group numbers
space_group_number_mapping = {
    "P1": 1, "P-1": 2, "P2": 3, "P21": 4, "C2": 5, "Pm": 6, "Pc": 7, "Cm": 8, "Cc": 9,
    "P2/m": 10, "P21/m": 11, "C2/m": 12, "P2/c": 13, "P21/c": 14, "C2/c": 15,
    "P222": 16, "P2221": 17, "P21212": 18, "P212121": 19, "C2221": 20, "C222": 21,
    "F222": 22, "I222": 23, "I212121": 24, "Pmm2": 25, "Pmc21": 26, "Pcc2": 27,
    "Pma2": 28, "Pca21": 29, "Pnc2": 30, "Pmn21": 31, "Pba2": 32, "Pna21": 33,
    "Pnn2": 34, "Cmm2": 35, "Cmc21": 36, "Ccc2": 37, "Amm2": 38, "Aem2": 39,
    "Ama2": 40, "Aea2": 41, "Fmm2": 42, "Fdd2": 43, "Imm2": 44, "Iba2": 45,
    "Ima2": 46, "Pmmm": 47, "Pnnn": 48, "Pccm": 49, "Pban": 50, "Pmma": 51,
    "Pnna": 52, "Pmna": 53, "Pcca": 54, "Pbam": 55, "Pccn": 56, "Pbcm": 57,
    "Pnnm": 58, "Pmmn": 59, "Pbcn": 60, "Pbca": 61, "Pnma": 62, "Cmcm": 63,
    "Cmce": 64, "Cmmm": 65, "Cccm": 66, "Cmme": 67, "Ccce": 68, "Fmmm": 69,
    "Fddd": 70, "Immm": 71, "Ibam": 72, "Ibca": 73, "Imma": 74, "P4": 75,
    "P41": 76, "P42": 77, "P43": 78, "I4": 79, "I41": 80, "P-4": 81, "I-4": 82,
    "P4/m": 83, "P42/m": 84, "P4/n": 85, "P42/n": 86, "I4/m": 87, "I41/a": 88,
    "P422": 89, "P4212": 90, "P4122": 91, "P41212": 92, "P4222": 93, "P42212": 94,
    "P4322": 95, "P43212": 96, "I422": 97, "I4122": 98, "P4mm": 99, "P4bm": 100,
    "P42cm": 101, "P42nm": 102, "P4cc": 103, "P4nc": 104, "P42mc": 105, "P42bc": 106,
    "I4mm": 107, "I4cm": 108, "I41md": 109, "I41cd": 110, "P-42m": 111, "P-42c": 112,
    "P-421m": 113, "P-421c": 114, "P-4m2": 115, "P-4c2": 116, "P-4b2": 117, "P-4n2": 118,
    "I-4m2": 119, "I-4c2": 120, "I-42m": 121, "I-42d": 122, "P4/mmm": 123, "P4/mcc": 124,
    "P4/nbm": 125, "P4/nnc": 126, "P4/mbm": 127, "P4/mnc": 128, "P4/nmm": 129, "P4/ncc": 130,
    "I4/mmm": 131, "I4/mcm": 132, "I41/amd": 133, "I41/acd": 134, "P3": 143, "P31": 144,
    "P32": 145, "R3": 146, "P-3": 147, "R-3": 148, "P312": 149, "P321": 150, "P3112": 151,
    "P3121": 152, "P3212": 153, "P3221": 154, "R32": 155, "P3m1": 156, "P31m": 157,
    "P3c1": 158, "P31c": 159, "R3m": 160, "R3c": 161, "P-31m": 162, "P-31c": 163,
    "P-3m1": 164, "P-3c1": 165, "R-3m": 166, "R-3c": 167, "P6": 168, "P61": 169,
    "P65": 170, "P62": 171, "P64": 172, "P63": 173, "P-6": 174, "P6/m": 175,
    "P63/m": 176, "P622": 177, "P6122": 178, "P6522": 179, "P6222": 180, "P6422": 181,
    "P6322": 182, "P6mm": 183, "P6cc": 184, "P63cm": 185, "P63mc": 186, "P-6m2": 187,
    "P-6c2": 188, "P-62m": 189, "P-62c": 190, "P6/mmm": 191, "P6/mcc": 192, "P63/mcm": 193,
    "P63/mmc": 194, "P23": 195, "F23": 196, "I23": 197, "P213": 198, "I213": 199,
    "Pm-3": 200, "Pn-3": 201, "Fm-3": 202, "Fd-3": 203, "Im-3": 204, "Pa-3": 205,
    "Ia-3": 206, "P432": 207, "P4232": 208, "F432": 209, "F4132": 210, "I432": 211,
    "P4332": 212, "P4132": 213, "I4132": 214, "P-43m": 215, "F-43m": 216, "I-43m": 217,
    "P-43n": 218, "F-43c": 219, "I-43d": 220, "Pm-3m": 221, "Pn-3n": 222, "Pm-3n": 223,
    "Pn-3m": 224, "Fm-3m": 225, "Fm-3c": 226, "Fd-3m": 227, "Fd-3c": 228, "Im-3m": 229,
    "Ia-3d": 230
}


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


def classify_dimensionality(supercell):
    """
    Classify the effective dimensionality (0D, 1D, 2D, or 3D) of a supercell that
    contains only the centroid points of identified clusters.

    Parameters:
        supercell (Structure): A pymatgen.core.structure.Structure object whose
                               sites are the centroid points of clusters.

    Returns:
        tuple: (cluster_type, normalized_singular_values)
               where cluster_type is one of "0D", "1D", "2D", or "3D",
               and normalized_singular_values is a list of the normalized singular values.

    The classification uses PCA via singular value decomposition. The idea is that:
      - If only one singular value is significant, the centroids are nearly collinear (0D or 1D).
      - If two are significant, they lie mostly in a plane (1D or 2D).
      - If all three are significant, then the network is volumetric (3D).
    Thresholds (e.g. 0.1) and ratios (e.g. 0.5) are used to distinguish these cases.
    """
    # Extract coordinates for all sites in the supercell
    coords = np.array([site.coords for site in supercell.sites])

    # Center the coordinates (subtract the mean)
    centered = coords - coords.mean(axis=0)

    # Perform SVD (equivalent to PCA for centered data)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)

    # Normalize singular values to range 0-1 (avoid division by zero)
    if s.max() != 0:
        s_normalized = s / s.max()
    else:
        s_normalized = s

    # Set a threshold to decide if a singular value is significant
    threshold = 0.2

    # Classify based on normalized singular values.
    # Here we assume that we always have three singular values for a 3D structure.
    if len(s_normalized) >= 3:
        # If both second and third singular values are very low, the points are nearly a single point (0D)
        if s_normalized[1] < threshold and s_normalized[2] < threshold:
            cluster_type = "0D"
        # If only the third singular value is insignificant, points lie mainly along a line (1D)
        elif (s_normalized[0]-s_normalized[1]) <= (threshold*0.1) and s_normalized[2] < 0.5:
            cluster_type = "1D"
        # If both the second and third are significant, decide between 2D and 3D using their ratio
        else:
            if s_normalized[2] / s_normalized[1] < 0.8:
                cluster_type = "2D"
            else:
                cluster_type = "3D"
    elif len(s_normalized) == 2:
        if (s_normalized[0]-s_normalized[1]) <= (threshold*0.1):
            cluster_type = "0D"
        else:
            cluster_type = "1D"
    else:
        cluster_type = "0D"

    return cluster_type, s_normalized.tolist()


def rank_clusters(data_source):
    """
    Rank clusters based on dimensionality, point group order, and space group order.
    
    Parameters:
        data_source (str or pandas.DataFrame): Path to a CSV file or a pandas DataFrame
        
    Returns:
        pandas.DataFrame: Sorted DataFrame with additional ranking columns
    """
    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()  # Create a copy to avoid modifying the original
    else:
        raise TypeError("data_source must be either a file path (str) or a pandas DataFrame")

    # Filter out rows where symmetry was not determined or cluster size > 6
    df_filtered = df[
        (df["space_group"] != "Symmetry Not Determined") &
        (df["cluster_sizes"].apply(lambda x: all(int(size) <= 6 for size in ast.literal_eval(x))))
    ].copy()

    # 'average_distance' column is stored as a list in string form; convert it.
    df_filtered["average_distance"] = df_filtered["average_distance"].apply(ast.literal_eval)
    # Create a new column for the minimum average distance.
    df_filtered["min_avg_distance"] = df_filtered["average_distance"].apply(
        lambda x: min(x) if isinstance(x, list) and len(x) > 0 else None
    )

    # Calculate point group order
    df_filtered["point_group_order"] = df_filtered["point_group"].apply(get_point_group_order)

    # Calculate space group order
    df_filtered["space_group_order"] = df_filtered["space_group"].apply(get_space_group_order)

    # Calculate rank score (higher is better)
    df_filtered["rank_score"] = (
        -df_filtered["min_avg_distance"]  # Lower distance is better (negative to invert)
        + df_filtered["point_group_order"] / 48  # Normalize by max point group order
        + df_filtered["space_group_order"] / 230  # Normalize by max space group number
    )

    # Sort by rank score in descending order
    df_sorted = df_filtered.sort_values("rank_score", ascending=False)

    return df_sorted