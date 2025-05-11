"""
Postprocessing functions for cluster_finder package.

This module contains functions for analyzing, classifying, and ranking clusters.
"""

import numpy as np
import pandas as pd
import ast
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ..core.structure import calculate_centroid
from ..utils.helpers import get_mp_property, get_mp_properties_batch
from ..utils.logger import console, get_logger

# Get a logger for this module
logger = get_logger('cluster_finder.analysis.postprocess')

# Define mappings for symmetry analysis
# Point group order mapping (simplified for common point groups)
point_group_order_mapping = {
    "1": 1, "-1": 2, "2": 2, "m": 2, "2/m": 4,
    "222": 4, "mm2": 4, "mmm": 8,
    "4": 4, "-4": 4, "4/m": 8, "422": 8, "4mm": 8, "-42m": 8, "4/mmm": 16,
    "3": 3, "-3": 6, "32": 6, "3m": 6, "-3m": 12,
    "6": 6, "-6": 6, "6/m": 12, "622": 12, "6mm": 12, "-62m": 12, "6/mmm": 24,
    "23": 12, "m-3": 24, "432": 24, "-43m": 24, "m-3m": 48,
    # Add common Schoenflies notation
    "C1": 1, "Ci": 2, "C2": 2, "Cs": 2, "C2h": 4,
    "D2": 4, "C2v": 4, "D2h": 8,
    "C4": 4, "S4": 4, "C4h": 8, "D4": 8, "C4v": 8, "D2d": 8, "D4h": 16,
    "C3": 3, "C3i": 6, "D3": 6, "C3v": 6, "D3d": 12,
    "C6": 6, "C3h": 6, "C6h": 12, "D6": 12, "C6v": 12, "D3h": 12, "D6h": 24,
    "T": 12, "Th": 24, "O": 24, "Td": 24, "Oh": 48,
    # Add special symbols from the dataset
    "C*v": 4, "D*h": 24, "C2v": 4
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


def get_highest_point_group_order(point_groups_dict):
    """
    Get the order of the highest point group from a dictionary of point groups.
    
    Parameters:
        point_groups_dict (dict): Dictionary of point groups (cluster_id -> point_group)
        
    Returns:
        int: Highest order of the point groups
    """
    if not point_groups_dict:
        return 0
        
    # Get the order for each point group
    orders = [get_point_group_order(pg) for pg in point_groups_dict.values()]
    
    # Return the highest order
    return max(orders) if orders else 0


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


def rank_clusters(data_source, api_key=None, custom_props=None, prop_weights=None, include_default_ranking=True):
    """
    Rank clusters based on dimensionality, point group order, space group order,
    and optionally additional custom materials properties with specified weights.
    
    Parameters:
        data_source (str or pandas.DataFrame): Path to a CSV file or a pandas DataFrame
        api_key (str, optional): Materials Project API key. If None, will use the API key
                                 set in the MAPI_KEY environment variable.
        custom_props (list, optional): List of material property names to include in ranking
                                       (e.g. ['band_gap', 'density'])
        prop_weights (dict, optional): Dictionary of weights for each property
                                       (e.g. {'band_gap': 2.0, 'density': -1.0})
                                       Positive weights favor higher values, negative weights favor lower values
        include_default_ranking (bool, optional): Whether to include the default ranking criteria
                                                 (min_avg_distance, point_group_order, space_group_order)
        
    Returns:
        pandas.DataFrame: Sorted DataFrame with additional ranking columns and custom properties
    """
    # Use logger instead of console.print for consistent formatting with timestamps
    logger.info(f"Starting rank_clusters function with initial dataset")
    
    if isinstance(data_source, str):
        # Use pandas optimized CSV reader
        df = pd.read_csv(data_source, engine='c', dtype={'material_id': str, 'formula': str})
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()  # Create a copy to avoid modifying the original
    else:
        raise TypeError("Data_source must be either a file path (str) or a pandas DataFrame")

    # Make a copy of the original DataFrame to avoid modifying it directly
    df_filtered = df.copy()
    logger.info(f"Initial dataframe has {len(df_filtered)} rows")
    
    # Pre-create all mapping series to avoid repetitive dict lookups
    point_group_order_series = pd.Series(point_group_order_mapping)
    space_group_order_series = pd.Series(space_group_number_mapping)
    
    # Check if necessary columns exist, if not add them with default values
    if "space_group" not in df_filtered.columns:
        logger.warning("'space_group' column not found. Using default value.")
        df_filtered["space_group"] = "P1"  # Default to lowest symmetry
    
    # Optimize point_groups column processing with vectorized operations
    if "point_groups" in df_filtered.columns:
        try:
            # Use a faster string-to-dict parser for the entire column at once
            df_filtered["point_groups_dict"] = df_filtered["point_groups"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or {})
            )
            
            # Vectorized extraction of highest order point group
            df_filtered["max_point_group_order"] = df_filtered["point_groups_dict"].apply(get_highest_point_group_order)
            
            # Optimize the highest point group extraction
            def get_highest_order_point_group(pg_dict):
                if not pg_dict:
                    return "C1"
                pg_with_order = [(pg, point_group_order_mapping.get(pg, 0)) for pg in pg_dict.values()]
                return max(pg_with_order, key=lambda x: x[1])[0] if pg_with_order else "C1"
            
            df_filtered["highest_point_group"] = df_filtered["point_groups_dict"].apply(get_highest_order_point_group)
            
        except (ValueError, SyntaxError, AttributeError) as e:
            logger.warning(f"Could not parse 'point_groups' column. Using default value. Error: {e}")
            df_filtered["max_point_group_order"] = 1  # Default to lowest order
            df_filtered["highest_point_group"] = "C1"
    elif "point_group" in df_filtered.columns:
        logger.info("Using 'point_group' column for ranking...")
        df_filtered["highest_point_group"] = df_filtered["point_group"]
        
        # Vectorized mapping using pre-created Series
        df_filtered["max_point_group_order"] = df_filtered["point_group"].map(point_group_order_series).fillna(0)
    else:
        logger.warning("Neither 'point_groups' nor 'point_group' column found. Using default value.")
        df_filtered["highest_point_group"] = "C1"
        df_filtered["max_point_group_order"] = 1  # Default to lowest order
    
    # Optimize cluster size filtering
    if "cluster_sizes" in df_filtered.columns:
        # Count materials before filtering
        num_before = len(df_filtered)
        
        # Convert string cluster_sizes to list if needed - with optimized function
        if df_filtered["cluster_sizes"].dtype == 'object':
            # Define a faster parser for lists that handles both strings and actual lists
            def parse_list(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return []
                return []
                
            df_filtered["cluster_sizes_list"] = df_filtered["cluster_sizes"].apply(parse_list)
        else:
            df_filtered["cluster_sizes_list"] = df_filtered["cluster_sizes"]
        
        # Optimize large cluster check with NumPy vectorization where possible
        def has_large_clusters(clusters):
            if not clusters:
                return False
            # Convert to numpy array for faster processing if possible
            try:
                if isinstance(clusters[0], (int, float, str)):
                    sizes = np.array([int(size) for size in clusters])
                    return np.any(sizes > 8)
                else:
                    return not all(int(size) <= 8 for size in clusters)
            except:
                return not all(int(size) <= 8 for size in clusters)
        
        # Apply filtering with optimized masks
        large_clusters_mask = df_filtered["cluster_sizes_list"].apply(has_large_clusters)
        
        if "space_group" in df_filtered.columns:
            # Create space group mask using vectorized comparison
            space_group_mask = (df_filtered["space_group"] != "Symmetry Not Determined")
            
            # Log materials being dropped
            if (~space_group_mask).any():
                symm_not_determined = df_filtered[~space_group_mask]
                logger.info(f"Dropping {len(symm_not_determined)} materials with 'Symmetry Not Determined' space group")
            
            if large_clusters_mask.any():
                large_clusters = df_filtered[large_clusters_mask]
                logger.info(f"Dropping {len(large_clusters)} materials with cluster size > 8")
            
            # Apply combined filtering in one operation
            df_filtered = df_filtered[space_group_mask & ~large_clusters_mask]
        else:
            # Only filter by cluster size
            if large_clusters_mask.any():
                large_clusters = df_filtered[large_clusters_mask]
                logger.info(f"Dropping {len(large_clusters)} materials with cluster size > 8")
            
            # Apply filter with boolean indexing
            df_filtered = df_filtered[~large_clusters_mask]
        
        # Report how many materials were filtered out
        num_after = len(df_filtered)
        if num_before > num_after:
            logger.info(f"Filtered out {num_before - num_after} materials due to symmetry or cluster size criteria")
            logger.info(f"Remaining materials: {num_after}")

    # Optimize average_distance processing
    if "average_distance" in df_filtered.columns:
        # Convert string to list with optimized parsing
        if df_filtered["average_distance"].dtype == 'object':
            def parse_distance_list(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, (int, float)):
                    return [float(x)]
                if isinstance(x, str):
                    try:
                        val = ast.literal_eval(x)
                        return val if isinstance(val, list) else [val]
                    except:
                        try:
                            return [float(x)]
                        except:
                            return []
                return []
                
            df_filtered["average_distance"] = df_filtered["average_distance"].apply(parse_distance_list)
        
        # Optimize minimum calculation using numpy
        def safe_min(distances):
            if isinstance(distances, list) and distances:
                return float(np.min(distances))
            return None
            
        df_filtered["min_avg_distance"] = df_filtered["average_distance"].apply(safe_min)
    else:
        logger.warning("'average_distance' column not found. Skipping distance-based ranking.")
        df_filtered["min_avg_distance"] = 0  # Default to 0 for ranking

    # Calculate space group order using pre-created mapping Series
    df_filtered["space_group_order"] = df_filtered["space_group"].map(space_group_order_series).fillna(0)
    
    # Optimize material_id lookup
    material_id_col = next((col for col in ["material_id", "compound_id", "id"] 
                          if col in df_filtered.columns), None)
  
    # Add custom properties more efficiently
    if custom_props and material_id_col:
        # Get all unique material IDs from the DataFrame - vectorized
        material_ids = df_filtered[material_id_col].dropna().unique().tolist()
        logger.info(f"Retrieving properties for {len(material_ids)} unique materials")
        
        # Filter out any custom properties that are already in the DataFrame
        existing_props = set(df_filtered.columns)
        props_to_fetch = [prop for prop in custom_props if prop not in existing_props]
        
        if props_to_fetch:
            logger.info("Retrieving data from Materials Project...")
            # Use the batch function to retrieve all properties at once
            properties_dict = get_mp_properties_batch(material_ids, props_to_fetch, api_key)
            
            # Add properties to the DataFrame - vectorized mapping
            for prop in props_to_fetch:
                # Create a temporary mapping dictionary for this property
                prop_map = {mid: properties_dict.get(mid, {}).get(prop) 
                           for mid in material_ids 
                           if mid in properties_dict and prop in properties_dict[mid]}
                
                # Add the property to the DataFrame using a Series for mapping
                if prop_map:
                    prop_series = pd.Series(prop_map)
                    df_filtered[prop] = df_filtered[material_id_col].map(prop_series)
                    logger.info(f"Added {prop} for {len(prop_map)} materials")
                else:
                    logger.warning(f"Property '{prop}' could not be retrieved from Materials Project")
    elif custom_props and not material_id_col:
        logger.warning("Cannot retrieve custom properties without a material_id column.")
    
    # Initialize rank score
    df_filtered["rank_score"] = 0.0
    
    # Default weights for standard properties
    default_weights = {
        "min_avg_distance": -1.0,  # Lower distance is better
        "max_point_group_order": 1.0 / 48,  # Normalize by max point group order
        "space_group_order": 1.0 / 230,  # Normalize by max space group number
    }
    
    # Add custom property weights if not provided
    if custom_props and prop_weights is None:
        prop_weights = {prop: 0.0 for prop in custom_props}  # Default weight for custom properties
    
    # Calculate normalization factors for numerical properties
    norm_factors = {}
    numerical_props = []
    
    if custom_props:
        for prop in custom_props:
            if prop in df_filtered.columns:
                # Check if property is numerical using pandas methods
                try:
                    # Convert column to numeric using pandas optimized function
                    numeric_values = pd.to_numeric(df_filtered[prop], errors='coerce')
                    
                    if not numeric_values.dropna().empty:
                        # Store the converted values for efficiency
                        df_filtered[prop] = numeric_values
                        
                        # Calculate normalization factor efficiently
                        prop_values = numeric_values.dropna()
                        value_range = prop_values.max() - prop_values.min()
                        
                        norm_factors[prop] = 1.0 / value_range if value_range > 0 else 1.0
                        numerical_props.append(prop)
                    else:
                        logger.warning(f"Property '{prop}' contains non-numerical values and will be excluded from ranking.")
                except Exception as e:
                    logger.warning(f"Property '{prop}' cannot be converted to numeric values: {e}")
    
    # Calculate rank score with vectorized operations
    if include_default_ranking:
        for prop, weight in default_weights.items():
            if prop in df_filtered.columns:
                # Handle missing values efficiently
                if prop == "min_avg_distance":
                    df_filtered[prop] = df_filtered[prop].fillna(df_filtered[prop].mean())
                else:
                    df_filtered[prop] = df_filtered[prop].fillna(0)
                
                # Add weighted contribution - vectorized
                df_filtered["rank_score"] += weight * df_filtered[prop]
    
    # Add custom property contributions
    if custom_props and prop_weights:
        for prop in numerical_props:
            if prop in df_filtered.columns:
                weight = prop_weights.get(prop, 1.0)
                norm_factor = norm_factors.get(prop, 1.0)
                
                # Handle missing values and add contribution efficiently
                df_filtered[prop] = df_filtered[prop].fillna(df_filtered[prop].mean())
                df_filtered["rank_score"] += weight * norm_factor * df_filtered[prop]
    
    # Clean up temporary columns
    cols_to_drop = []
    if "cluster_sizes_list" in df_filtered.columns:
        cols_to_drop.append("cluster_sizes_list")
    if "point_groups_dict" in df_filtered.columns:
        cols_to_drop.append("point_groups_dict")
    
    if cols_to_drop:
        df_filtered = df_filtered.drop(columns=cols_to_drop)
    
    # Sort by rank score in descending order (higher is better)
    df_sorted = df_filtered.sort_values("rank_score", ascending=False)
    logger.info(f"Final ranked dataframe has {len(df_sorted)} materials")
    
    return df_sorted