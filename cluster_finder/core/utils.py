"""
Utility functions for cluster_finder package.

This module provides common utility functions used throughout the package,
particularly focusing on:
- Statistical analysis of clusters
- Data summarization and reporting
- Error handling and input validation
- Geometric calculations

The functions in this module are designed to work with the data structures
used in the cluster_finder package, particularly the compound and cluster
dictionaries produced by the analysis functions.
"""

import numpy as np

def calculate_centroid(sites, lattice=None):
    """
    Calculate the centroid (geometric center) of a group of sites.
    
    For periodic structures (when lattice is provided), the function handles
    periodic boundary conditions by:
    1. Converting to fractional coordinates
    2. Wrapping coordinates to [0, 1) range
    3. Converting back to cartesian coordinates
    
    Parameters:
        sites (list): List of pymatgen Site objects representing atomic positions
        lattice (Lattice, optional): Lattice object for periodic boundary conditions.
                                   If None, treats coordinates as non-periodic.
        
    Returns:
        numpy.ndarray: Cartesian coordinates of the centroid
    
    Example:
        >>> sites = [site1, site2, site3]  # List of pymatgen Site objects
        >>> centroid = calculate_centroid(sites, structure.lattice)
    """
    coords = np.array([site.coords for site in sites])
    if lattice is not None:
        # Apply periodic boundary conditions if lattice is provided
        frac_coords = lattice.get_fractional_coords(coords)
        # Wrap to [0, 1)
        frac_coords = frac_coords % 1.0
        # Convert back to cartesian
        coords = lattice.get_cartesian_coords(frac_coords)
    return np.mean(coords, axis=0)

def cluster_summary_stat(compounds_with_clusters, entries):
    """
    Generate a comprehensive summary of cluster statistics and analysis results.
    
    This function creates a detailed report that includes:
    - Total number of compounds analyzed
    - Number and percentage of compounds containing clusters
    - Detailed information for each compound:
        * Material ID and formula
        * Total magnetization
        * Number of clusters
        * For each cluster:
            - Size (number of atoms)
            - Average interatomic distance
            - Detailed site information
    
    The compounds are sorted by number of clusters in descending order
    for easier analysis of the most interesting cases.
    
    Parameters:
        compounds_with_clusters (list): List of compound dictionaries, each containing:
            - material_id (str): Unique identifier for the compound
            - formula (str): Chemical formula
            - total_magnetization (float): Total magnetic moment
            - clusters (list): List of cluster dictionaries
        entries (list): List of all analyzed entries (for statistics)
        
    Returns:
        str: A formatted multi-line string containing the analysis summary
        
    Raises:
        TypeError: If input parameters are None or not lists
        KeyError: If compound dictionaries are missing required fields
            (material_id, formula, total_magnetization, clusters)
    
    Example:
        >>> compounds = [{"material_id": "mp-123", "formula": "Fe2O3", ...}]
        >>> entries = [entry1, entry2, entry3]
        >>> summary = cluster_summary_stat(compounds, entries)
        >>> print(summary)
        Summary of Cluster Analysis
        -------------------------
        Total Compounds Analyzed: 3
        Compounds with Clusters: 1 (33.3%)
        ...
    
    Notes:
        - The function assumes that cluster dictionaries contain 'sites',
          'size', and 'average_distance' keys
        - Empty or missing clusters lists are handled gracefully
        - The summary includes blank lines between compounds for readability
    """
    if compounds_with_clusters is None or entries is None:
        raise TypeError("Input parameters cannot be None")
    
    if not isinstance(compounds_with_clusters, list) or not isinstance(entries, list):
        raise TypeError("Input parameters must be lists")

    if not entries:
        summary = ["Total Compounds Analyzed: 0"]
        summary.append("Compounds with Clusters: 0")
        return "\n".join(summary)

    total_compounds = len(entries)
    compounds_with_clusters_count = sum(1 for compound in compounds_with_clusters if compound.get('clusters', []))
    
    summary = []
    summary.append("Summary of Cluster Analysis")
    summary.append("-" * 30)
    summary.append(f"Total Compounds Analyzed: {total_compounds}")
    summary.append(f"Compounds with Clusters: {compounds_with_clusters_count} ({(compounds_with_clusters_count / total_compounds) * 100:.1f}%)")
    summary.append("")
    
    # Sort compounds by number of clusters in descending order
    sorted_compounds = sorted(compounds_with_clusters, 
                            key=lambda x: len(x.get('clusters', [])), 
                            reverse=True)
    
    for compound in sorted_compounds:
        try:
            summary.append(f"Material ID: {compound['material_id']}")
            summary.append(f"Formula: {compound['formula']}")
            summary.append(f"Total Magnetization: {compound['total_magnetization']}")
            
            clusters = compound.get('clusters', [])
            summary.append(f"Number of Clusters: {len(clusters)}")
            
            if clusters:
                for i, cluster in enumerate(clusters, 1):
                    summary.append(f"\nCluster {i}:")
                    summary.append(f"Size: {cluster['size']}")
                    summary.append(f"Average Distance: {cluster['average_distance']:.2f} Å")
                    
                    summary.append("Sites:")
                    for site in cluster['sites']:
                        summary.append(f"Element: {site.specie.symbol}, Position: {site.frac_coords}")
            else:
                summary.append("No clusters found in this structure")
            
            summary.append("")  # Add blank line between compounds
        except KeyError as e:
            raise KeyError(f"Missing required field in compound data: {str(e)}")
    
    return "\n".join(summary)