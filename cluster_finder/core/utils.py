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
                    for site in cluster.get('sites', []):
                        # Extract element and position from site object or dictionary
                        element = "Unknown"
                        position = "Unknown"
                        
                        # Case 1: pymatgen Site object
                        if hasattr(site, 'specie') and hasattr(site.specie, 'symbol'):
                            element = site.specie.symbol
                            position = site.frac_coords
                            
                        # Case 2: Dictionary representation of a site
                        elif isinstance(site, dict):
                            # Try to extract element information
                            if "species" in site:
                                species_data = site["species"]
                                
                                # Handle various formats for species data
                                if isinstance(species_data, list) and len(species_data) > 0:
                                    if isinstance(species_data[0], dict):
                                        if "element" in species_data[0]:
                                            element_data = species_data[0]["element"]
                                            if isinstance(element_data, dict):
                                                element = next(iter(element_data.keys()), "Unknown")
                                            else:
                                                element = str(element_data)
                                    else:
                                        element = str(species_data[0])
                                elif isinstance(species_data, dict) and "element" in species_data:
                                    element_data = species_data["element"]
                                    if isinstance(element_data, dict):
                                        element = next(iter(element_data.keys()), "Unknown")
                                    else:
                                        element = str(element_data)
                                else:
                                    element = str(species_data)
                            elif "element" in site:
                                element = site["element"]
                            elif "elements" in cluster and isinstance(cluster["elements"], list):
                                try:
                                    idx = cluster['sites'].index(site)
                                    if idx < len(cluster["elements"]):
                                        element = cluster["elements"][idx]
                                except (ValueError, IndexError):
                                    pass
                            
                            # Try to extract position information
                            if "frac_coords" in site:
                                position = site["frac_coords"]
                            elif "xyz" in site:
                                position = site["xyz"]
                            elif "coords" in site:
                                position = site["coords"]
                            # Check for properties dictionary structure (common in MP API)
                            elif "properties" in site and "frac_coords" in site["properties"]:
                                position = site["properties"]["frac_coords"]
                        
                        # Final attempt to find position in alternative locations
                        if position == "Unknown" and isinstance(site, dict):
                            # Some MP API formats nest coordinates differently
                            for key in ["abc", "fractional_coords", "position", "pos"]:
                                if key in site:
                                    position = site[key]
                                    break
                                    
                            # For JSON format where we might have "sites" as serialized dicts
                            # Check if there's a "coords" key at the top level
                            if "coords" in cluster:
                                try:
                                    idx = cluster['sites'].index(site)
                                    if idx < len(cluster["coords"]):
                                        position = cluster["coords"][idx]
                                except (ValueError, IndexError):
                                    pass
                        
                        summary.append(f"Element: {element}, Position: {position}")
            else:
                summary.append("No clusters found in this structure")
            
            summary.append("")  # Add blank line between compounds
        except KeyError as e:
            summary.append(f"Error processing compound: {str(e)}")
    
    return "\n".join(summary)