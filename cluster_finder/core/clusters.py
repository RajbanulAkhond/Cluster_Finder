"""
Cluster identification and analysis functionality for cluster_finder package.

This module contains functions for finding, analyzing, and manipulating clusters.
"""

import numpy as np
import networkx as nx
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from .graph import structure_to_graph, create_connectivity_matrix
from .utils import calculate_centroid

# Default constants
DEFAULT_MAX_RADIUS = 3.5  # Maximum atom-to-atom distance for cluster search
DEFAULT_CLUSTER_SIZE = 2  # Minimum number of atoms in a cluster


def find_clusters(structure, graph, tm_indices, min_cluster_size=DEFAULT_CLUSTER_SIZE):
    """
    Find clusters in a structure using graph components.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        graph (networkx.Graph): Graph representation of connectivity
        tm_indices (list): Indices of transition metal atoms
        min_cluster_size (int): Minimum number of atoms for a valid cluster
        
    Returns:
        list: List of clusters as lists of pymatgen Site objects
    """
    clusters = []
    for component in nx.connected_components(graph):
        cluster_indices = [tm_indices[i] for i in component]
        cluster = [structure.sites[i] for i in cluster_indices]
        if len(cluster) >= min_cluster_size:  
            clusters.append(cluster)
    return clusters


def calculate_average_distance(sites, max_radius=3.5):
    """
    Calculate average distance between sites in a cluster.
    
    Parameters:
        sites (list): List of pymatgen Site objects
        max_radius (float): Maximum radius to consider for distances
        
    Returns:
        float: Average distance between sites
    """
    if len(sites) < 2:
        return 0.0
    
    # Extract coordinates once for all sites
    coords = np.array([site.coords for site in sites])
    n_sites = len(sites)
    
    # Create indices for all pairs
    i, j = np.triu_indices(n_sites, k=1)
    
    # Calculate all pairwise distances at once using vectorized operations
    # Reshape into a format that allows broadcasting
    site_i_coords = coords[i]
    site_j_coords = coords[j]
    distances = np.linalg.norm(site_i_coords - site_j_coords, axis=1)
    
    # Filter distances within max_radius
    valid_distances = distances[distances <= max_radius]
    
    # Convert NumPy float to Python native float
    return float(np.mean(valid_distances)) if len(valid_distances) > 0 else 0.0


def build_graph(sites, cutoff=3.5, distances_cache=None):
    """
    Build a graph from a list of sites.
    
    Parameters:
        sites (list): List of pymatgen Site objects
        cutoff (float): Maximum distance for connectivity
        distances_cache (dict, optional): Cache of pre-computed distances
        
    Returns:
        networkx.Graph: Graph representation of connectivity
    """
    n_sites = len(sites)
    G = nx.Graph()
    G.add_nodes_from(range(n_sites))
    
    # If no distances cache provided, calculate all pairwise distances in batch
    if distances_cache is None:
        coords = np.array([site.coords for site in sites])
        i, j = np.triu_indices(n_sites, k=1)
        site_i_coords = coords[i]
        site_j_coords = coords[j]
        distances = np.linalg.norm(site_i_coords - site_j_coords, axis=1)
        
        # Add edges for pairs with distance <= cutoff
        edges = [(i[k], j[k]) for k in range(len(i)) if distances[k] <= cutoff]
        G.add_edges_from(edges)
    else:
        # Use pre-computed distances
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                key = tuple(sorted([i, j]))
                if key not in distances_cache:
                    # Calculate only if needed
                    distances_cache[key] = sites[i].distance(sites[j])
                
                if distances_cache[key] <= cutoff:
                    G.add_edge(i, j)
    
    return G


def split_cluster(cluster, parent_avg_distance, lattice, cluster_size=DEFAULT_CLUSTER_SIZE+1, 
                 initial_cutoff=3.5, cutoff_step=0.1, min_cutoff=2.5, 
                 tolerance=0.01, centroid_threshold=DEFAULT_MAX_RADIUS):
    """
    Split large clusters into smaller sub-clusters with centroid distance checks.

    Parameters:
        cluster (list): List of pymatgen Site objects
        parent_avg_distance (float): Average distance of the parent cluster
        lattice (Lattice): Lattice object for centroid calculations
        cluster_size (int): Desired maximum cluster size
        initial_cutoff (float): Initial distance cutoff
        cutoff_step (float): Step to reduce cutoff
        min_cutoff (float): Minimum cutoff distance
        tolerance (float): Tolerance for average distance difference
        centroid_threshold (float): Minimum distance between centroids

    Returns:
        list: List of sub-clusters or [cluster] if no split is performed
    """
    # Skip small clusters immediately
    if len(cluster) <= cluster_size:
        return [cluster]
        
    # Pre-compute all distances at once using NumPy arrays
    coords = np.array([site.coords for site in cluster])
    n_sites = len(cluster)
    
    # Create distance matrix once
    row_idx, col_idx = np.triu_indices(n_sites, k=1)
    distances = np.linalg.norm(coords[row_idx] - coords[col_idx], axis=1)
    
    # Create a mapping from (i, j) indices to distance array index
    i, j = np.triu_indices(n_sites, k=1)
    idx_mapping = {(i[k], j[k]): k for k in range(len(i))}
    
    # Create distances cache for faster lookup
    distances_cache = {}
    for i_idx in range(n_sites):
        for j_idx in range(i_idx+1, n_sites):
            key = (i_idx, j_idx)
            dist_idx = idx_mapping.get(key)
            if dist_idx is not None:
                distances_cache[key] = distances[dist_idx]
    
    # Use binary search approach to find cutoff more efficiently
    cutoffs_to_try = np.arange(initial_cutoff, min_cutoff - cutoff_step, -cutoff_step)
    
    for current_cutoff in cutoffs_to_try:
        # Build graph with cached distances
        graph = build_graph(cluster, current_cutoff, distances_cache)

        # Get candidate sub-clusters efficiently 
        components = list(nx.connected_components(graph))
        
        # Skip if there's only one component (the whole cluster)
        if len(components) <= 1:
            continue
            
        # Process all components at once
        candidate_sub_clusters = []
        
        for component in components:
            component = list(component)  # Convert to list
            if len(component) < 2:
                continue
                
            sub_cluster = [cluster[i] for i in component]
            # Extract relevant distances for this sub-component
            sub_dist_indices = []
            for idx, (i_idx, j_idx) in enumerate(zip(row_idx, col_idx)):
                if i_idx in component and j_idx in component:
                    sub_dist_indices.append(idx)
                    
            if sub_dist_indices:
                sub_distances = distances[sub_dist_indices]
                sub_distances = sub_distances[sub_distances <= initial_cutoff]
                sub_avg_distance = float(np.mean(sub_distances)) if len(sub_distances) > 0 else 0.0
                
                if (parent_avg_distance - sub_avg_distance) > tolerance:
                    candidate_sub_clusters.append(sub_cluster)
        
        # If no suitable candidates found, move to the next cutoff
        if not candidate_sub_clusters:
            continue
            
        # Calculate centroids once
        centroids = np.array([calculate_centroid(sub_cluster, lattice) for sub_cluster in candidate_sub_clusters])
        
        # Calculate all centroid distances at once using broadcasting
        if len(centroids) > 1:
            i, j = np.triu_indices(len(centroids), k=1)
            centroid_distances = np.linalg.norm(centroids[i] - centroids[j], axis=1)
            
            # Check if any centroids are too close
            if np.any(centroid_distances < centroid_threshold):
                return [cluster]  # Early exit if centroids are too close
        
        # Process sub-clusters
        sub_clusters = []
        for sub_cluster in candidate_sub_clusters:
            if len(sub_cluster) > cluster_size:
                # Recursively split large sub-clusters
                sub_avg_distance = calculate_average_distance(sub_cluster, initial_cutoff)
                sub_clusters.extend(
                    split_cluster(sub_cluster, sub_avg_distance, lattice, 
                                 cluster_size, initial_cutoff, cutoff_step,
                                 min_cutoff, tolerance, centroid_threshold))
            else:
                sub_clusters.append(sub_cluster)
        
        # Return results if we found valid sub-clusters
        if sub_clusters:
            return sub_clusters

    # No valid split found
    return [cluster]


def analyze_clusters(clusters, lattice, cluster_size=DEFAULT_CLUSTER_SIZE+1, max_radius=DEFAULT_MAX_RADIUS, n_jobs=4):
    """
    Analyze and process clusters found in a structure.
    
    Parameters:
        clusters (list): List of clusters (each a list of pymatgen Site objects)
        lattice (Lattice): The lattice of the structure
        cluster_size (int): Desired maximum cluster size
        max_radius (float): Maximum radius to consider
        n_jobs (int): Number of parallel jobs for processing clusters
        
    Returns:
        list: Processed clusters with metadata
    """
    import concurrent.futures
    
    # Process a single cluster
    def process_cluster(cluster):
        avg_distance = calculate_average_distance(cluster, max_radius)
        sub_clusters = split_cluster(cluster, avg_distance, lattice, cluster_size)
        
        result = []
        for sub_cluster in sub_clusters:
            sub_avg_distance = calculate_average_distance(sub_cluster, max_radius)
            centroid = calculate_centroid(sub_cluster, lattice)
            result.append({
                "sites": sub_cluster,
                "size": len(sub_cluster),
                "average_distance": sub_avg_distance,
                "centroid": centroid
            })
        return result
    
    # For small number of clusters, process serially
    if len(clusters) < 3:
        processed_clusters = []
        for cluster in clusters:
            processed_clusters.extend(process_cluster(cluster))
        return processed_clusters
    
    # Process clusters in parallel
    processed_clusters = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all clusters for processing
        future_to_cluster = {executor.submit(process_cluster, cluster): cluster for cluster in clusters}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_cluster):
            try:
                result = future.result()
                processed_clusters.extend(result)
            except Exception as exc:
                # Fall back to sequential processing for any failures
                cluster = future_to_cluster[future]
                processed_clusters.extend(process_cluster(cluster))
    
    return processed_clusters


def identify_unique_clusters(clusters, use_symmetry=True, tolerance=1e-5):
    """
    Identify unique clusters based on atoms, connectivity, and optionally point group symmetry.
    Assigns consistent labels to clusters with the same characteristics.
    
    Parameters:
        clusters (list): List of cluster dictionaries
        use_symmetry (bool): Whether to include point group symmetry in uniqueness criteria
        tolerance (float): Distance threshold for considering clusters as unique
        
    Returns:
        list: List of all clusters with metadata and consistent labels
    """
    # Create a mapping of cluster_key -> label
    cluster_key_to_label = {}
    label_count = 0
    labeled_clusters = []
    
    for cluster in clusters:
        # Always consider basic properties (size, distance, elements)
        atom_types = sorted([site.specie.symbol for site in cluster["sites"]])
        basic_key = (
            cluster["size"],
            round(cluster["average_distance"], 3),
            tuple(atom_types)
        )
        
        # Optionally add point group symmetry for more precise classification
        if use_symmetry:
            # Calculate point group symmetry if not already present
            if "point_group" not in cluster:
                # Create a molecule object for symmetry analysis
                species = [site.specie for site in cluster["sites"]]
                cartesian_coords = [site.coords for site in cluster["sites"]]
                molecule = Molecule(species, cartesian_coords)
                pga = PointGroupAnalyzer(molecule)
                point_group = pga.get_pointgroup()
                point_group_symbol = point_group.sch_symbol
                cluster["point_group"] = point_group_symbol
            else:
                point_group_symbol = cluster["point_group"]
                
            # Create a graph to analyze connectivity
            conn_graph = build_graph(cluster["sites"], cluster["average_distance"] * 1.1)
            edge_count = len(conn_graph.edges())
            
            # Extended key including symmetry and connectivity
            cluster_key = (basic_key, edge_count, point_group_symbol)
        else:
            # Use only basic properties for the key
            cluster_key = basic_key
        
        # Create a copy of the cluster to avoid modifying the original
        cluster_with_label = cluster.copy()
        
        # If this cluster key is already seen, use the existing label
        # Otherwise, create a new label
        if cluster_key not in cluster_key_to_label:
            label = f"X{label_count}"
            cluster_key_to_label[cluster_key] = label
            label_count += 1
        else:
            label = cluster_key_to_label[cluster_key]
        
        # Assign the label to the cluster
        cluster_with_label["label"] = label
        labeled_clusters.append(cluster_with_label)
    
    return labeled_clusters


def get_compounds_with_clusters(entries, transition_metals, primary_transition_metal=None):
    """
    Process a list of entries to find and analyze clusters in each compound.
    
    Parameters:
        entries (list): List of entries containing material data
        transition_metals (list): List of transition metal element symbols
        primary_transition_metal (str, optional): The main transition metal to filter clusters
        
    Returns:
        tuple: (compounds_with_clusters, graph, structure, tm_indices) where:
            - compounds_with_clusters (list): List of dictionaries containing compound data and clusters
            - graph (networkx.Graph): Graph representation of the last processed structure
            - structure (Structure): Last processed structure
            - tm_indices (list): Transition metal indices from the last processed structure
    """
    compounds_with_clusters = []
    graph = None
    structure = None
    tm_indices = None
    
    for entry in entries:
        material_id = entry.material_id
        formula = entry.formula_pretty
        structure = entry.structure
        total_magnetization = entry.total_magnetization
        connectivity_matrix, tm_indices = create_connectivity_matrix(structure, transition_metals)
        graph = structure_to_graph(connectivity_matrix)
        # Find all clusters
        clusters = find_clusters(structure, graph, tm_indices)
        analyzed_clusters = analyze_clusters(clusters, structure.lattice)

        # Filter clusters to ensure they contain the primary transition metal
        # and exclude clusters with average distance less than equal to 2.3
        if primary_transition_metal:
            analyzed_clusters = [
                c for c in analyzed_clusters
                if any(site.specie.symbol == primary_transition_metal for site in c["sites"])
                and not round(c["average_distance"], 1) <= 2.3
            ]

        # Skip compounds that now have zero valid clusters
        if not analyzed_clusters:
            continue

        compounds_with_clusters.append({
            "material_id": material_id,
            "formula": formula,
            "total_magnetization": total_magnetization,
            "clusters": analyzed_clusters,
            "graph": graph,
            "structure": structure
        })
    
    return compounds_with_clusters