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
from ..utils.config_utils import load_config

# Load configuration
try:
    config = load_config()
    DEFAULT_MAX_RADIUS = config['clustering']['default_max_radius']
    DEFAULT_CLUSTER_SIZE = config['clustering']['default_cluster_size']
    MAX_CLUSTER_SIZE = config['clustering']['max_cluster_size']
    SPLIT_CLUSTER_INITIAL_CUTOFF = config['clustering']['split_cluster']['initial_cutoff']
    SPLIT_CLUSTER_CUTOFF_STEP = config['clustering']['split_cluster']['cutoff_step']
    SPLIT_CLUSTER_MIN_CUTOFF = config['clustering']['split_cluster']['min_cutoff']
    SPLIT_CLUSTER_TOLERANCE = config['clustering']['split_cluster']['tolerance']
    SPLIT_CLUSTER_CENTROID_THRESHOLD = config['clustering']['split_cluster']['centroid_threshold']
    SPLIT_CLUSTER_MAX_RECURSION = config['clustering']['split_cluster'].get('max_recursion', 3)
    ANALYSIS_N_JOBS = config['clustering']['analyze_clusters']['n_jobs']
    STR_EXTENSION = config['clustering']['analyze_clusters'].get('str_extension', 2)
    IDENTIFY_UNIQUE_CLUSTERS_USE_SYMMETRY = config['clustering']['identify_unique_clusters']['use_symmetry']
    IDENTIFY_UNIQUE_CLUSTERS_TOLERANCE = config['clustering']['identify_unique_clusters']['tolerance']
except (KeyError, FileNotFoundError):
    # Fallback to default values if config loading fails
    DEFAULT_MAX_RADIUS = 3.0  # Maximum atom-to-atom distance for cluster search
    DEFAULT_CLUSTER_SIZE = 2  # Minimum number of atoms in a cluster
    MAX_CLUSTER_SIZE = 8  # Maximum cluster size for ranking
    SPLIT_CLUSTER_INITIAL_CUTOFF = 3.0
    SPLIT_CLUSTER_CUTOFF_STEP = 0.01
    SPLIT_CLUSTER_MIN_CUTOFF = 2.5
    SPLIT_CLUSTER_TOLERANCE = 0.2
    SPLIT_CLUSTER_CENTROID_THRESHOLD = 3.0
    SPLIT_CLUSTER_MAX_RECURSION = 3  # Maximum number of recursive splits
    ANALYSIS_N_JOBS = 4
    STR_EXTENSION = 2  # Default value for supercell extension
    IDENTIFY_UNIQUE_CLUSTERS_USE_SYMMETRY = True
    IDENTIFY_UNIQUE_CLUSTERS_TOLERANCE = 0.001


def find_clusters(structure, graph, tm_indices, min_cluster_size=None):
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
    if min_cluster_size is None:
        min_cluster_size = DEFAULT_CLUSTER_SIZE
    
    clusters = []
    for component in nx.connected_components(graph):
        cluster_indices = [tm_indices[i] for i in component]
        cluster = [structure.sites[i] for i in cluster_indices]
        if len(cluster) >= min_cluster_size:  
            clusters.append(cluster)
    return clusters


def calculate_average_distance(sites, max_radius=None):
    """
    Calculate average distance between sites in a cluster.
    
    Parameters:
        sites (list): List of pymatgen Site objects
        max_radius (float): Maximum radius to consider for distances
        
    Returns:
        float: Average distance between sites
    """
    if max_radius is None:
        max_radius = DEFAULT_MAX_RADIUS
    
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


def build_graph(sites, cutoff=None, distances_cache=None):
    """
    Build a graph from a list of sites.
    
    Parameters:
        sites (list): List of pymatgen Site objects
        cutoff (float): Maximum distance for connectivity
        distances_cache (dict, optional): Cache of pre-computed distances
        
    Returns:
        networkx.Graph: Graph representation of connectivity
    """
    if cutoff is None:
        cutoff = DEFAULT_MAX_RADIUS
    
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


def split_cluster(cluster, parent_avg_distance, lattice, cluster_size=None, 
                 initial_cutoff=None, cutoff_step=None, min_cutoff=None, 
                 tolerance=None, centroid_threshold=None, max_recursion=None, _recursion_depth=0):
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
        max_recursion (int): Maximum number of recursive splits
        _recursion_depth (int): Internal parameter to track recursion depth

    Returns:
        list: List of sub-clusters or [cluster] if no split is performed
    """
    # Use config values if parameters are not provided
    if cluster_size is None:
        cluster_size = MAX_CLUSTER_SIZE
    if initial_cutoff is None:
        initial_cutoff = SPLIT_CLUSTER_INITIAL_CUTOFF
    if cutoff_step is None:
        cutoff_step = SPLIT_CLUSTER_CUTOFF_STEP
    if min_cutoff is None:
        min_cutoff = SPLIT_CLUSTER_MIN_CUTOFF
    if tolerance is None:
        tolerance = SPLIT_CLUSTER_TOLERANCE
    if centroid_threshold is None:
        centroid_threshold = SPLIT_CLUSTER_CENTROID_THRESHOLD
    if max_recursion is None:
        max_recursion = SPLIT_CLUSTER_MAX_RECURSION
    
    # Check recursion depth limit
    if _recursion_depth >= max_recursion:
        return [cluster]
    
    # Skip small clusters immediately
    if len(cluster) < cluster_size:
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
            if len(sub_cluster) >= cluster_size:
                # Recursively split large sub-clusters
                sub_avg_distance = calculate_average_distance(sub_cluster, initial_cutoff)
                sub_clusters.extend(
                    split_cluster(sub_cluster, sub_avg_distance, lattice, 
                                 cluster_size, current_cutoff, cutoff_step,
                                 min_cutoff, tolerance, centroid_threshold,
                                 max_recursion, _recursion_depth + 1))
            else:
                sub_clusters.append(sub_cluster)
        
        # Return results if we found valid sub-clusters
        if sub_clusters:
            return sub_clusters

    # No valid split found
    return [cluster]


def analyze_clusters(clusters, lattice, cluster_size=None, max_radius=None, n_jobs=None, str_extension=None):
    """
    Analyze and process clusters found in a structure.
    
    Parameters:
        clusters (list): List of clusters (each a list of pymatgen Site objects)
        lattice (Lattice): The lattice of the structure
        cluster_size (int): Desired maximum cluster size
        max_radius (float): Maximum radius to consider
        n_jobs (int): Number of parallel jobs for processing clusters
        str_extension (int): Extension factor for creating supercell to check for extended structures
        
    Returns:
        list: Processed clusters with metadata
    """
    import concurrent.futures
    
    # Use config values if parameters are not provided
    if cluster_size is None:
        cluster_size = MAX_CLUSTER_SIZE
    if max_radius is None:
        max_radius = DEFAULT_MAX_RADIUS
    if n_jobs is None:
        n_jobs = ANALYSIS_N_JOBS
    if str_extension is None:
        str_extension = STR_EXTENSION
    
    # Process a single cluster
    def process_cluster(cluster):
        avg_distance = calculate_average_distance(cluster, max_radius)
        
        # Create a structure from the cluster to enable supercell creation
        species = [site.specie for site in cluster]
        coords = [site.coords for site in cluster]
        temp_structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        
        # Create a supercell
        supercell = temp_structure.make_supercell([str_extension, str_extension, str_extension])
        
        # Find clusters in the supercell
        connectivity_matrix, sc_indices = create_connectivity_matrix(supercell, [s.symbol for s in species], cutoff=max_radius)
        sc_graph = structure_to_graph(connectivity_matrix)
        sc_clusters = find_clusters(supercell, sc_graph, sc_indices)
        
        # If supercell clusters found, analyze the first (largest) one
        is_extended = False
        is_shared = False
        
        if sc_clusters:
            largest_sc_cluster = sc_clusters[0]
            sc_avg_distance = calculate_average_distance(largest_sc_cluster, max_radius)
            
            # Check if cluster is extended: supercell has larger cluster with same avg distance
            if len(largest_sc_cluster) > len(cluster) and abs(sc_avg_distance - avg_distance) <= 0.0001:
                is_extended = True
            # Check if cluster is shared: supercell has different avg distance
            elif abs(sc_avg_distance - avg_distance) > 0.0001:
                is_shared = True
        
        # Only split clusters when they are neither extended nor shared
        if not is_extended and not is_shared:
            sub_clusters = split_cluster(cluster, avg_distance, lattice, cluster_size)
        else:
            sub_clusters = [cluster]
        
        result = []
        for sub_cluster in sub_clusters:
            sub_avg_distance = calculate_average_distance(sub_cluster, max_radius)
            centroid = calculate_centroid(sub_cluster, lattice)
            result.append({
                "sites": sub_cluster,
                "size": len(sub_cluster),
                "average_distance": sub_avg_distance,
                "centroid": centroid,
                "is_extended": is_extended,
                "is_shared": is_shared
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


def identify_unique_clusters(clusters, use_symmetry=None, tolerance=None):
    """
    Identify unique clusters based on atoms, connectivity, point group symmetry,
    and orientation. Assigns consistent labels to clusters with the same characteristics.
    
    Parameters:
        clusters (list): List of cluster dictionaries
        use_symmetry (bool): Whether to include point group symmetry in uniqueness criteria
        tolerance (float): Distance threshold for considering clusters as unique
        
    Returns:
        list: List of all clusters with metadata and consistent labels
    """
    from collections import defaultdict
    
    # Use config values if parameters are not provided
    if use_symmetry is None:
        use_symmetry = IDENTIFY_UNIQUE_CLUSTERS_USE_SYMMETRY
    if tolerance is None:
        tolerance = IDENTIFY_UNIQUE_CLUSTERS_TOLERANCE
    
    # First, prepare all clusters with the necessary properties
    for cluster in clusters:
        # Calculate point group if not already present
        if use_symmetry and "point_group" not in cluster:
            species = [site.specie for site in cluster["sites"]]
            cart_coords = np.array([site.coords for site in cluster["sites"]])
            molecule = Molecule(species, cart_coords)
            pga = PointGroupAnalyzer(molecule)
            cluster["point_group"] = pga.get_pointgroup().sch_symbol
            
        # Ensure average_distance is calculated
        if "average_distance" not in cluster:
            cluster["average_distance"] = calculate_average_distance(cluster["sites"])
            
        # Calculate the moment of inertia tensor for orientation comparison
        cart_coords = np.array([site.coords for site in cluster["sites"]])
        # Center the coordinates to calculate orientation properly
        center = np.mean(cart_coords, axis=0)
        centered_coords = cart_coords - center
        
        # Calculate moment of inertia tensor
        inertia_tensor = np.zeros((3, 3))
        for coord in centered_coords:
            # Diagonal elements
            inertia_tensor[0, 0] += coord[1]**2 + coord[2]**2
            inertia_tensor[1, 1] += coord[0]**2 + coord[2]**2
            inertia_tensor[2, 2] += coord[0]**2 + coord[1]**2
            # Off-diagonal elements
            inertia_tensor[0, 1] -= coord[0] * coord[1]
            inertia_tensor[0, 2] -= coord[0] * coord[2]
            inertia_tensor[1, 2] -= coord[1] * coord[2]
        
        # Make the tensor symmetric
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]
        
        # Calculate eigenvalues (principal moments of inertia)
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        # Sort eigenvalues to ensure consistent comparison
        eigenvalues = np.sort(eigenvalues)
        
        # Store eigenvalues for orientation comparison
        cluster["orientation_eigvals"] = eigenvalues
    
    # Group clusters by size, elements, and point group first (quick filters)
    size_groups = defaultdict(list)
    for cluster in clusters:
        # Create a key based on size, element composition, and point group
        elements = sorted([site.specie.symbol for site in cluster["sites"]])
        element_counts = {}
        for element in elements:
            element_counts[element] = element_counts.get(element, 0) + 1
        element_key = "-".join([f"{el}_{count}" for el, count in sorted(element_counts.items())])
        
        key = (
            cluster["size"], 
            element_key,
            cluster["point_group"] if use_symmetry and "point_group" in cluster else ""
        )
        size_groups[key].append(cluster)
    
    # Within each group, compare average distances and orientations to find truly unique clusters
    label_count = 0
    
    for group_key, size_group in size_groups.items():
        for i, cluster in enumerate(size_group):
            if "label" in cluster:
                continue  # Already processed
                
            # Assign a label to current cluster
            cluster["label"] = f"X{label_count}"
            
            # Compare with all other clusters in this group that haven't been processed
            for j in range(i+1, len(size_group)):
                other = size_group[j]
                if "label" in other:
                    continue  # Already processed
                
                # Compare average distances with tolerance
                if abs(cluster["average_distance"] - other["average_distance"]) <= tolerance:
                    # Check for orientation differences
                    orientation_same = True
                    for k in range(3):  # Compare all 3 eigenvalues
                        # If eigenvalues differ significantly, orientation is different
                        if abs(cluster["orientation_eigvals"][k] - other["orientation_eigvals"][k]) > tolerance:
                            orientation_same = False
                            break
                    
                    # Only consider clusters the same if orientation is also the same
                    if orientation_same:
                        # These clusters are geometrically equivalent with the same orientation
                        other["label"] = cluster["label"]  # Use same label as current cluster
                    else:
                        # Different orientation, assign a new label
                        label_count += 1
                        other["label"] = f"X{label_count}"
                else:
                    # Different average distance, assign a new label
                    label_count += 1
                    other["label"] = f"X{label_count}"
            
            # Increment label counter for next unique cluster
            label_count += 1
    
    # Return all clusters
    return clusters


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
        connectivity_matrix, tm_indices = create_connectivity_matrix(structure, transition_metals, cutoff=DEFAULT_MAX_RADIUS)
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