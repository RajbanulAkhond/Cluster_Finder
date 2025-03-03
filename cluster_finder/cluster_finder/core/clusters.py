"""
Cluster identification and analysis functionality for cluster_finder package.

This module contains functions for finding, analyzing, and manipulating clusters.
"""

import numpy as np
import networkx as nx
from .structure import calculate_centroid, structure_to_graph

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


def calculate_average_distance(cluster, max_radius=DEFAULT_MAX_RADIUS):
    """
    Calculate the average distance between atoms in a cluster.
    
    Parameters:
        cluster (list): List of pymatgen Site objects
        max_radius (float): Maximum radius to consider
        
    Returns:
        float: Average distance
    """
    distances = []
    for i, site1 in enumerate(cluster):
        for j, site2 in enumerate(cluster[i+1:], start=i+1):
            distance = site1.distance(site2)
            if distance <= max_radius:
                distances.append(distance)
    return np.mean(distances) if distances else float('inf')


def build_graph(cluster, cutoff, distances_cache=None):
    """
    Build a graph from a cluster based on distance cutoff.
    
    Parameters:
        cluster (list): List of pymatgen Site objects
        cutoff (float): Distance cutoff for connections
        distances_cache (dict, optional): Cache of precomputed distances
        
    Returns:
        networkx.Graph: Graph representation of the cluster
    """
    G = nx.Graph()
    for i in range(len(cluster)):
        for j in range(i+1, len(cluster)):
            if distances_cache is not None and (i, j) in distances_cache:
                distance = distances_cache[(i, j)]
            else:
                distance = cluster[i].distance(cluster[j])
                if distances_cache is not None:
                    distances_cache[(i, j)] = distance
                    
            if distance <= cutoff:
                G.add_edge(i, j)
    return G


def split_cluster(cluster, parent_avg_distance, lattice, cluster_size=DEFAULT_CLUSTER_SIZE+1, 
                 initial_cutoff=3.5, cutoff_step=0.001, min_cutoff=2.5, 
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
    # Cache distances once to avoid redundant calculations
    distances_cache = {}
    for i, site1 in enumerate(cluster):
        for j, site2 in enumerate(cluster[i+1:], start=i+1):
            distances_cache[(i, j)] = site1.distance(site2)

    current_cutoff = initial_cutoff
    while current_cutoff >= min_cutoff:
        # Build graph with cached distances
        graph = build_graph(cluster, current_cutoff, distances_cache)

        # Identify candidate sub-clusters
        candidate_sub_clusters = []
        for component in nx.connected_components(graph):
            sub_cluster = [cluster[i] for i in component]
            if len(sub_cluster) >= 2:
                sub_avg_distance = calculate_average_distance(sub_cluster, initial_cutoff)
                if (parent_avg_distance - sub_avg_distance) > tolerance:
                    candidate_sub_clusters.append(sub_cluster)

        # Process candidates if any
        if candidate_sub_clusters:
            # Calculate centroids
            centroids = [calculate_centroid(sub_cluster, lattice)
                        for sub_cluster in candidate_sub_clusters]

            # Check centroid distances efficiently
            for i, centroid_i in enumerate(centroids):
                for centroid_j in centroids[i+1:]:
                    if np.linalg.norm(centroid_i - centroid_j) < centroid_threshold:
                        return [cluster]  # Early exit if centroids are too close

            # Process sub-clusters
            sub_clusters = []
            for sub_cluster in candidate_sub_clusters:
                if len(sub_cluster) > cluster_size:
                    sub_clusters.extend(
                        split_cluster(sub_cluster,
                                    calculate_average_distance(sub_cluster, initial_cutoff),
                                    lattice, cluster_size, initial_cutoff, cutoff_step,
                                    min_cutoff, tolerance, centroid_threshold))
                else:
                    sub_clusters.append(sub_cluster)
            return sub_clusters

        current_cutoff -= cutoff_step

    # No valid split found
    return [cluster]


def analyze_clusters(clusters, lattice, cluster_size=DEFAULT_CLUSTER_SIZE+1, max_radius=DEFAULT_MAX_RADIUS):
    """
    Analyze and process clusters found in a structure.
    
    Parameters:
        clusters (list): List of clusters (each a list of pymatgen Site objects)
        lattice (Lattice): The lattice of the structure
        cluster_size (int): Desired maximum cluster size
        max_radius (float): Maximum radius to consider
        
    Returns:
        list: Processed clusters with metadata
    """
    processed_clusters = []
    for cluster in clusters:
        avg_distance = calculate_average_distance(cluster, max_radius)
        sub_clusters = split_cluster(cluster, avg_distance, lattice, cluster_size)
        
        for sub_cluster in sub_clusters:
            sub_avg_distance = calculate_average_distance(sub_cluster, max_radius)
            centroid = calculate_centroid(sub_cluster, lattice)
            processed_clusters.append({
                "sites": sub_cluster,
                "size": len(sub_cluster),
                "average_distance": sub_avg_distance,
                "centroid": centroid
            })
    
    return processed_clusters


def identify_unique_clusters(clusters):
    """
    Identify unique clusters based on atoms and connectivity.
    
    Parameters:
        clusters (list): List of cluster dictionaries
        
    Returns:
        list: List of unique clusters with metadata
    """
    unique_clusters = []
    seen_cluster_keys = set()
    
    for cluster in clusters:
        # Generate a key for each cluster based on atom types and connectivity
        atom_types = sorted([site.specie.symbol for site in cluster["sites"]])
        
        # Create a simplified connectivity representation
        conn_graph = build_graph(cluster["sites"], cluster["average_distance"] * 1.1)
        edge_count = len(conn_graph.edges())
        
        cluster_key = (tuple(atom_types), edge_count)
        
        if cluster_key not in seen_cluster_keys:
            seen_cluster_keys.add(cluster_key)
            unique_clusters.append(cluster)
    
    return unique_clusters 