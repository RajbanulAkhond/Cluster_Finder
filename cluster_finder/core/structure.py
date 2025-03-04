"""
Structure related functions for cluster_finder package.

This module contains functions for analyzing, manipulating, and inspecting crystal structures,
particularly focusing on transition metal clusters. It provides functionality for:
- Finding and analyzing clusters in crystal structures
- Computing centroids and connectivity
- Converting structures to graph representations
- Generating supercells and lattices
- Identifying unique clusters and their symmetries

The module primarily works with pymatgen Structure objects and integrates with
networkx for graph operations.
"""

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.periodic_table import Element, DummySpecies
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer
import numpy as np
import networkx as nx
from .graph import structure_to_graph, create_connectivity_matrix

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

def find_non_equivalent_positions(structure, transition_metals):
    """
    Find non-equivalent transition metal atom positions in a crystal structure.
    
    This function identifies unique transition metal sites by considering both
    the element type and fractional coordinates. Sites are considered equivalent
    if they have the same element and identical fractional coordinates.
    
    Parameters:
        structure (Structure): A pymatgen Structure object representing the crystal
        transition_metals (list): List of transition metal element symbols to search for
        
    Returns:
        list: List of unique transition metal sites (as pymatgen Site objects)
    
    Example:
        >>> structure = Structure.from_file("crystal.cif")
        >>> transition_metals = ["Fe", "Co", "Ni"]
        >>> unique_sites = find_non_equivalent_positions(structure, transition_metals)
    """
    unique_sites = []
    seen = set()  # Set to store unique (symbol, fractional coordinates) tuples

    for site in structure:
        if hasattr(site.specie, 'symbol') and site.specie.symbol in transition_metals:
            site_key = (site.specie.symbol, tuple(site.frac_coords))
            if site_key not in seen:
                seen.add(site_key)
                unique_sites.append(site)

    return unique_sites


def create_connectivity_matrix(structure, transition_metals, cutoff=3.5):
    """
    Create a connectivity matrix for transition metal atoms in a structure.
    
    The connectivity matrix is a symmetric binary matrix where entry (i,j) is 1
    if atoms i and j are within the cutoff distance of each other, and 0 otherwise.
    Only transition metal atoms specified in the transition_metals list are considered.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        transition_metals (list): List of transition metal element symbols to consider
        cutoff (float): Maximum distance (in Angstroms) for considering atoms as connected
        
    Returns:
        tuple: (connectivity_matrix, tm_indices) where:
            - connectivity_matrix (numpy.ndarray): Binary matrix of connections
            - tm_indices (list): Indices of transition metal atoms in the structure
    
    Example:
        >>> matrix, indices = create_connectivity_matrix(structure, ["Fe"], 3.5)
        >>> print(f"Found {len(indices)} Fe atoms")
    """
    tm_indices = [i for i, site in enumerate(structure) 
                 if hasattr(site.specie, 'symbol') and site.specie.symbol in transition_metals]
    
    n = len(tm_indices)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i+1, n):
            if structure[tm_indices[i]].distance(structure[tm_indices[j]]) <= cutoff:
                matrix[i, j] = matrix[j, i] = 1
                
    return matrix, tm_indices


def structure_to_graph(connectivity_matrix):
    """
    Convert a connectivity matrix to a networkx graph representation.
    
    Creates an undirected graph where vertices represent atoms and edges
    represent connections between atoms (as defined by the connectivity matrix).
    
    Parameters:
        connectivity_matrix (numpy.ndarray): Square binary matrix where 1 indicates
                                          a connection between atoms i and j
        
    Returns:
        networkx.Graph: Graph representation where:
            - Nodes are integers representing atom indices
            - Edges connect atoms that are within the cutoff distance
    
    Example:
        >>> matrix = np.array([[0, 1], [1, 0]])  # Two connected atoms
        >>> graph = structure_to_graph(matrix)
        >>> print(f"Graph has {graph.number_of_nodes()} nodes")
    """
    G = nx.Graph()
    for i in range(len(connectivity_matrix)):
        for j in range(len(connectivity_matrix)):
            if connectivity_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G


def generate_supercell(structure, supercell_matrix=(2, 2, 2)):
    """
    Generate a supercell from a crystal structure.
    
    Creates an expanded unit cell by replicating the original structure
    along each crystallographic direction according to the supercell_matrix.
    
    Parameters:
        structure (Structure): A pymatgen Structure object to expand
        supercell_matrix (tuple): Three integers specifying the expansion
                                along each lattice vector (a, b, c)
        
    Returns:
        Structure: New pymatgen Structure representing the supercell
    
    Example:
        >>> supercell = generate_supercell(structure, (3, 3, 3))
        >>> print(f"Original atoms: {len(structure)}")
        >>> print(f"Supercell atoms: {len(supercell)}")
    """
    return structure.make_supercell(supercell_matrix)


def identify_unique_clusters(clusters, tolerance=1e-5):
    """
    Identify unique clusters and assign them distinct labels.
    
    Clusters are considered unique based on:
    1. Number of atoms (size)
    2. Average interatomic distance
    3. Chemical composition (element types)
    
    Parameters:
        clusters (list[dict]): List of cluster dictionaries, each containing:
            - "sites": list of pymatgen Site objects
            - "size": number of sites in the cluster
            - "average_distance": average distance between sites
            - "label" (optional): existing label for the cluster
        tolerance (float): Distance threshold for considering clusters as unique
    
    Returns:
        list[dict]: List of unique clusters with assigned labels ("X0", "X1", etc.)
    
    Example:
        >>> unique = identify_unique_clusters(clusters)
        >>> print(f"Found {len(unique)} unique cluster types")
    """
    unique_clusters = []
    seen_clusters = set()

    for i, cluster in enumerate(clusters):
        # Create a unique identifier for the cluster based on its properties
        cluster_key = (
            cluster["size"],
            round(cluster["average_distance"], 3),
            tuple(sorted(site.specie.symbol for site in cluster["sites"]))
        )

        if cluster_key not in seen_clusters:
            seen_clusters.add(cluster_key)
            # Add a label to the cluster
            cluster_with_label = cluster.copy()
            if "label" not in cluster_with_label:
                cluster_with_label["label"] = f"X{len(unique_clusters)}"
            unique_clusters.append(cluster_with_label)

    return unique_clusters


def generate_lattice_with_clusters(structure, clusters, tolerance=1e-5):
    """
    Generate a new crystal structure using transition metal clusters as lattice sites.
    
    This function:
    1. Identifies unique cluster types
    2. Computes cluster centroids
    3. Creates a new structure with dummy atoms representing clusters
    4. Analyzes symmetry of the resulting structure
    5. Determines point group symmetry of each cluster
    
    Parameters:
        structure (Structure): The original crystal structure
        clusters (list[dict]): List of cluster dictionaries, each containing:
            - "sites": list of pymatgen Site objects
        tolerance (float): Distance threshold for considering clusters as unique
    
    Returns:
        tuple: (conventional_structure, space_group_symbol, point_groups) where:
            - conventional_structure: Structure with clusters as lattice sites
            - space_group_symbol: Space group symbol of the new structure
            - point_groups: Dictionary mapping cluster labels to point group symbols
    
    Example:
        >>> conv_struct, sg, pg = generate_lattice_with_clusters(structure, clusters)
        >>> print(f"Space group: {sg}")
        >>> print(f"Cluster point groups: {pg}")
    
    Notes:
        - Each unique cluster type is represented by a distinct DummySpecies
        - The function attempts to find the conventional cell representation
        - If symmetry analysis fails, returns the primitive cell with error info
    """
    # Extract the lattice from the original structure
    lattice = structure.lattice

    # Identify clusters and assign unique labels
    unique_clusters = identify_unique_clusters(clusters)

    cluster_sites = []    # To store unique fractional coordinates
    species_labels = []   # To store DummySpecies labels
    point_groups = {}     # To store point groups for each unique cluster

    for cluster in unique_clusters:
        # Compute the centroid using fractional coordinates
        centroid = calculate_centroid(cluster["sites"], lattice)
        # Ensure fractional coordinates are within [0,1)
        fractional_coords = np.mod(lattice.get_fractional_coords(centroid), 1.0)

        cluster_sites.append(fractional_coords)
        # Assign a DummySpecies label based on the cluster's unique identifier
        species_labels.append(DummySpecies(cluster["label"]))
        # Extract species and Cartesian coordinates from the cluster's sites
        species = [site.specie for site in cluster["sites"]]
        cartesian_coords = [site.coords for site in cluster["sites"]]
        # Create a Molecule object
        molecule = Molecule(species, cartesian_coords)

        # Analyze the point group of the molecule
        pga = PointGroupAnalyzer(molecule)
        point_group = pga.get_pointgroup()

        # Store the point group information
        point_groups[cluster["label"]] = point_group.sch_symbol

    # Create a new structure with labeled DummySpecies for each cluster type
    primitive_structure = Structure(lattice, species_labels, cluster_sites)

    try:
        sg_analyzer = SpacegroupAnalyzer(primitive_structure)
        conventional_structure = sg_analyzer.get_conventional_standard_structure()
        space_group_symbol = sg_analyzer.get_space_group_symbol()
    except Exception as e:
        conventional_structure = primitive_structure
        space_group_symbol = "Symmetry Not Determined"
        point_groups = {"error": str(e)}

    return conventional_structure, space_group_symbol, point_groups