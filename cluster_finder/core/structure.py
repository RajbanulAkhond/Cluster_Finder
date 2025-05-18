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
from .clusters import identify_unique_clusters
from .utils import calculate_centroid

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
    # Create a copy of the structure to avoid modifying the original
    structure_copy = structure.copy()
    return structure_copy.make_supercell(supercell_matrix)

def generate_lattice_with_clusters(structure, clusters, tolerance=1e-5):
    """
    Generate a new crystal structure using transition metal clusters as lattice sites.
    
    This function:
    1. Identifies unique cluster types
    2. Uses existing centroids or computes cluster centroids if not provided
    3. Creates a new structure with dummy atoms representing clusters
    4. Analyzes symmetry of the resulting structure
    5. Determines point group symmetry of each cluster
    
    Parameters:
        structure (Structure): The original crystal structure
        clusters (list[dict]): List of cluster dictionaries, each containing:
            - "sites": list of pymatgen Site objects
            - "centroid": (optional) pre-calculated centroid coordinates
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

    # Identify clusters and assign unique labels using the imported function
    
    unique_clusters = identify_unique_clusters(clusters, tolerance=tolerance)

    cluster_sites = []    # To store unique fractional coordinates
    species_labels = []   # To store DummySpecies labels
    point_groups = {}     # To store point groups for each unique cluster

    for cluster in unique_clusters:
        # Check if centroid is already provided in the cluster data
        if "centroid" in cluster:
            centroid = cluster["centroid"]
        else:
            # Compute the centroid using fractional coordinates only if not provided
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
        sg_analyzer_c = SpacegroupAnalyzer(conventional_structure)
        space_group_symbol = sg_analyzer_c.get_space_group_symbol()
    except Exception as e:
        conventional_structure = primitive_structure
        space_group_symbol = "Symmetry Not Determined"
        point_groups = {"error": str(e)}

    return conventional_structure, space_group_symbol, point_groups