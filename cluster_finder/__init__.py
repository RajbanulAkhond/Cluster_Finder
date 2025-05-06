"""
Cluster Finder
=============

A package for finding and analyzing atomic clusters in crystal structures.

This package provides tools to identify, analyze, and visualize
clusters of atoms (particularly transition metals) in crystal structures.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

__version__ = "0.1.0"

# Import core functionality
from .core.structure import (
    find_non_equivalent_positions,
    create_connectivity_matrix,
    generate_supercell,
    structure_to_graph,
    generate_lattice_with_clusters
)

from .core.clusters import (
    find_clusters,
    calculate_average_distance,
    build_graph,
    split_cluster,
    analyze_clusters,
    identify_unique_clusters,
    calculate_centroid
)

# Import simple search functionality
from .core.simple_search import (
    find_trimers,
    analyze_compound_trimers,
    print_trimer_results,
    search_and_analyze_trimers
)

# Import visualization functionality
from .visualization.visualize import (
    visualize_graph,
    visualize_clusters_in_compound,
    visualize_cluster_lattice
)

# Import analysis functionality
from .analysis.postprocess import (
    get_point_group_order,
    get_space_group_order,
    classify_dimensionality,
    rank_clusters
)

from .analysis.dataframe import (
    cluster_compounds_dataframe,
    postprocessed_clusters_dataframe
)

# Import I/O functionality
from .io.fileio import (
    export_structure_to_cif,
    import_csv_data,
    export_csv_data
)

# Import utility functions
from .utils.helpers import (
    calculate_metal_distances,
    get_transition_metals,
    search_transition_metal_compounds,
    generate_rotation_matrix
)

# Constants
from .core.clusters import DEFAULT_MAX_RADIUS, DEFAULT_CLUSTER_SIZE

# Define what gets imported with "from cluster_finder import *"
__all__ = [
    # Constants
    'DEFAULT_MAX_RADIUS',
    'DEFAULT_CLUSTER_SIZE',
    
    # Core - Structure
    'find_non_equivalent_positions',
    'create_connectivity_matrix',
    'generate_supercell',
    'structure_to_graph',
    'generate_lattice_with_clusters',
    
    # Core - Clusters
    'calculate_centroid',
    'find_clusters',
    'calculate_average_distance',
    'build_graph',
    'split_cluster',
    'analyze_clusters',
    'identify_unique_clusters',
    
    # Visualization
    'visualize_graph',
    'visualize_clusters_in_compound',
    'visualize_cluster_lattice',
    
    # Analysis
    'get_point_group_order',
    'get_space_group_order',
    'classify_dimensionality',
    'cluster_compounds_dataframe',
    'postprocessed_clusters_dataframe',
    'rank_clusters',
    
    # I/O
    'export_structure_to_cif',
    'import_csv_data',
    'export_csv_data',
    
    # Utilities
    'calculate_metal_distances',
    'get_transition_metals',
    'search_transition_metal_compounds',
    'generate_rotation_matrix'
]
