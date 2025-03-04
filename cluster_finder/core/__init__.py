"""
Core functionality for cluster_finder package.

This module contains the core functionality for finding and analyzing clusters.
"""

# Import from structure module
from .structure import (
    find_non_equivalent_positions,
    generate_supercell,
    generate_lattice_with_clusters
)

# Import from clusters module
from .clusters import (
    find_clusters,
    calculate_average_distance,
    calculate_centroid,
    build_graph,
    split_cluster,
    analyze_clusters,
    identify_unique_clusters
)

# Import from simple_search module
from .simple_search import (
    find_trimers,
    analyze_compound_trimers,
    print_trimer_results,
    search_and_analyze_trimers
)

# Import from graph module
from .graph import (
    structure_to_graph,
    create_connectivity_matrix
)

__all__ = [
    'calculate_centroid',
    'find_non_equivalent_positions',
    'generate_supercell',
    'generate_lattice_with_clusters',
    'find_clusters',
    'calculate_average_distance',
    'build_graph',
    'split_cluster',
    'analyze_clusters',
    'identify_unique_clusters',
    'structure_to_graph',
    'create_connectivity_matrix'
]
