"""
Core functionality for cluster_finder package.
"""

# Import from structure module
from .structure import (
    find_non_equivalent_positions,
    create_connectivity_matrix,
    structure_to_graph,
    generate_supercell
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
    find_trimers_in_materials
)
