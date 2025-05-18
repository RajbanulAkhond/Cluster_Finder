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

# Apply monkey patch to fix PyMatGen's infinite loop issue
try:
    import sys
    import warnings
    import importlib.util
    import numpy as np
    from pymatgen.core.operations import SymmOp
    from pymatgen.symmetry.analyzer import generate_full_symmops
    from .utils.logger import logger

    # Flag to track if we've already logged the patch message
    _patch_logged = False

    # Filter to suppress the specific warning we're patching
    warnings.filterwarnings(
        "ignore", 
        message=".*matrices have been generated. The tol may be too small.*",
        category=UserWarning, 
        module="pymatgen.symmetry.analyzer"
    )

    # Store the original function to maintain backward compatibility
    _original_generate_full_symmops = generate_full_symmops

    # Define the patched function
    def patched_generate_full_symmops(symmops, tol):
        """Patched version of generate_full_symmops that fixes infinite loop issues.
        This monkey patch ensures that the function properly returns when too many 
        matrices are generated, preventing infinite loops due to symmetry tolerance issues.
        """
        global _patch_logged
        
        # Uses an algorithm described in:
        # Gregory Butler. Fundamental Algorithms for Permutation Groups.
        # Lecture Notes in Computer Science (Book 559). Springer, 1991. page 15
        identity = np.eye(4)
        generators = [op.affine_matrix for op in symmops if not np.allclose(op.affine_matrix, identity)]
        if not generators:
            # C1 symmetry breaks assumptions in the algorithm afterwards
            return symmops

        full = list(generators)

        for g in full:
            for s in generators:
                op = np.dot(g, s)
                d = np.abs(full - op) < tol
                if not np.any(np.all(np.all(d, axis=2), axis=1)):
                    full.append(op)
                if len(full) > 1000:
                    # We're silencing the UserWarning but still adding our own debug log
                    # Don't use warnings.warn here as we've filtered these messages
                    if not _patch_logged:
                        _patch_logged = True
                        logger.debug(f"Returning early from generate_full_symmops with {len(full)} matrices due to patched function")
                    
                    # Fix: Return when too many matrices are generated
                    return [SymmOp(op) for op in full]

        d = np.abs(full - identity) < tol
        if not np.any(np.all(np.all(d, axis=2), axis=1)):
            full.append(identity)
        return [SymmOp(op) for op in full]

    # Apply the monkey patch
    sys.modules['pymatgen.symmetry.analyzer'].generate_full_symmops = patched_generate_full_symmops
    
    # Log the patch once using the logger instead of print
    logger.info("PyMatGen's generate_full_symmops function patched to prevent infinite loops")
except Exception as e:
    # Use logger if available, otherwise fall back to warnings
    try:
        from .utils.logger import logger
        logger.warning(f"Failed to patch PyMatGen's generate_full_symmops function: {str(e)}")
    except ImportError:
        warnings.warn(f"Failed to patch PyMatGen's generate_full_symmops function: {str(e)}")

# Check system dependencies
try:
    from .utils.system_compat import check_dependency_support
    # Run system checks but don't be verbose on import - just issue warnings if needed
    check_dependency_support(verbose=True)
except ImportError:
    # The utils module might not be available during initial installation
    pass

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
