"""
Analysis module for cluster_finder package.

This module contains functions for analyzing, classifying, and ranking clusters.
"""

from .postprocess import (
    get_point_group_order,
    get_space_group_order,
    classify_dimensionality,
    rank_clusters
)

from .dataframe import (
    cluster_compounds_dataframe,
    postprocessed_clusters_dataframe,
    postprocess_clusters  # For backward compatibility
)

__all__ = [
    'get_point_group_order',
    'get_space_group_order',
    'classify_dimensionality',
    'rank_clusters',
    'cluster_compounds_dataframe',
    'postprocessed_clusters_dataframe',
    'postprocess_clusters'  # For backward compatibility
]
