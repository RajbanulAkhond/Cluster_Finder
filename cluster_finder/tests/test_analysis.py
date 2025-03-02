"""
Tests for analysis functionality of the cluster_finder package.
"""

import pytest
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from cluster_finder.analysis.analysis import (
    get_point_group_order,
    get_space_group_order,
    classify_dimensionality,
    cluster_compounds_dataframe,
    rank_clusters
)
from cluster_finder.core.structure import calculate_centroid


class TestAnalysis:
    """Test functions in analysis.py"""
    
    def test_get_point_group_order(self):
        """Test getting point group order."""
        # Test common point groups
        assert get_point_group_order("1") == 1
        assert get_point_group_order("2") == 2
        assert get_point_group_order("mmm") == 8
        assert get_point_group_order("m-3m") == 48
        
        # Test unknown point group
        assert get_point_group_order("unknown") == 0
    
    def test_get_space_group_order(self):
        """Test getting space group order."""
        # Test common space groups
        assert get_space_group_order("P1") == 1
        assert get_space_group_order("P21/c") == 0  # Not directly in our simple mapping
        
        # Test unknown space group
        assert get_space_group_order("unknown") == 0
    
    def test_classify_dimensionality(self, simple_cubic_structure):
        """Test classifying dimensionality."""
        # Simple cubic structure should be 3D
        dim = classify_dimensionality(simple_cubic_structure)
        assert dim == "3D"
        
        # Test with very small cutoff (should be 0D)
        dim = classify_dimensionality(simple_cubic_structure, distance_cutoff=1.0)
        assert dim == "0D"
    
    def test_cluster_compounds_dataframe(self, simple_cubic_structure, sample_cluster):
        """Test creating cluster compounds dataframe."""
        # Create a single cluster
        centroid = calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
        cluster = {
            "sites": sample_cluster,
            "size": 3,
            "average_distance": 3.0,
            "centroid": centroid
        }
        
        # Create a compound with the cluster
        compound = {
            "material_id": "test_id",
            "formula": "Fe8",
            "total_magnetization": 8.0,
            "clusters": [cluster],
            "structure": simple_cubic_structure
        }
        
        # Test dataframe creation
        df = cluster_compounds_dataframe([compound], "test_system")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["material_id"] == "test_id"
        assert df.iloc[0]["formula"] == "Fe8"
        assert df.iloc[0]["total_magnetization"] == 8.0
        assert df.iloc[0]["num_clusters"] == 1
        assert df.iloc[0]["compound_system"] == "test_system"
    
    def test_rank_clusters(self):
        """Test ranking clusters."""
        # Create a simple dataframe
        data = {
            "material_id": ["id1", "id2", "id3"],
            "space_group": ["P1", "Fm-3m", "P1"],
            "point_group": ["1", "m-3m", "1"],
            "cluster_sizes": [[2, 3], [4], [2, 2, 2]],
            "average_distance": [[2.5, 3.0], [2.2], [3.1, 3.2, 3.3]]
        }
        df = pd.DataFrame(data)
        
        # Test ranking
        ranked_df = rank_clusters(df)
        
        assert isinstance(ranked_df, pd.DataFrame)
        assert len(ranked_df) == 3
        
        # The order should be: id2 (min_avg_distance=2.2), id1 (min_avg_distance=2.5), id3 (min_avg_distance=3.1)
        assert ranked_df.iloc[0]["material_id"] == "id2" 