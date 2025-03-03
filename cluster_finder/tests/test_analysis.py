"""
Tests for analysis functionality of the cluster_finder package.
"""

import pytest
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pathlib import Path

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
        """Test creating dataframe from compounds with clusters."""
        # Create test data
        compounds = [{
            "material_id": "test-1",
            "formula": "Fe2O3",
            "total_magnetization": 5.0,
            "structure": simple_cubic_structure,
            "clusters": [{
                "sites": sample_cluster,
                "size": 3,
                "average_distance": 2.5,
                "centroid": [0, 0, 0]
            }]
        }]
        
        # Create dataframe
        df = cluster_compounds_dataframe(compounds, "Fe-O")
        
        # Check basic properties
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["material_id"] == "test-1"
        assert df.iloc[0]["formula"] == "Fe2O3"
        assert df.iloc[0]["compound_system"] == "Fe-O"
        assert df.iloc[0]["cluster_sizes"] == [3]
        assert df.iloc[0]["num_clusters"] == 1
    
    def test_rank_clusters(self):
        """Test ranking clusters."""
        # Create test data
        data = {
            "material_id": ["id1", "id2", "id3"],
            "space_group": ["P1", "P1", "P1"],
            "point_group": ["1", "1", "1"],
            "cluster_sizes": ["[2]", "[2]", "[2]"],
            "average_distance": ["[2.5]", "[2.2]", "[3.1]"]
        }
        df = pd.DataFrame(data)
        
        # Rank clusters
        ranked_df = rank_clusters(df)
        
        # Check results
        assert isinstance(ranked_df, pd.DataFrame)
        assert len(ranked_df) == 3
        
        # The order should be: id2 (min_avg_distance=2.2), id1 (min_avg_distance=2.5), id3 (min_avg_distance=3.1)
        assert ranked_df.iloc[0]["material_id"] == "id2" 