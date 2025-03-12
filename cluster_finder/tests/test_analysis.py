"""
Tests for analysis functionality of the cluster_finder package.
"""

import pytest
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pathlib import Path

from cluster_finder.analysis.postprocess import (
    get_point_group_order,
    get_space_group_order,
    classify_dimensionality,
    rank_clusters
)
from cluster_finder.analysis.dataframe import (
    cluster_compounds_dataframe
)
from cluster_finder.core.structure import calculate_centroid


class TestAnalysis:
    """Test functions in analysis.py"""
    
    def test_get_point_group_order(self):
        """Test getting point group order."""
        # Test common point groups
        assert get_point_group_order("1") == 1
        assert get_point_group_order("m-3m") == 48
        assert get_point_group_order("unknown") == 0  # Default for unknown
    
    def test_get_space_group_order(self):
        """Test getting space group order."""
        # Test common space groups
        assert get_space_group_order("P1") == 1
        assert get_space_group_order("P21/c") == 14  # P21/c is space group number 14
        assert get_space_group_order("Fm-3m") == 225  # Corrected from 192 to 225
        assert get_space_group_order("unknown") == 0  # Default for unknown
    
    def test_classify_dimensionality(self, simple_cubic_structure):
        """Test classifying dimensionality."""
        # Simple cubic structure should be 3D
        dim, scores = classify_dimensionality(simple_cubic_structure)
        assert dim == "3D"
        assert len(scores) == 3  # Should have scores for x, y, z dimensions
        assert all(score >= 0.9 for score in scores)  # All scores should be high for 3D
    
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
        # Create test data with energy_above_hull to prevent API calls
        # Using more extreme values to ensure energy_above_hull dominates the ranking
        data = {
            "material_id": ["id1", "id2", "id3"],
            "space_group": ["P1", "P1", "P1"],
            "point_group": ["1", "1", "1"],
            "cluster_sizes": ["[2]", "[2]", "[2]"],
            "average_distance": ["[2.5]", "[2.2]", "[2.4]"],
            "min_avg_distance": [2.5, 2.2, 2.4],
            "energy_above_hull": [0.3, 0.2, 0.01]  # More extreme difference for id3
        }
        df = pd.DataFrame(data)
        
        # Rank clusters
        ranked_df = rank_clusters(df)
        
        # Check that we have the expected columns
        assert "max_point_group_order" in ranked_df.columns
        assert "space_group_order" in ranked_df.columns
        assert "rank_score" in ranked_df.columns
        
        # Check that ranking is correct
        # With the default weights, energy_above_hull has higher weight (-2.0) than min_avg_distance (-1.0)
        # id3 should be ranked highest due to much lower energy_above_hull value
        assert ranked_df.iloc[0]["material_id"] == "id3"  # Lowest energy_above_hull
    
    def test_rank_clusters_with_custom_props(self):
        """Test ranking clusters with custom properties."""
        # Create test data with custom properties
        data = {
            "material_id": ["id1", "id2", "id3", "id4"],
            "space_group": ["P1", "P1", "P1", "P1"],
            "point_group": ["1", "1", "1", "1"],
            "cluster_sizes": ["[2]", "[2]", "[2]", "[2]"],
            "average_distance": ["[2.5]", "[2.2]", "[3.1]", "[2.8]"],
            "min_avg_distance": [2.5, 2.2, 3.1, 2.8],
            "energy_above_hull": [0.1, 0.3, 0.05, 0.2],
            "band_gap": [1.5, 0.8, 2.2, 1.9],
            "density": [5.2, 6.1, 4.8, 5.5]
        }
        df = pd.DataFrame(data)
        
        # Test with single custom property (band_gap, higher value is better)
        ranked_df = rank_clusters(
            df,
            custom_props=["band_gap"],
            prop_weights={"band_gap": 10.0}  # Use a high weight to ensure it dominates
        )
        
        # Check that ranking is influenced by band_gap (higher band_gap should rank higher)
        assert ranked_df.iloc[0]["material_id"] == "id3"  # Highest band_gap (2.2)
        
        # Test with two custom properties and weights
        # band_gap (positive weight) and density (negative weight)
        ranked_df = rank_clusters(
            df,
            custom_props=["band_gap", "density"],
            prop_weights={"band_gap": 10.0, "density": -10.0}  # High weights to ensure they dominate
        )
        
        # Verify that the ranking is influenced by both properties
        # id3 has highest band_gap (2.2) and lowest density (4.8), making it optimal
        assert ranked_df.iloc[0]["material_id"] == "id3"
    
    def test_rank_clusters_without_default_ranking(self):
        """Test ranking clusters with custom properties only (no default ranking)."""
        # Create test data
        data = {
            "material_id": ["id1", "id2", "id3"],
            "space_group": ["P1", "P1", "P1"],
            "point_group": ["1", "1", "1"],
            "cluster_sizes": ["[2]", "[2]", "[2]"],
            "average_distance": ["[2.5]", "[2.2]", "[3.1]"],
            "min_avg_distance": [2.5, 2.2, 3.1],
            "formation_energy_per_atom": [-1.5, -0.8, -2.2]  # Lower is better
        }
        df = pd.DataFrame(data)
        
        # Rank using only formation_energy_per_atom (lower is better, so use negative weight)
        ranked_df = rank_clusters(
            df,
            custom_props=["formation_energy_per_atom"],
            prop_weights={"formation_energy_per_atom": -1.0},
            include_default_ranking=False
        )
        
        # Check that ranking is based only on formation_energy_per_atom
        assert ranked_df.iloc[0]["material_id"] == "id3"  # Lowest formation_energy_per_atom
        assert ranked_df.iloc[-1]["material_id"] == "id2"  # Highest formation_energy_per_atom
    
    def test_rank_clusters_with_property_weights(self):
        """Test that property weights properly influence the ranking."""
        # Create a completely new test case with simplified data to isolate the weight effect
        # Use data where min_avg_distance varies significantly but energy_above_hull is constant
        data = {
            "material_id": ["id1", "id2", "id3"],
            "space_group": ["P1", "P1", "P1"],
            "point_group": ["1", "1", "1"], 
            "average_distance": ["[2.5]", "[1.0]", "[3.0]"],  # id2 has much lower distance
            "min_avg_distance": [2.5, 1.0, 3.0],
            "energy_above_hull": [0.1, 0.1, 0.1]  # All equal energy_above_hull
        }
        df = pd.DataFrame(data)
        
        # With default weights, min_avg_distance should determine ranking since energy_above_hull is equal
        ranked_df_default = rank_clusters(df)
        
        # id2 should be ranked highest (lowest min_avg_distance)
        assert ranked_df_default.iloc[0]["material_id"] == "id2"
        
        # Now test with a modified dataset where energy difference should override distance
        data2 = {
            "material_id": ["id1", "id2", "id3"],
            "space_group": ["P1", "P1", "P1"],
            "point_group": ["1", "1", "1"], 
            "average_distance": ["[2.5]", "[1.0]", "[3.0]"],  # id2 has much lower distance
            "min_avg_distance": [2.5, 1.0, 3.0],
            "energy_above_hull": [0.3, 0.2, 0.01]  # id3 has much lower energy (more stable)
        }
        df2 = pd.DataFrame(data2)
        
        # Use extreme weights to ensure min_avg_distance dominates energy_above_hull
        prop_weights = {
            "min_avg_distance": -100.0,  # Very strong negative weight (lower is better)
            "energy_above_hull": -0.1    # Very weak negative weight (lower is better)
        }
        
        # Use custom weights to force distance-based ranking
        ranked_df_custom = rank_clusters(df2, prop_weights=prop_weights)
        
        # Despite id3 having lowest energy, id2 should rank highest due to extreme min_avg_distance weight
        assert ranked_df_custom.iloc[0]["material_id"] == "id2"