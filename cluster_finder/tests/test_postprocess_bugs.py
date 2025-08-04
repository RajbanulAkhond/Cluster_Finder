"""
Tests for postprocessing functionality with focus on critical bug fixes.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import ast

from cluster_finder.analysis.postprocess import (
    rank_clusters,
    get_point_group_order,
    get_highest_point_group_order,
    classify_dimensionality
)


class TestRankClustersStringParsing:
    """Test rank_clusters function with focus on safe string parsing bug fixes."""
    
    def test_rank_clusters_malformed_point_groups_string(self):
        """Test rank_clusters handles malformed point_groups strings safely."""
        # Create test DataFrame with malformed point_groups data
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456', 'mp-789'],
            'formula': ['Fe2O3', 'Co3O4', 'NiO'],
            'space_group': ['P1', 'Fm-3m', 'P63/mmc'],
            'point_groups': [
                "{'cluster_1': 'C2v'}",  # Valid
                "invalid_string_format",   # Invalid - should not crash
                "{'cluster_2': 'D4h'}"    # Valid
            ],
            'cluster_sizes': [
                "[2, 3]",  # Valid
                "[4, 5]",  # Valid  
                "[6]"      # Valid
            ],
            'average_distance': [
                "[2.5, 3.1]",  # Valid
                "[1.8]",       # Valid
                "[2.9, 3.2]"   # Valid
            ]
        })
        
        # This should not crash due to malformed string
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'rank_score' in result.columns
        
    def test_rank_clusters_malformed_cluster_sizes(self):
        """Test rank_clusters handles malformed cluster_sizes strings safely."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456'],
            'formula': ['Fe2O3', 'Co3O4'],
            'space_group': ['P1', 'Fm-3m'],
            'cluster_sizes': [
                "[2, 3]",              # Valid
                "not_a_list_at_all"    # Invalid - should not crash
            ],
            'average_distance': [
                "[2.5]",
                "[1.8]"
            ]
        })
        
        # Should handle malformed data gracefully
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        
    def test_rank_clusters_malformed_average_distance(self):
        """Test rank_clusters handles malformed average_distance strings safely."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456'],
            'formula': ['Fe2O3', 'Co3O4'],
            'space_group': ['P1', 'Fm-3m'],
            'cluster_sizes': ["[2]", "[3]"],
            'average_distance': [
                "[2.5]",               # Valid
                "completely_invalid"   # Invalid - should not crash
            ]
        })
        
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'min_avg_distance' in result.columns
        
    def test_rank_clusters_mixed_data_types(self):
        """Test rank_clusters with mixed valid and invalid data types."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456', 'mp-789'],
            'formula': ['Fe2O3', 'Co3O4', 'NiO'],
            'space_group': ['P1', 'Fm-3m', 'P63/mmc'],
            'point_groups': [
                {'cluster_1': 'C2v'},     # Already a dict (valid)
                "{'cluster_2': 'D4h'}",   # String representation (valid)
                None                      # None value (should handle)
            ],
            'cluster_sizes': [
                [2, 3],                   # Already a list (valid)
                "[4, 5]",                 # String representation (valid)
                "malformed["              # Malformed string (should handle)
            ]
        })
        
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestRankClustersNoneHandling:
    """Test rank_clusters function handles None values properly."""
    
    def test_rank_clusters_none_average_distances(self):
        """Test rank_clusters handles None values in min_avg_distance."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456'],
            'formula': ['Fe2O3', 'Co3O4'],
            'space_group': ['P1', 'Fm-3m'],
            'average_distance': [
                None,      # None value - should be handled
                "[2.5]"    # Valid value
            ],
            'cluster_sizes': ["[2]", "[3]"]
        })
        
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'min_avg_distance' in result.columns
        
        # Check that None values were handled (filled with mean or 0)
        assert not result['min_avg_distance'].isna().any()
        
    def test_rank_clusters_empty_lists(self):
        """Test rank_clusters handles empty lists in data."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456'],
            'formula': ['Fe2O3', 'Co3O4'],
            'space_group': ['P1', 'Fm-3m'],
            'cluster_sizes': [
                "[]",      # Empty list
                "[2, 3]"   # Valid list
            ],
            'average_distance': [
                "[]",      # Empty list
                "[2.5]"    # Valid list
            ]
        })
        
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestRankClustersValidation:
    """Test input validation for rank_clusters function."""
    
    def test_rank_clusters_invalid_data_source_type(self):
        """Test rank_clusters with invalid data source type."""
        with pytest.raises(TypeError, match="Data_source must be either a file path"):
            rank_clusters(123)  # Invalid type
            
    def test_rank_clusters_file_path_validation(self):
        """Test rank_clusters with non-existent file path."""
        with pytest.raises(FileNotFoundError):
            rank_clusters("non_existent_file.csv")
            
    def test_rank_clusters_empty_dataframe(self):
        """Test rank_clusters with empty DataFrame."""
        df = pd.DataFrame()
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestFilteringLogic:
    """Test the filtering logic improvements."""
    
    def test_rank_clusters_large_cluster_filtering(self):
        """Test filtering of clusters with size > 8."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456', 'mp-789'],
            'formula': ['Fe2O3', 'Co3O4', 'NiO'],
            'space_group': ['P1', 'Fm-3m', 'P63/mmc'],
            'cluster_sizes': [
                "[2, 3]",      # Valid - all clusters <= 8
                "[4, 5, 9]",   # Invalid - has cluster > 8
                "[6, 7]"       # Valid - all clusters <= 8
            ]
        })
        
        result = rank_clusters(df)
        
        # Should filter out mp-456 due to cluster size > 8
        assert len(result) == 2
        assert 'mp-456' not in result['material_id'].values
        
    def test_rank_clusters_symmetry_filtering(self):
        """Test filtering of materials with undetermined symmetry."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456', 'mp-789'],
            'formula': ['Fe2O3', 'Co3O4', 'NiO'],
            'space_group': [
                'P1',                          # Valid
                'Symmetry Not Determined',     # Invalid - should be filtered
                'Fm-3m'                        # Valid
            ],
            'cluster_sizes': ["[2]", "[3]", "[4]"]
        })
        
        result = rank_clusters(df)
        
        # Should filter out mp-456 due to undetermined symmetry
        assert len(result) == 2
        assert 'mp-456' not in result['material_id'].values


class TestPointGroupHandling:
    """Test point group order calculations and error handling."""
    
    def test_get_point_group_order_known_groups(self):
        """Test get_point_group_order with known point groups."""
        assert get_point_group_order("C1") == 1
        assert get_point_group_order("Td") == 24
        assert get_point_group_order("Oh") == 48
        
    def test_get_point_group_order_unknown_group(self):
        """Test get_point_group_order with unknown point group."""
        assert get_point_group_order("UnknownGroup") == 0
        
    def test_get_highest_point_group_order_empty_dict(self):
        """Test get_highest_point_group_order with empty dictionary."""
        assert get_highest_point_group_order({}) == 0
        
    def test_get_highest_point_group_order_mixed_groups(self):
        """Test get_highest_point_group_order with mixed point groups."""
        pg_dict = {
            'cluster_1': 'C1',   # Order 1
            'cluster_2': 'Td',   # Order 24
            'cluster_3': 'C2v'   # Order 4
        }
        assert get_highest_point_group_order(pg_dict) == 24


class TestPerformanceAndMemory:
    """Test performance and memory usage with larger datasets."""
    
    def test_rank_clusters_large_dataset_performance(self):
        """Test rank_clusters performance with larger dataset."""
        # Create a larger test dataset
        n_materials = 1000
        
        df = pd.DataFrame({
            'material_id': [f'mp-{i:05d}' for i in range(n_materials)],
            'formula': [f'Element{i % 10}' for i in range(n_materials)],
            'space_group': ['P1'] * n_materials,
            'cluster_sizes': ['[2, 3]'] * n_materials,
            'average_distance': ['[2.5, 3.1]'] * n_materials
        })
        
        # Should complete without memory issues
        result = rank_clusters(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_materials
        assert 'rank_score' in result.columns
        
    def test_rank_clusters_memory_efficiency(self):
        """Test that rank_clusters doesn't create excessive copies."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456'],
            'formula': ['Fe2O3', 'Co3O4'],
            'space_group': ['P1', 'Fm-3m']
        })
        
        original_columns = set(df.columns)
        result = rank_clusters(df)
        
        # Should not modify the original DataFrame
        assert set(df.columns) == original_columns
        
        # Result should have additional columns
        assert 'rank_score' in result.columns
        assert len(result.columns) > len(df.columns)


class TestClassifyDimensionality:
    """Test the classify_dimensionality function."""
    
    def test_classify_dimensionality_empty_structure(self):
        """Test classify_dimensionality with structure containing no sites."""
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice import Lattice
        
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, [], [])
        
        # Should handle empty structure gracefully
        try:
            result = classify_dimensionality(structure)
            # If it doesn't crash, that's good
            assert isinstance(result, tuple)
        except (ValueError, IndexError):
            # It's acceptable if it raises an appropriate error for empty input
            pass


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_ranking_pipeline(self):
        """Test complete ranking pipeline with realistic data."""
        df = pd.DataFrame({
            'material_id': ['mp-123', 'mp-456', 'mp-789'],
            'formula': ['Fe2O3', 'Co3O4', 'NiO'],
            'space_group': ['P1', 'Fm-3m', 'P63/mmc'],
            'point_groups': [
                "{'cluster_1': 'C2v', 'cluster_2': 'D4h'}",
                "{'cluster_1': 'Td'}",
                "{'cluster_1': 'Oh'}"
            ],
            'cluster_sizes': [
                "[2, 3]",
                "[4, 5]",
                "[6]"
            ],
            'average_distance': [
                "[2.5, 3.1]",
                "[1.8, 2.2]",
                "[2.9]"
            ]
        })
        
        result = rank_clusters(df, include_default_ranking=True)
        
        # Verify complete pipeline works
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'rank_score' in result.columns
        assert 'max_point_group_order' in result.columns
        assert 'space_group_order' in result.columns
        assert 'min_avg_distance' in result.columns
        
        # Verify data is sorted by rank_score
        assert result['rank_score'].is_monotonic_decreasing