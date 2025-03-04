"""
Tests for utility functions in the cluster_finder package.
"""

import pytest
from unittest.mock import MagicMock
from pymatgen.core.structure import Structure
from pymatgen.core.sites import Site
from pymatgen.core.lattice import Lattice
from cluster_finder.core.utils import cluster_summary_stat

@pytest.fixture
def mock_compounds_with_clusters():
    """Create mock compounds with clusters for testing."""
    # Create mock sites with proper attributes
    site1 = MagicMock()
    site1.specie.symbol = "Fe"
    site1.frac_coords = [0, 0, 0]
    
    site2 = MagicMock()
    site2.specie.symbol = "Fe"
    site2.frac_coords = [0.5, 0, 0]
    
    site3 = MagicMock()
    site3.specie.symbol = "Fe"
    site3.frac_coords = [0, 0.5, 0]
    
    # Create mock clusters
    cluster1 = {
        "sites": [site1, site2],
        "size": 2,
        "average_distance": 2.5
    }
    
    cluster2 = {
        "sites": [site2, site3],
        "size": 2,
        "average_distance": 2.8
    }
    
    # Create mock compounds
    compounds = [
        {
            "material_id": "mp-123",
            "formula": "Fe2O3",
            "total_magnetization": 5.0,
            "clusters": [cluster1, cluster2]
        },
        {
            "material_id": "mp-456",
            "formula": "Co3O4",
            "total_magnetization": 3.0,
            "clusters": [cluster1]
        },
        {
            "material_id": "mp-789",
            "formula": "NiO",
            "total_magnetization": 2.0,
            "clusters": []
        }
    ]
    
    return compounds

@pytest.fixture
def mock_entries():
    """Create mock entries for testing."""
    return [MagicMock() for _ in range(3)]

class TestUtils:
    """Test functions in utils.py"""
    
    def test_cluster_summary_stat(self, mock_compounds_with_clusters, mock_entries):
        """Test generating cluster analysis summary."""
        # Generate summary
        summary = cluster_summary_stat(mock_compounds_with_clusters, mock_entries)
        
        # Check that summary is a string
        assert isinstance(summary, str)
        
        # Check that summary contains expected information
        assert "Summary of Cluster Analysis" in summary
        assert "Total Compounds Analyzed: 3" in summary
        assert "Compounds with Clusters: 2" in summary
        
        # Check compound details
        assert "Material ID: mp-123" in summary
        assert "Formula: Fe2O3" in summary
        assert "Total Magnetization: 5.0" in summary
        assert "Number of Clusters: 2" in summary
        
        # Check cluster details
        assert "Size: 2" in summary
        assert "Average Distance: 2.50 Å" in summary
        assert "Average Distance: 2.80 Å" in summary
        
        # Check site details
        assert "Element: Fe, Position: [0, 0, 0]" in summary
        assert "Element: Fe, Position: [0.5, 0, 0]" in summary
        
        # Check sorting order (compounds should be sorted by number of clusters)
        first_compound_idx = summary.find("Material ID: mp-123")
        second_compound_idx = summary.find("Material ID: mp-456")
        third_compound_idx = summary.find("Material ID: mp-789")
        
        assert first_compound_idx < second_compound_idx < third_compound_idx
        
        # Check compounds with no clusters
        assert "No clusters found in this structure" in summary
    
    def test_cluster_summary_stat_empty_input(self):
        """Test generating summary with empty input."""
        summary = cluster_summary_stat([], [])
        
        assert isinstance(summary, str)
        assert "Total Compounds Analyzed: 0" in summary
        assert "Compounds with Clusters: 0" in summary
    
    def test_cluster_summary_stat_invalid_input(self):
        """Test generating summary with invalid input."""
        # Test with None values
        with pytest.raises(TypeError, match="Input parameters cannot be None"):
            cluster_summary_stat(None, None)
        
        # Test with non-list inputs
        with pytest.raises(TypeError, match="Input parameters must be lists"):
            cluster_summary_stat("not a list", [])
        
        # Test with invalid compound structure
        invalid_compounds = [{"invalid": "data"}]
        with pytest.raises(KeyError):
            cluster_summary_stat(invalid_compounds, [MagicMock()]) 