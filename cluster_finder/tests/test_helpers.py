"""
Tests for utility helper functions of the cluster_finder package.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pymatgen.core.structure import Structure

from cluster_finder.utils.helpers import (
    get_transition_metals,
    generate_rotation_matrix,
    search_transition_metal_compounds,
    calculate_metal_distances,
    find_trimers,
    search_and_analyze_trimers
)


class TestHelpers:
    """Test functions in helpers.py"""
    
    def test_get_transition_metals(self):
        """Test getting transition metals list."""
        tms = get_transition_metals()
        
        assert isinstance(tms, list)
        assert len(tms) > 0
        assert "Fe" in tms
        assert "Co" in tms
        assert "Ni" in tms
    
    def test_generate_rotation_matrix(self):
        """Test generating rotation matrix."""
        # Test with default parameters
        matrix = generate_rotation_matrix()
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        
        # Test with custom axis and angle
        axis = np.array([1, 0, 0])  # x-axis
        angle = np.pi/4  # 45 degrees
        
        matrix = generate_rotation_matrix(axis, angle)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        
        # Test that the matrix is orthogonal (rotation matrices preserve distances)
        assert np.allclose(np.dot(matrix, matrix.T), np.eye(3), atol=1e-10)
        
        # Test that the determinant is 1 (preserves orientation)
        assert np.isclose(np.linalg.det(matrix), 1.0)
    
    # Note: We're not testing functions that require MP API access directly
    # Instead, we'll mock them in the following tests
    
    @patch('cluster_finder.utils.helpers.MPRester')
    def test_calculate_metal_distances_mock(self, mock_mprester):
        """Test calculating metal distances using mocked MPRester."""
        from cluster_finder.utils.helpers import calculate_metal_distances
        from pymatgen.core.structure import Structure

        # Create a mock structure and result
        mock_structure = MagicMock()
        mock_structure.__iter__.return_value = [MagicMock(), MagicMock()]

        mock_site1 = MagicMock()
        mock_site2 = MagicMock()
        mock_site1.distance.return_value = 2.5

        mock_structure.__getitem__.side_effect = [mock_site1, mock_site2]

        # Set up mock MPRester instance
        mock_mpr_instance = MagicMock()
        mock_mprester.return_value.__enter__.return_value = mock_mpr_instance

        # Set up mock search results
        mock_result = MagicMock()
        mock_result.structure.as_dict.return_value = {"mock": "structure"}
        mock_mpr_instance.materials.summary.search.return_value = [mock_result]

        # Mock Structure.from_dict to return our mock structure
        with patch('pymatgen.core.structure.Structure.from_dict', return_value=mock_structure):
            # Test the function
            distances = calculate_metal_distances("Fe3O4", ["Fe"])
            assert isinstance(distances, list)
            assert len(distances) > 0
            assert distances[0] == 2.5
    
    @patch('cluster_finder.utils.helpers.MPRester')
    def test_search_transition_metal_compounds_mock(self, mock_mprester):
        """Test searching transition metal compounds using mocked MPRester."""
        from cluster_finder.utils.helpers import search_transition_metal_compounds
        
        # Set up mock MPRester instance
        mock_mpr_instance = MagicMock()
        mock_mprester.return_value.__enter__.return_value = mock_mpr_instance
        
        # Mock search result
        mock_mpr_instance.materials.summary.search.return_value = ["result1", "result2"]
        
        # Call function with mocked dependencies
        result = search_transition_metal_compounds(
            ["Fe", "Co"], 
            "fake_api_key",
            min_elements=2,
            max_elements=3
        )
        
        # Check that the function called the API correctly
        mock_mpr_instance.materials.summary.search.assert_called_once()
        
        # Check that it returned the expected results
        assert result == ["result1", "result2"]

    def test_calculate_metal_distances_mock(self):
        """Test calculate_metal_distances with mocked dependencies."""
        # Create mock structure
        mock_structure = MagicMock()
        mock_structure.get_distance.return_value = 2.5
        
        # Create mock sites
        mock_sites = [MagicMock(), MagicMock()]
        
        # Calculate distances
        distances = calculate_metal_distances(mock_structure, mock_sites)
        
        # Check result
        assert isinstance(distances, list)
        assert distances[0] == 2.5
        mock_structure.get_distance.assert_called_once()

    def test_find_trimers_mock(self):
        """Test find_trimers with mocked dependencies."""
        # Create mock structure and sites
        mock_structure = MagicMock()
        mock_structure.get_distance.return_value = 2.5
        
        mock_sites = [MagicMock() for _ in range(3)]
        
        # Mock distances to form a triangle
        distances = [2.5, 2.5, 2.5]  # All sides equal
        
        with patch('cluster_finder.utils.helpers.calculate_metal_distances', return_value=distances):
            # Find trimers
            trimers = find_trimers(mock_structure, mock_sites, max_distance=3.0)
            
            # Check result
            assert isinstance(trimers, list)
            assert len(trimers) == 1  # Should find one trimer
            assert len(trimers[0]) == 3  # Each trimer should have 3 sites

    def test_search_and_analyze_trimers_mock(self):
        """Test search_and_analyze_trimers with mocked dependencies."""
        # Create mock structure
        mock_structure = MagicMock()
        mock_structure.composition.reduced_formula = "Fe2O3"
        
        # Mock trimer results
        mock_trimers = [[MagicMock() for _ in range(3)]]
        
        with patch('cluster_finder.utils.helpers.find_trimers', return_value=mock_trimers):
            # Search and analyze trimers
            results = search_and_analyze_trimers(mock_structure, elements=["Fe"], max_distance=3.0)
            
            # Check results
            assert isinstance(results, dict)
            assert "formula" in results
            assert results["formula"] == "Fe2O3"
            assert "trimers" in results
            assert isinstance(results["trimers"], list) 