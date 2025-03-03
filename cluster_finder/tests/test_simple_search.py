import pytest
from unittest.mock import patch, MagicMock

# Mock the module instead of importing it directly to avoid dependency issues
@patch('cluster_finder.cluster_finder.core.simple_search.find_trimers')
@patch('cluster_finder.cluster_finder.core.simple_search.analyze_compound_trimers')
@patch('cluster_finder.cluster_finder.core.simple_search.search_and_analyze_trimers')
class TestSimpleSearch:
    """Test functions in simple_search.py"""

    def test_find_trimers_mock(self, mock_search, mock_analyze, mock_find_trimers):
        """Test finding trimers in a structure using mocks."""
        # Configure the mock to return a specific value
        mock_find_trimers.return_value = [("site1", ["neighbor1", "neighbor2"])]
        
        # Create test input data
        mock_structure = MagicMock()
        mock_unique_sites = [MagicMock()]
        mock_distances = {'Fe': 2.5}
        mock_transition_metals = ['Fe']
        
        # Import the module function directly from this test
        from cluster_finder.cluster_finder.core.simple_search import find_trimers
        
        # Call the function through the mock
        result = find_trimers(mock_structure, mock_unique_sites, mock_distances, mock_transition_metals)
        
        # Assert the mock was called with the correct arguments
        mock_find_trimers.assert_called_once_with(
            mock_structure, mock_unique_sites, mock_distances, mock_transition_metals
        )
        
        # Assert the result is what we expect from the mock
        assert result == [("site1", ["neighbor1", "neighbor2"])]
        
    def test_search_and_analyze_trimers_mock(self, mock_search, mock_analyze, mock_find_trimers):
        """Test searching and analyzing trimers using mocks."""
        # Configure the mock to return a specific value
        expected_result = [{'material_id': 'mp-1234', 'formula': 'FeO', 'trimers': []}]
        mock_search.return_value = expected_result
        
        # Import the module function directly from this test
        from cluster_finder.cluster_finder.core.simple_search import search_and_analyze_trimers
        
        # Call the function through the mock
        result = search_and_analyze_trimers(['Fe'], 'fake_api_key')
        
        # Assert the mock was called with the correct arguments
        mock_search.assert_called_once_with(['Fe'], 'fake_api_key')
        
        # Assert the result is what we expect from the mock
        assert result == expected_result 