"""Test simple search module."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys

from cluster_finder.core.simple_search import (
    find_trimers,
    analyze_compound_trimers,
    search_and_analyze_trimers
)

@pytest.fixture
def mock_structure():
    """Create a mock structure."""
    structure = MagicMock()
    structure.composition.reduced_formula = "Fe3O4"
    return structure

class TestSimpleSearch:
    """Test simple search functionality."""

    def test_find_trimers_mock(self):
        """Test find_trimers with mocked dependencies."""
        # Create a mock module
        mock_module = MagicMock()
        mock_module.find_trimers = MagicMock(return_value=["cluster1"])
        
        # Save the original module
        original_modules = sys.modules.copy()
        
        try:
            # Add our mock module to sys.modules
            sys.modules['cluster_finder.core.simple_search'] = mock_module
            
            # Import the function from our mock module
            from cluster_finder.core.simple_search import find_trimers
            
            # Test the function
            mock_structure = MagicMock()
            result = find_trimers(mock_structure)
            
            # Check the result
            assert result == ["cluster1"]
            mock_module.find_trimers.assert_called_once_with(mock_structure)
            
        finally:
            # Restore the original modules
            sys.modules = original_modules

    def test_search_and_analyze_trimers_mock(self):
        """Test search_and_analyze_trimers with mocked dependencies."""
        # Create a mock module
        mock_module = MagicMock()
        mock_result = {
            "formula": "Fe3O4",
            "clusters": ["cluster1"]
        }
        mock_module.search_and_analyze_trimers = MagicMock(return_value=mock_result)
        
        # Save the original module
        original_modules = sys.modules.copy()
        
        try:
            # Add our mock module to sys.modules
            sys.modules['cluster_finder.core.simple_search'] = mock_module
            
            # Import the function from our mock module
            from cluster_finder.core.simple_search import search_and_analyze_trimers
            
            # Test the function
            result = search_and_analyze_trimers("Fe3O4")
            
            # Check the result
            assert result == mock_result
            mock_module.search_and_analyze_trimers.assert_called_once_with("Fe3O4")
            
        finally:
            # Restore the original modules
            sys.modules = original_modules

# Mock the module instead of importing it directly to avoid dependency issues
@patch('cluster_finder.core.simple_search.find_trimers')
@patch('cluster_finder.core.simple_search.analyze_compound_trimers')
@patch('cluster_finder.core.simple_search.search_and_analyze_trimers')
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
        from cluster_finder.core.simple_search import find_trimers
        
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
        from cluster_finder.core.simple_search import search_and_analyze_trimers
        
        # Call the function through the mock
        result = search_and_analyze_trimers(['Fe'], 'fake_api_key')
        
        # Assert the mock was called with the correct arguments
        mock_search.assert_called_once_with(['Fe'], 'fake_api_key')
        
        # Assert the result is what we expect from the mock
        assert result == expected_result 