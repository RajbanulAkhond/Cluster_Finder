import pytest
from unittest.mock import patch, MagicMock
from cluster_finder.core.simple_search import (
    find_trimers,
    analyze_compound_trimers,
    search_and_analyze_trimers
)


class TestSimpleSearch:
    """Test functions in simple_search.py"""

    @patch('cluster_finder.utils.helpers.calculate_metal_distances')
    def test_find_trimers(self, mock_calculate_metal_distances):
        """Test finding trimers in a structure."""
        # Mock the calculate_metal_distances function
        mock_calculate_metal_distances.return_value = {'Fe': 2.5}

        # Create a mock structure and unique sites
        mock_structure = MagicMock()
        mock_unique_sites = [MagicMock() for _ in range(3)]
        mock_transition_metals = ['Fe']

        # Mock the get_neighbors method
        for site in mock_unique_sites:
            site.get_neighbors.return_value = [MagicMock(specie=MagicMock(symbol='Fe'))]

        # Call the function
        trimers = find_trimers(mock_structure, mock_unique_sites, {'Fe': 2.5}, mock_transition_metals)

        # Assert that trimers are found
        assert len(trimers) == 3

    @patch('cluster_finder.utils.helpers.search_transition_metal_compounds')
    @patch('cluster_finder.utils.helpers.calculate_metal_distances')
    def test_search_and_analyze_trimers(self, mock_calculate_metal_distances, mock_search_transition_metal_compounds):
        """Test searching and analyzing trimers."""
        # Mock the calculate_metal_distances function
        mock_calculate_metal_distances.return_value = {'Fe': 2.5}

        # Mock the search_transition_metal_compounds function
        mock_entry = MagicMock()
        mock_entry.structure = MagicMock()
        mock_entry.material_id = 'mp-1234'
        mock_entry.formula_pretty = 'FeO'
        mock_search_transition_metal_compounds.return_value = [mock_entry]

        # Call the function
        results = search_and_analyze_trimers(['Fe'], 'fake_api_key')

        # Assert that results are returned
        assert len(results) == 1
        assert results[0]['material_id'] == 'mp-1234'
        assert results[0]['formula'] == 'FeO' 