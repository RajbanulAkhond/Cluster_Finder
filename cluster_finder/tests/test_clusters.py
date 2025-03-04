"""
Tests for cluster functionality of the cluster_finder package.
"""

import pytest
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from collections import namedtuple
import networkx as nx

from cluster_finder.core.clusters import (
    get_compounds_with_clusters,
    find_clusters,
    analyze_clusters,
    calculate_centroid
)

# Create a mock Entry class for testing
MockEntry = namedtuple('MockEntry', ['material_id', 'formula_pretty', 'structure', 'total_magnetization'])

@pytest.fixture
def mock_entries(simple_cubic_structure):
    """Create mock entries for testing."""
    entries = [
        MockEntry(
            material_id="mp-123",
            formula_pretty="Fe2O3",
            structure=simple_cubic_structure,
            total_magnetization=5.0
        ),
        MockEntry(
            material_id="mp-456",
            formula_pretty="Co3O4",
            structure=simple_cubic_structure,
            total_magnetization=8.0
        )
    ]
    return entries

@pytest.fixture
def simple_ionic_structure():
    """Create a simple ionic structure without transition metals."""
    lattice = Lattice.cubic(3.0)
    coords = [
        [0.0, 0.0, 0.0],  # Na
        [0.5, 0.5, 0.5],  # Na
        [0.25, 0.25, 0.25],  # O
        [0.75, 0.75, 0.75]   # O
    ]
    species = ["Na", "Na", "O", "O"]
    return Structure(lattice, species, coords)

def test_get_compounds_with_clusters(mock_entries):
    """Test get_compounds_with_clusters function."""
    transition_metals = ["Fe", "Co"]
    
    # Get compounds with clusters
    compounds, graph, structure, tm_indices = get_compounds_with_clusters(mock_entries, transition_metals)
    
    # Check that we got the expected number of compounds
    assert len(compounds) == 2
    
    # Check the structure of the returned data
    for compound in compounds:
        assert "material_id" in compound
        assert "formula" in compound
        assert "total_magnetization" in compound
        assert "clusters" in compound
        assert "structure" in compound
        
        # Check that clusters have the expected structure
        for cluster in compound["clusters"]:
            assert "sites" in cluster
            assert "size" in cluster
            assert "average_distance" in cluster
            assert "centroid" in cluster
            
            # Check that centroid is a numpy array with 3 components
            assert isinstance(cluster["centroid"], np.ndarray)
            assert len(cluster["centroid"]) == 3
    
    # Check that graph is a networkx Graph
    assert isinstance(graph, nx.Graph)
    
    # Check that structure is a pymatgen Structure
    assert isinstance(structure, Structure)
    
    # Check that tm_indices is a list
    assert isinstance(tm_indices, list)

def test_get_compounds_with_clusters_empty_input():
    """Test get_compounds_with_clusters function with empty input."""
    transition_metals = ["Fe", "Co"]
    
    # Test with empty entries list
    compounds, graph, structure, tm_indices = get_compounds_with_clusters([], transition_metals)
    
    assert len(compounds) == 0
    assert graph is None
    assert structure is None
    assert tm_indices is None

def test_get_compounds_with_clusters_no_clusters(simple_ionic_structure):
    """Test get_compounds_with_clusters function with a structure that has no clusters."""
    # Create a mock entry with a structure that won't form clusters
    entries = [
        MockEntry(
            material_id="mp-789",
            formula_pretty="Na2O",  # No transition metals
            structure=simple_ionic_structure,
            total_magnetization=0.0
        )
    ]
    
    transition_metals = ["Fe", "Co"]  # These metals aren't in the structure
    
    # Get compounds with clusters
    compounds, graph, structure, tm_indices = get_compounds_with_clusters(entries, transition_metals)
    
    # Check that we got one compound
    assert len(compounds) == 1
    
    # Check that the compound has no clusters
    assert len(compounds[0]["clusters"]) == 0 