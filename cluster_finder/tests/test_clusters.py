"""
Tests for cluster functionality of the cluster_finder package.
"""

import pytest
import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import Site, PeriodicSite
from collections import namedtuple
import networkx as nx

from cluster_finder.core.clusters import (
    get_compounds_with_clusters,
    find_clusters,
    analyze_clusters,
    calculate_centroid,
    identify_unique_clusters
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
    compounds = get_compounds_with_clusters(mock_entries, transition_metals)
    
    # Check that we got the expected number of compounds
    assert len(compounds) == 2
    
    # Check the structure of the returned data
    for compound in compounds:
        assert "material_id" in compound
        assert "formula" in compound
        assert "total_magnetization" in compound
        assert "clusters" in compound
        assert "structure" in compound
        assert "graph" in compound  # Now graph is inside the compound dict
        
        # Check that clusters have the expected structure
        for cluster in compound["clusters"]:
            assert "sites" in cluster
            assert "size" in cluster
            assert "average_distance" in cluster
            assert "centroid" in cluster
            
            # Check that centroid is a numpy array with 3 components
            assert isinstance(cluster["centroid"], np.ndarray)
            assert len(cluster["centroid"]) == 3
    
    # Check that graph is a networkx Graph (now contained in the compound dict)
    assert isinstance(compounds[0]["graph"], nx.Graph)
    
    # Check that structure is a pymatgen Structure
    assert isinstance(compounds[0]["structure"], Structure)

def test_get_compounds_with_clusters_empty_input():
    """Test get_compounds_with_clusters function with empty input."""
    transition_metals = ["Fe", "Co"]
    
    # Test with empty entries list
    compounds = get_compounds_with_clusters([], transition_metals)
    
    assert len(compounds) == 0

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
    compounds = get_compounds_with_clusters(entries, transition_metals)
    
    # Check that we got no compounds since there are no clusters
    assert len(compounds) == 0

def test_identify_unique_clusters_with_point_group():
    """Test identify_unique_clusters function with point group symmetry criterion."""
    # Create a mock lattice
    lattice = Lattice.cubic(5.0)

    # Create clusters with different geometries that will have different point group symmetries
    # Tetrahedral cluster (Td point group)
    sites1 = [
        PeriodicSite("Fe", [0, 0, 0], lattice),
        PeriodicSite("Fe", [1, 1, 1], lattice),
        PeriodicSite("Fe", [1, -1, -1], lattice),
        PeriodicSite("Fe", [-1, 1, -1], lattice),
        PeriodicSite("Fe", [-1, -1, 1], lattice)
    ]

    # Square planar cluster (D4h point group)
    sites2 = [
        PeriodicSite("Fe", [0, 0, 0], lattice),
        PeriodicSite("Fe", [1, 0, 0], lattice),
        PeriodicSite("Fe", [0, 1, 0], lattice),
        PeriodicSite("Fe", [-1, 0, 0], lattice),
        PeriodicSite("Fe", [0, -1, 0], lattice)
    ]

    # Octahedral cluster (Oh point group)
    sites3 = [
        PeriodicSite("Fe", [0, 0, 0], lattice),
        PeriodicSite("Fe", [1, 0, 0], lattice),
        PeriodicSite("Fe", [-1, 0, 0], lattice),
        PeriodicSite("Fe", [0, 1, 0], lattice),
        PeriodicSite("Fe", [0, -1, 0], lattice),
        PeriodicSite("Fe", [0, 0, 1], lattice),
        PeriodicSite("Fe", [0, 0, -1], lattice)
    ]

    # Calculate centroids
    centroid1 = calculate_centroid(sites1, lattice)
    centroid2 = calculate_centroid(sites2, lattice)
    centroid3 = calculate_centroid(sites3, lattice)

    # Create cluster dictionaries with relative coordinates
    cluster1 = {
        "sites": sites1,
        "size": len(sites1),
        "average_distance": 1.5,
        "centroid": centroid1,
        "relative_coords": np.array([site.frac_coords - centroid1 for site in sites1])
    }

    cluster2 = {
        "sites": sites2,
        "size": len(sites2),
        "average_distance": 1.0,
        "centroid": centroid2,
        "relative_coords": np.array([site.frac_coords - centroid2 for site in sites2])
    }

    cluster3 = {
        "sites": sites3,
        "size": len(sites3),
        "average_distance": 1.0,
        "centroid": centroid3,
        "relative_coords": np.array([site.frac_coords - centroid3 for site in sites3])
    }

    # Test with clusters that have different point group symmetries
    mock_clusters = [cluster1, cluster2, cluster3]
    labeled_clusters = identify_unique_clusters(mock_clusters, use_symmetry=True, tolerance=1e-5)

    # All three clusters should be considered unique due to their different point group symmetries
    # Instead of checking the number of unique clusters, we'll check that they all have different labels
    labels = set(cluster["label"] for cluster in labeled_clusters)
    assert len(labels) == 3

    # Check that point group information was added to each cluster
    point_groups = set()
    for cluster in labeled_clusters:
        assert "point_group" in cluster
        point_groups.add(cluster["point_group"])

    # Verify we have three different point groups
    assert len(point_groups) == 3

    # Create another tetrahedral cluster with a small translation
    # We'll translate each site by (0.1, 0.1, 0.1)
    sites4 = [
        PeriodicSite("Fe", site.frac_coords + np.array([0.1, 0.1, 0.1]), lattice)
        for site in sites1
    ]

    centroid4 = calculate_centroid(sites4, lattice)
    cluster4 = {
        "sites": sites4,
        "size": len(sites4),
        "average_distance": 1.5,
        "centroid": centroid4,
        "relative_coords": np.array([site.frac_coords - centroid4 for site in sites4])
    }

    # Test with two geometrically identical clusters at different positions
    mock_clusters = [cluster1, cluster4]
    labeled_clusters = identify_unique_clusters(mock_clusters, use_symmetry=True, tolerance=0.2)

    # Should have the same label since they have identical geometries up to translation
    labels = [cluster["label"] for cluster in labeled_clusters]
    assert labels[0] == labels[1], f"Expected the same label for both clusters, but got {labels}"
    assert all(cluster["point_group"] == "Td" for cluster in labeled_clusters)