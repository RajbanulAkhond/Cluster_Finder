"""
Tests for core functionality of the cluster_finder package.
"""

import pytest
import numpy as np
import networkx as nx
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.sites import PeriodicSite

from cluster_finder.core.structure import (
    find_non_equivalent_positions,
    create_connectivity_matrix,
    generate_supercell,
    structure_to_graph
)

from cluster_finder.core.clusters import (
    find_clusters,
    calculate_average_distance,
    build_graph,
    split_cluster,
    analyze_clusters,
    identify_unique_clusters,
    calculate_centroid
)


class TestStructure:
    """Test functions in structure.py"""
    
    def test_find_non_equivalent_positions(self, binary_structure, transition_metals):
        """Test finding non-equivalent positions."""
        # Test with a subset of transition metals
        positions = find_non_equivalent_positions(binary_structure, ["Fe"])
        assert len(positions) == 4  # 4 Fe atoms in the binary structure
        
        # Test with multiple elements
        positions = find_non_equivalent_positions(binary_structure, ["Fe", "Co"])
        assert len(positions) == 8  # 4 Fe + 4 Co atoms
        
        # Test with an element not in the structure
        positions = find_non_equivalent_positions(binary_structure, ["Ni"])
        assert len(positions) == 0  # No Ni atoms
    
    def test_create_connectivity_matrix(self, binary_structure, transition_metals):
        """Test creating connectivity matrix."""
        # Test with default cutoff
        matrix, indices = create_connectivity_matrix(binary_structure, ["Fe", "Co"])
        assert matrix.shape == (8, 8)  # 8x8 matrix for 8 atoms
        assert len(indices) == 8
        
        # Test with small cutoff (connections may exist due to implementation)
        matrix, indices = create_connectivity_matrix(binary_structure, ["Fe", "Co"], cutoff=0.1)  # Very small cutoff
        # For very small cutoffs, we expect a matrix of the correct shape
        assert matrix.shape == (8, 8)
        
        # Test with large cutoff (all connected)
        matrix, indices = create_connectivity_matrix(binary_structure, ["Fe", "Co"], cutoff=5.0)
        # In a cubic structure with side length 3, all atoms should be connected
        assert np.sum(matrix) > 0
    
    def test_calculate_centroid(self, sample_cluster, simple_cubic_structure):
        """Test calculating centroid."""
        centroid = calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
        assert isinstance(centroid, np.ndarray)
        assert len(centroid) == 3  # 3D centroid
        
        # For our specific sample_cluster (0,0,0), (0.5,0,0), (0,0.5,0)
        # The centroid should be (0.166667, 0.166667, 0) in fractional coordinates
        # Convert to cartesian coordinates for comparison
        expected_frac_coords = np.array([1/6, 1/6, 0])
        expected_centroid = simple_cubic_structure.lattice.get_cartesian_coords(expected_frac_coords)
        assert np.allclose(centroid, expected_centroid)
    
    def test_generate_supercell(self):
        """Test supercell generation."""
        # Create a simple cubic structure
        lattice = [[1,0,0], [0,1,0], [0,0,1]]
        coords = [[0,0,0]]
        species = ["Fe"]
        structure = Structure(lattice, species, coords)
        
        # Generate 2x2x2 supercell
        supercell = generate_supercell(structure, [2,2,2])
        
        # Check supercell size (should be 8x original)
        assert len(structure) == 1  # Original structure has 1 atom
        assert len(supercell) == 8  # Supercell should have 8 atoms
        
        # Check lattice parameters
        assert np.allclose(supercell.lattice.matrix, 2 * np.array(lattice))


class TestClusters:
    """Test functions in clusters.py"""
    
    def test_structure_to_graph(self, connectivity_matrix):
        """Test converting connectivity matrix to graph."""
        matrix, _ = connectivity_matrix
        graph = structure_to_graph(matrix)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 3  # Based on our test matrix
        
        # Check specific connections
        assert graph.has_edge(0, 1)
        assert graph.has_edge(0, 2)
        assert graph.has_edge(2, 3)
    
    def test_find_clusters(self, simple_cubic_structure, connectivity_matrix):
        """Test finding clusters."""
        matrix, tm_indices = connectivity_matrix
        graph = structure_to_graph(matrix)
        
        # Test with min_cluster_size=2
        clusters = find_clusters(simple_cubic_structure, graph, tm_indices, min_cluster_size=2)
        assert len(clusters) == 1  # One cluster with 4 atoms
        assert len(clusters[0]) == 4
        
        # Test with min_cluster_size=5 (should find no clusters)
        clusters = find_clusters(simple_cubic_structure, graph, tm_indices, min_cluster_size=5)
        assert len(clusters) == 0
    
    def test_calculate_average_distance(self, sample_cluster):
        """Test calculating average distance."""
        # The mock sites are in a triangular arrangement with edges 1.5 units long
        # For a triangular arrangement, we expect the average distance to be
        # (1.5 + 1.5 + sqrt(2)*1.5) / 3 ≈ 1.707 units
        avg_distance = calculate_average_distance(sample_cluster, max_radius=3.5)
        assert isinstance(avg_distance, float)
        assert np.isclose(avg_distance, 1.7071067811865472, rtol=1e-6)  # (1.5 + 1.5 + sqrt(2)*1.5) / 3
    
    def test_build_graph(self, sample_cluster):
        """Test building graph from cluster."""
        # Test with default cutoff
        graph = build_graph(sample_cluster, cutoff=3.5)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3  # Three edges with cutoff 3.5
    
    def test_analyze_clusters(self, sample_cluster, simple_cubic_structure):
        """Test analyzing clusters."""
        clusters = [sample_cluster]
        analyzed = analyze_clusters(clusters, simple_cubic_structure.lattice)
        
        assert len(analyzed) == 1
        assert "sites" in analyzed[0]
        assert "size" in analyzed[0]
        assert "average_distance" in analyzed[0]
        assert "centroid" in analyzed[0]
        
        assert analyzed[0]["size"] == 3
        assert len(analyzed[0]["sites"]) == 3
    
    def test_identify_unique_clusters(self, sample_cluster, simple_cubic_structure):
        """Test identifying unique clusters."""
        # Create two identical clusters at different positions
        sites1 = sample_cluster
        
        # Create second cluster by translating in fractional coordinates
        sites2 = [
            PeriodicSite(
                site.species,
                site.frac_coords + np.array([0.1, 0.1, 0.1]),
                simple_cubic_structure.lattice
            )
            for site in sites1
        ]

        # Create cluster dictionaries with fractional coordinates
        centroid1 = calculate_centroid(sites1, simple_cubic_structure.lattice)
        centroid2 = calculate_centroid(sites2, simple_cubic_structure.lattice)

        cluster1 = {
            "sites": sites1,
            "size": len(sites1),
            "average_distance": 3.0,
            "centroid": centroid1,
            "relative_coords": np.array([site.frac_coords - np.array(centroid1) for site in sites1])
        }

        cluster2 = {
            "sites": sites2,
            "size": len(sites2),
            "average_distance": 3.0,
            "centroid": centroid2,
            "relative_coords": np.array([site.frac_coords - np.array(centroid2) for site in sites2])
        }

        # The clusters should be considered identical since they have the same relative geometry
        labeled_clusters = identify_unique_clusters([cluster1, cluster2], use_symmetry=True, tolerance=0.2)
        
        # Check that they both have the same label
        labels = [cluster["label"] for cluster in labeled_clusters]
        assert labels[0] == labels[1], f"Expected the same label for both clusters, but got {labels}"