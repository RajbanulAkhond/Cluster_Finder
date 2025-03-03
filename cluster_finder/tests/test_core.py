"""
Tests for core functionality of the cluster_finder package.
"""

import pytest
import numpy as np
import networkx as nx
from pymatgen.core.structure import Structure, Lattice

from cluster_finder.core.structure import (
    find_non_equivalent_positions,
    create_connectivity_matrix,
    calculate_centroid,
    generate_supercell,
    structure_to_graph
)

from cluster_finder.core.clusters import (
    find_clusters,
    calculate_average_distance,
    build_graph,
    split_cluster,
    analyze_clusters,
    identify_unique_clusters
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
        
        # Test with small cutoff (no connections)
        matrix, indices = create_connectivity_matrix(binary_structure, ["Fe", "Co"], cutoff=1.0)
        assert np.sum(matrix) == 0  # No connections
        
        # Test with large cutoff (all connected)
        matrix, indices = create_connectivity_matrix(binary_structure, ["Fe", "Co"], cutoff=5.0)
        # In a cubic structure with side length 3, all atoms should be connected
        assert np.sum(matrix) > 0
    
    def test_calculate_centroid(self, sample_cluster, simple_cubic_structure):
        """Test calculating centroid."""
        centroid = calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
        assert isinstance(centroid, np.ndarray)
        assert len(centroid) == 3  # 3D centroid
        
        # For our specific sample_cluster (0,0,0), (1,0,0), (0,1,0)
        # The centroid should be (1/3, 1/3, 0) in cartesian coordinates
        expected_centroid = np.array([1, 1, 0]) * 3.0 / 3  # 3.0 is the lattice parameter
        assert np.allclose(centroid, expected_centroid)
    
    def test_generate_supercell(self, simple_cubic_structure):
        """Test generating supercell."""
        supercell = generate_supercell(simple_cubic_structure, (2, 2, 2))
        assert len(supercell) == 8 * 8  # 8 sites in original * 8 (2x2x2)
        
        # Test with custom supercell matrix
        supercell = generate_supercell(simple_cubic_structure, (3, 1, 1))
        assert len(supercell) == 8 * 3  # 8 sites in original * 3 (3x1x1)


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
        avg_distance = calculate_average_distance(sample_cluster)
        assert isinstance(avg_distance, float)
        
        # For our specific sample (cube with side 3.0)
        # Distance from (0,0,0) to (1,0,0) and (0,1,0) should be 3.0
        # Distance from (1,0,0) to (0,1,0) should be sqrt(2) * 3.0
        expected_avg = (3.0 + 3.0) / 2  # only consider distances within max_radius
        assert np.isclose(avg_distance, expected_avg)
    
    def test_build_graph(self, sample_cluster):
        """Test building graph from cluster."""
        graph = build_graph(sample_cluster, cutoff=3.5)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2  # Only distances of 3.0
        
        # Test with distances_cache
        distances_cache = {}
        graph = build_graph(sample_cluster, cutoff=3.5, distances_cache=distances_cache)
        assert len(distances_cache) == 3  # 3 pairs of distances should be cached
    
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
        # Create two identical clusters
        clusters = [
            {
                "sites": sample_cluster,
                "size": 3,
                "average_distance": 3.0,
                "centroid": calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
            },
            {
                "sites": sample_cluster,
                "size": 3,
                "average_distance": 3.0,
                "centroid": calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
            }
        ]
        
        unique_clusters = identify_unique_clusters(clusters)
        assert len(unique_clusters) == 1  # Only one unique cluster 