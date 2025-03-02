"""
Tests for visualization functionality of the cluster_finder package.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cluster_finder.visualization.visualize import (
    visualize_graph,
    visualize_clusters_in_compound,
    visualize_cluster_lattice
)
from cluster_finder.core.clusters import structure_to_graph
from cluster_finder.core.structure import calculate_centroid


# Use non-interactive backend for testing
matplotlib.use('Agg')


class TestVisualization:
    """Test functions in visualize.py"""
    
    def test_visualize_graph(self, simple_cubic_structure, connectivity_matrix):
        """Test visualizing graph."""
        matrix, tm_indices = connectivity_matrix
        graph = structure_to_graph(matrix)
        
        # Test visualization
        fig = visualize_graph(graph, simple_cubic_structure, tm_indices)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
        
        # Test with additional parameters
        fig = visualize_graph(graph, simple_cubic_structure, tm_indices, 
                             material_id="test_id", formula="Fe8")
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_visualize_clusters_in_compound(self, simple_cubic_structure, sample_cluster):
        """Test visualizing clusters in compound."""
        # Create a cluster with metadata
        centroid = calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
        cluster = {
            "sites": sample_cluster,
            "size": len(sample_cluster),
            "average_distance": 3.0,
            "centroid": centroid
        }
        
        # Test visualization with one cluster
        fig = visualize_clusters_in_compound(simple_cubic_structure, [cluster])
        
        assert isinstance(fig, Figure)
        plt.close(fig)
        
        # Test with empty clusters list
        fig = visualize_clusters_in_compound(simple_cubic_structure, [])
        
        assert fig is None  # Should return None for empty clusters
    
    def test_visualize_cluster_lattice(self, simple_cubic_structure):
        """Test visualizing cluster lattice."""
        # Test without rotation
        fig = visualize_cluster_lattice(simple_cubic_structure)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
        
        # Test with rotation matrix
        rotation_matrix = np.array([
            [0.8660254, -0.5, 0],
            [0.5, 0.8660254, 0],
            [0, 0, 1]
        ])  # 30 degree rotation around z-axis
        
        fig = visualize_cluster_lattice(simple_cubic_structure, rotation_matrix)
        
        assert isinstance(fig, Figure)
        plt.close(fig) 