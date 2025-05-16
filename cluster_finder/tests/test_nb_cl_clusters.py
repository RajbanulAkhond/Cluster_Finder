"""
Tests for cluster finder functionality using Nb-Cl clusters data.
"""

import pytest
import pandas as pd
import json
from pathlib import Path
from pymatgen.core.structure import Structure

from cluster_finder.core.clusters import (
    find_clusters,
    calculate_average_distance,
    build_graph,
    analyze_clusters
)
from cluster_finder.core.structure import (
    create_connectivity_matrix,
    structure_to_graph
)

class TestNbClClusters:
    """Test cluster finder functionality with Nb-Cl clusters data."""
    
    @pytest.fixture
    def nb_cl_data(self):
        """Load Nb-Cl clusters data from CSV."""
        csv_path = Path(__file__).parent / "data" / "Nb-Cl_clusters.csv"
        return pd.read_csv(csv_path)
    
    def test_cluster_analysis_mp686087(self, nb_cl_data):
        """Test cluster analysis for mp-686087 (Li3(Nb2Cl5)8)."""
        # Get the first row (mp-686087)
        data = nb_cl_data.iloc[0]

        # Create Structure object from the structure data
        structure_dict = json.loads(data["structure"])
        structure = Structure.from_dict(structure_dict)

        # Create connectivity matrix for Nb atoms with a smaller cutoff
        matrix, indices = create_connectivity_matrix(
            structure=structure,
            transition_metals=["Nb"],
            cutoff=3.0  # Reduced cutoff to avoid including distant connections
        )

        # Convert to graph
        graph = structure_to_graph(matrix)

        # Find clusters
        clusters = find_clusters(
            structure=structure,
            graph=graph,
            tm_indices=indices,
            min_cluster_size=2  # Looking for dimers and larger clusters
        )

        # Analyze clusters
        analyzed_clusters = analyze_clusters(
            clusters=clusters,
            lattice=structure.lattice,
            max_radius=3.0  # Match cutoff used in connectivity matrix
        )

        # Verify that each cluster has valid metal-metal distances
        for cluster in analyzed_clusters:
            sites = cluster["sites"]
            assert cluster["average_distance"] <= 3.0, f"Average distance {cluster['average_distance']} exceeds cutoff"
            
            # Verify basic cluster properties
            assert cluster["size"] >= 2, "Cluster size should be at least 2"
            assert len(cluster["sites"]) == cluster["size"], "Number of sites should match cluster size"
            assert len(cluster["centroid"]) == 3, "Centroid should be a 3D point"
    
    def test_cluster_analysis_mp570445(self, nb_cl_data):
        """Test cluster analysis for mp-570445 (RbNb3VCl11)."""
        # Get the second row (mp-570445)
        data = nb_cl_data.iloc[1]
        
        # Create Structure object from the structure data
        structure_dict = json.loads(data["structure"])
        structure = Structure.from_dict(structure_dict)
        
        # Create connectivity matrix for Nb atoms
        matrix, indices = create_connectivity_matrix(
            structure=structure,
            transition_metals=["Nb"],
            cutoff=3.5
        )
        
        # Convert to graph
        graph = structure_to_graph(matrix)
        
        # Find clusters
        clusters = find_clusters(
            structure=structure,
            graph=graph,
            tm_indices=indices,
            min_cluster_size=2
        )
        
        # Analyze clusters
        analyzed_clusters = analyze_clusters(
            clusters=clusters,
            lattice=structure.lattice,
            max_radius=3.5
        )
        
        # Verify number of clusters
        assert len(analyzed_clusters) == data["num_clusters"]
        
        # Verify cluster sizes
        expected_sizes = json.loads(data["cluster_sizes"])
        actual_sizes = [cluster["size"] for cluster in analyzed_clusters]
        assert sorted(actual_sizes) == sorted(expected_sizes)
        
        # Verify average distances
        expected_distances = json.loads(data["average_distance"])
        actual_distances = [cluster["average_distance"] for cluster in analyzed_clusters]
        
        # Compare distances with a tolerance
        for actual, expected in zip(sorted(actual_distances), sorted(expected_distances)):
            assert abs(actual - expected) < 0.1  # Allow 0.1 Å tolerance