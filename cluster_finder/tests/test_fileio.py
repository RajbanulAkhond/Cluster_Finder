"""
Tests for I/O functionality of the cluster_finder package.
"""

import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure

from cluster_finder.io.fileio import (
    export_structure_to_cif,
    import_csv_data,
    export_csv_data
)
from cluster_finder.analysis.dataframe import (
    postprocessed_clusters_dataframe
)
from cluster_finder.core.structure import calculate_centroid

class TestFileIO:
    """Test functions in fileio.py"""
    
    def test_export_structure_to_cif(self, simple_cubic_structure, tmp_path):
        """Test exporting structure to CIF."""
        # Export structure to CIF
        cif_path = tmp_path / "test.cif"
        export_structure_to_cif(simple_cubic_structure, str(cif_path))
        
        # Check that file exists
        assert os.path.exists(cif_path)
        
        # Check that file contains expected content
        with open(cif_path, 'r') as f:
            content = f.read()
            assert "data_" in content
            assert "_cell_length_a" in content
    
    def test_generate_lattice_with_clusters(self, simple_cubic_structure, sample_cluster):
        """Test generating lattice with clusters."""
        from cluster_finder.core.structure import generate_lattice_with_clusters, calculate_centroid

        # Calculate centroid for the sample cluster
        centroid = calculate_centroid(sample_cluster, simple_cubic_structure.lattice)

        # Create a cluster dictionary as expected by generate_lattice_with_clusters
        cluster_dict = {
            "sites": sample_cluster,
            "centroid": centroid,
            "size": len(sample_cluster),
            "average_distance": 2.5,
            "label": "X0"  # Label should be X0 for first cluster
        }

        # Generate lattice with clusters
        conventional_structure, space_group_symbol, point_groups = generate_lattice_with_clusters(simple_cubic_structure, [cluster_dict])

        # Check that new structure has expected properties
        assert isinstance(conventional_structure, Structure)
        assert len(conventional_structure) == 1  # One site for the centroid
        assert conventional_structure[0].species_string == "X00+"  # DummySpecie with charge 1
    
    def test_import_export_csv(self, tmp_path):
        """Test importing and exporting CSV data."""
        # Create test data
        data = {
            "material_id": ["mp-123", "mp-456"],
            "formula": ["Fe2O3", "Co3O4"],
            "cluster_sizes": ["[2, 3]", "[2, 2, 2]"],
            "average_distance": ["[2.5, 2.8]", "[2.2, 2.3, 2.4]"]
        }
        df = pd.DataFrame(data)
        
        # Export to CSV
        csv_path = tmp_path / "test.csv"
        export_csv_data(df, str(csv_path))
        
        # Import from CSV
        imported_df = import_csv_data(str(csv_path))
        
        # Check that data matches
        pd.testing.assert_frame_equal(df, imported_df)
    
    def test_postprocessed_clusters_dataframe(self, tmp_path, simple_cubic_structure, sample_cluster):
        """Test postprocessing cluster data."""
        # Create test data
        cluster_sites_json = [[site.as_dict() for site in sample_cluster]]
        data = {
            "material_id": ["mp-123"],
            "formula": ["Fe3"],
            "cluster_sizes": [[3]],  # List containing a list with one value
            "average_distance": [[2.5]],  # List containing a list with one value
            "num_clusters": [1],
            "magnetization": [5.0],
            "structure": [simple_cubic_structure.as_dict()],  # Changed to as_dict() instead of to(fmt="json")
            "cluster_sites": [str(cluster_sites_json)],
            "compound_system": ["Fe-O"]
        }
        df = pd.DataFrame(data)
        
        # Export to CSV
        csv_path = tmp_path / "test.csv"
        export_csv_data(df, str(csv_path))
        
        # Test with file path
        processed_df = postprocessed_clusters_dataframe(str(csv_path))
        
        # Test with DataFrame directly
        processed_df_direct = postprocessed_clusters_dataframe(df)
        
        # Check that both methods give the same result
        pd.testing.assert_frame_equal(processed_df, processed_df_direct)
        
        # Check that the expected columns are present
        expected_columns = [
            "material_id", "formula", "magnetization", "num_clusters",
            "cluster_sizes", "average_distance", "space_group", "point_groups",
            "predicted_dimentionality", "norm_svals", "conventional_cluster_lattice",
            "cluster_sites"
        ]
        for col in expected_columns:
            assert col in processed_df.columns, f"Column {col} not found in processed DataFrame"