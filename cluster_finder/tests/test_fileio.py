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
        from cluster_finder.core.structure import generate_lattice_with_clusters
        
        # Calculate centroid for the sample cluster
        centroid = calculate_centroid(sample_cluster, simple_cubic_structure.lattice)
        
        # Create a cluster dictionary as expected by generate_lattice_with_clusters
        cluster_dict = {
            "sites": sample_cluster,
            "centroid": centroid,
            "size": len(sample_cluster),
            "average_distance": 2.5
        }
        
        # Generate lattice with clusters
        new_structure = generate_lattice_with_clusters(simple_cubic_structure, [cluster_dict])
        
        # Check that new structure has expected properties
        assert isinstance(new_structure, Structure)
        assert len(new_structure) == 1  # One site for the centroid
        assert new_structure[0].species_string == "X0+"  # Dummy atom with charge
    
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
    
    def test_postprocessed_clusters_dataframe(self, tmp_path):
        """Test postprocessing cluster data."""
        # Create test data
        data = {
            "material_id": ["mp-123", "mp-456"],
            "formula": ["Fe2O3", "Co3O4"],
            "cluster_sizes": ["[2, 3]", "[2, 2, 2]"],
            "average_distance": ["[2.5, 2.8]", "[2.2, 2.3, 2.4]"],
            "space_group": ["P1", "Fm-3m"],
            "point_group": ["1", "m-3m"],
            "total_magnetization": [5.0, 8.0],
            "dimensionality": ["3D", "3D"]
        }
        df = pd.DataFrame(data)
        
        # Export to CSV
        csv_path = tmp_path / "test.csv"
        export_csv_data(df, str(csv_path))
        
        # Test with file path
        processed_df = postprocessed_clusters_dataframe(str(csv_path))
        
        # Check that we have the expected columns
        assert "min_avg_distance" in processed_df.columns
        assert "point_group_order" in processed_df.columns
        assert "space_group_order" in processed_df.columns
        
        # Test with DataFrame directly
        processed_df2 = postprocessed_clusters_dataframe(df)
        
        # Check that results are the same
        pd.testing.assert_frame_equal(processed_df, processed_df2) 