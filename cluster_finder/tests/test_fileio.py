"""
Tests for I/O functionality of the cluster_finder package.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure

from cluster_finder.io.fileio import (
    export_structure_to_cif,
    import_csv_data,
    export_csv_data,
    postprocess_clusters
)
from cluster_finder.core.structure import generate_lattice_with_clusters


class TestFileIO:
    """Test functions in fileio.py"""
    
    def test_export_structure_to_cif(self, simple_cubic_structure, tmp_path):
        """Test exporting structure to CIF."""
        # Create output path
        output_path = tmp_path / "test.cif"
        
        # Export structure
        result_path = export_structure_to_cif(simple_cubic_structure, str(output_path))
        
        # Check that file exists
        assert Path(result_path).exists()
        
        # Check that we can read it back
        imported_structure = Structure.from_file(result_path)
        assert len(imported_structure) == len(simple_cubic_structure)
    
    def test_generate_lattice_with_clusters(self, simple_cubic_structure, sample_cluster):
        """Test generating lattice with cluster centroids."""
        # Create a cluster dictionary
        cluster = {
            "sites": sample_cluster,
            "size": len(sample_cluster),
            "average_distance": 3.0,
            "centroid": [0, 0, 0]
        }
        
        # Generate cluster structure
        cluster_structure = generate_lattice_with_clusters(simple_cubic_structure, [cluster])
        
        # Check basic properties
        assert isinstance(cluster_structure, Structure)
        assert len(cluster_structure) == 1  # One centroid
        assert cluster_structure[0].specie.symbol == "X"  # Dummy element
    
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
    
    def test_postprocess_clusters(self, tmp_path):
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
        
        # Postprocess
        processed_df = postprocess_clusters(str(csv_path))
        
        # Check that we have the expected columns
        assert "min_avg_distance" in processed_df.columns
        assert "point_group_order" in processed_df.columns
        assert "space_group_order" in processed_df.columns 