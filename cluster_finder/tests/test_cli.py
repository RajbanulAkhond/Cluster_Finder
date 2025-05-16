"""
Tests for command-line interface of the cluster_finder package.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from pymatgen.core.structure import Structure
from typer.testing import CliRunner
from cluster_finder.cli import (
    app,
    find_command,
    analyze_command,
    visualize_command,
    batch_command,
    summary_command,
    validate_input_file
)

runner = CliRunner()

class TestCLI:
    """Test functions in cli.py"""
    
    def test_find_command(self):
        """Test find command with invalid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.cif")
            with open(test_file, "w") as f:
                f.write("test structure")
            
            result = runner.invoke(app, ["find", test_file])
            assert result.exit_code == 1  # Invalid CIF file should return 1
            
    def test_validate_input_file(self, tmp_path):
        """Test input file validation."""
        # Test non-existent file
        result = runner.invoke(app, ["analyze", "nonexistent.json"])
        assert result.exit_code == 2  # Typer exits with code 2 for invalid input
        
        # Test wrong format
        test_file = tmp_path / "test.txt"
        test_file.write_text("")
        result = runner.invoke(app, ["analyze", str(test_file)])
        assert result.exit_code == 2  # Invalid file format
        
        # Test valid file format but invalid content
        test_json = tmp_path / "test.json"
        test_json.write_text("{}")
        result = runner.invoke(app, ["analyze", str(test_json)])
        assert result.exit_code == 2  # Invalid JSON content
    
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.create_connectivity_matrix')
    @patch('cluster_finder.cli.structure_to_graph')
    @patch('cluster_finder.cli.find_clusters')
    @patch('cluster_finder.cli.analyze_clusters')
    @patch('cluster_finder.cli.visualize_clusters_in_compound')
    @patch('matplotlib.pyplot.savefig')
    def test_find_command_mocked(self, mock_savefig, mock_visualize, mock_analyze, 
                                mock_find, mock_graph, mock_connect, mock_structure):
        """Test find command with mocked dependencies."""
        # Set up mocks
        mock_structure.from_file.side_effect = ValueError("Invalid CIF file")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.cif")
            with open(test_file, "w") as f:
                f.write("test structure")
            
            result = runner.invoke(app, ["find", test_file, "--output", os.path.join(tmpdir, "test_output")])
            assert result.exit_code == 1  # Invalid CIF file should return 1
    
    @patch('cluster_finder.cli.Structure')
    def test_batch_command(self, mock_structure):
        """Test batch command with invalid input files."""
        # Set up the mock Structure class with a side effect that will raise an error
        def mock_from_file(*args, **kwargs):
            raise ValueError("Invalid CIF file")
        mock_structure.from_file.side_effect = mock_from_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            
            (input_dir / "test1.cif").write_text("test")
            (input_dir / "test2.cif").write_text("test")
            
            result = runner.invoke(app, ["batch", str(input_dir)])
            assert result.exit_code == 1  # Invalid CIF files should return 1
    
    def test_summary_command(self, tmp_path):
        """Test summary command."""
        test_data = {
            "material_id": "test-123",
            "formula": "Fe8",
            "num_clusters": 1,
            "total_magnetization": 5.0,
            "clusters": [{"sites": [], "size": 2}]
        }
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        result = runner.invoke(app, ["summary", str(json_file)])
        assert result.exit_code == 0
    
    def test_version_command(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0