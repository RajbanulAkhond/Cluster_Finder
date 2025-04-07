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

from cluster_finder.cli import (
    get_parser,
    find_command,
    analyze_command,
    visualize_command,
    batch_command,
    summary_command,
    validate_input_file,
    main
)


class TestCLI:
    """Test functions in cli.py"""
    
    def test_get_parser(self):
        """Test getting argument parser."""
        parser = get_parser()
        
        # Test that the parser has the expected commands
        subparsers = next(action for action in parser._actions 
                         if action.dest == 'command')
        choices = subparsers.choices
        
        assert 'find' in choices
        assert 'analyze' in choices
        assert 'visualize' in choices
        assert 'batch' in choices
        assert 'summary' in choices
        
        # Test find command arguments
        find_parser = choices['find']
        find_args = find_parser.parse_args(['test.cif'])
        assert find_args.structure_file == 'test.cif'
        assert find_args.format == 'json'
        
        # Test analyze command arguments
        analyze_parser = choices['analyze']
        analyze_args = analyze_parser.parse_args(['test.json'])
        assert analyze_args.json_file == 'test.json'
        assert analyze_args.format == 'csv'
        
        # Test batch command arguments
        batch_parser = choices['batch']
        batch_args = batch_parser.parse_args(['input_dir'])
        assert batch_args.input_dir == 'input_dir'
        assert batch_args.pattern == '*.cif'
    
    def test_validate_input_file(self, tmp_path):
        """Test input file validation."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            validate_input_file("nonexistent.json")
        
        # Test wrong format
        test_file = tmp_path / "test.txt"
        test_file.write_text("")
        with pytest.raises(ValueError):
            validate_input_file(str(test_file), '.json')
        
        # Test valid file
        test_json = tmp_path / "test.json"
        test_json.write_text("{}")
        assert validate_input_file(str(test_json), '.json')
    
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.create_connectivity_matrix')
    @patch('cluster_finder.cli.structure_to_graph')
    @patch('cluster_finder.cli.find_clusters')
    @patch('cluster_finder.cli.analyze_clusters')
    @patch('cluster_finder.cli.visualize_clusters_in_compound')
    @patch('matplotlib.pyplot.savefig')
    @patch('cluster_finder.cli.validate_input_file')
    @patch('cluster_finder.cli.cluster_compounds_dataframe')
    @patch('cluster_finder.cli.export_csv_data')
    def test_find_command(self, mock_export, mock_df, mock_validate, mock_savefig, 
                         mock_visualize, mock_analyze, mock_find, mock_graph, 
                         mock_connect, mock_structure):
        """Test find command with mocked dependencies."""
        # Set up mocks
        mock_validate.return_value = True
        mock_structure.from_file.return_value = MagicMock()
        mock_structure.from_file.return_value.composition.reduced_formula = "Fe8"
        mock_structure.from_file.return_value.as_dict.return_value = {"mock": "structure"}
        
        mock_connect.return_value = (MagicMock(), [0, 1, 2, 3])
        mock_graph.return_value = MagicMock()
        mock_find.return_value = ["cluster1"]
        
        # Create mock sites with proper specie attribute and as_dict method
        mock_site1 = MagicMock()
        mock_site2 = MagicMock()
        mock_specie = MagicMock()
        mock_specie.symbol = "Fe"
        mock_site1.specie = mock_specie
        mock_site2.specie = mock_specie
        mock_site1.as_dict.return_value = {"species": [{"element": "Fe"}], "xyz": [0, 0, 0]}
        mock_site2.as_dict.return_value = {"species": [{"element": "Fe"}], "xyz": [1, 1, 1]}
        
        mock_cluster = {
            "sites": [mock_site1, mock_site2],
            "size": 2,
            "average_distance": 3.0,
            "centroid": [0, 0, 0]
        }
        mock_analyze.return_value = [mock_cluster]
        
        mock_visualize.return_value = MagicMock()
        mock_df.return_value = MagicMock()
        
        # Make export_csv_data create an empty file
        def create_csv_file(df, filename):
            with open(filename, 'w') as f:
                f.write("test")
        mock_export.side_effect = create_csv_file
        
        # Create args object
        args = MagicMock()
        args.structure_file = "test.cif"
        args.elements = ["Fe", "Co"]
        args.radius = 3.5
        args.min_size = 2
        args.output = "test_output"
        args.no_vis = False
        args.format = 'both'
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to the temp directory
            original_dir = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Run the command
                find_command(args)
                
                # Check that output files were created
                assert os.path.exists("test_output_clusters.json")
                assert os.path.exists("test_output_clusters.csv")
                
                # Check JSON content
                with open("test_output_clusters.json", 'r') as f:
                    data = json.load(f)
                    assert data["formula"] == "Fe8"
                    assert data["num_clusters"] == 1
                    assert len(data["clusters"]) == 1
                    assert data["clusters"][0]["size"] == 2
                    assert data["clusters"][0]["elements"] == ["Fe", "Fe"]
                
                # Check that visualization was called
                mock_visualize.assert_called_once()
                mock_savefig.assert_called_once_with("test_output_clusters.png", dpi=300)
                
                # Check that DataFrame functions were called
                mock_df.assert_called_once()
                mock_export.assert_called_once()
            finally:
                # Restore original directory
                os.chdir(original_dir)
    
    @patch('cluster_finder.cli.generate_lattice_with_clusters')
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.cluster_compounds_dataframe')
    @patch('cluster_finder.cli.export_csv_data')
    @patch('cluster_finder.cli.export_structure_to_cif')  # Add patch for export_structure_to_cif
    def test_analyze_command(self, mock_export_cif, mock_export, mock_df, mock_structure, mock_generate_lattice):
        """Test analyze command with mocked dependencies."""
        # Set up mocks
        mock_structure_instance = MagicMock()
        mock_structure.from_dict.return_value = mock_structure_instance
        mock_structure_instance.lattice.as_dict.return_value = {"mock": "lattice"}
        
        # Mock the generate_lattice_with_clusters function
        mock_conv_structure = MagicMock()
        mock_conv_structure.as_dict.return_value = {"mock": "conventional_structure"}
        mock_generate_lattice.return_value = (
            mock_conv_structure,
            "P1",
            {"X1": "1"}
        )
        
        mock_df.return_value = MagicMock()
        
        # Create args object
        args = MagicMock()
        args.json_file = "test_clusters.json"
        args.output = "test_output"
        args.format = 'both'
        args.export_conventional = False  # Default to False
        
        # Create a temporary directory and test JSON file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON file with proper site data
            json_path = os.path.join(tmpdir, "test_clusters.json")
            data = {
                "formula": "Fe8",
                "num_clusters": 1,
                "clusters": [{
                    "sites": [
                        {
                            "species": [{"element": "Fe", "occu": 1.0}],
                            "xyz": [1.0, 1.0, 1.0],
                            "properties": {}
                        },
                        {
                            "species": [{"element": "Fe", "occu": 1.0}],
                            "xyz": [2.0, 2.0, 2.0],
                            "properties": {}
                        }
                    ],
                    "size": 2,
                    "average_distance": 2.5,
                    "centroid": [1.5, 1.5, 1.5],
                    "elements": ["Fe", "Fe"]
                }],
                "structure": {
                    "lattice": {
                        "a": 5.0,
                        "b": 5.0,
                        "c": 5.0,
                        "alpha": 90.0,
                        "beta": 90.0,
                        "gamma": 90.0
                    },
                    "sites": [
                        {
                            "species": [{"element": "Fe", "occu": 1.0}],
                            "xyz": [0.0, 0.0, 0.0],
                            "properties": {}
                        }
                    ]
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f)
                
            # Change args to point to our test file
            args.json_file = json_path
            
            # Run the command
            
            analyze_command(args)
            
            # Verify that export_structure_to_cif was not called (since export_conventional=False)
            mock_export_cif.assert_not_called()
            
            # Now test with export_conventional=True
            args.export_conventional = True
            analyze_command(args)
            
            # Verify that export_structure_to_cif was called
            mock_export_cif.assert_called_once_with(mock_conv_structure, f"test_output_conventional.cif")
    
    @patch('cluster_finder.cli.validate_input_file')  # Add patch for validate_input_file
    @patch('cluster_finder.visualization.visualize.visualize_cluster_lattice')
    @patch('cluster_finder.cli.create_connectivity_matrix')
    @patch('cluster_finder.cli.structure_to_graph')
    @patch('cluster_finder.cli.visualize_graph')
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.visualize_clusters_in_compound')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_command(self, mock_show, mock_savefig, mock_visualize_clusters, 
                              mock_structure, mock_vis_graph, mock_graph, mock_connect,
                              mock_vis_lattice, mock_validate_file):
        """Test visualize command with mocked dependencies."""
        # Set up mocks
        mock_validate_file.return_value = True  # Always return True for file validation
        
        mock_structure_instance = MagicMock()
        mock_structure.from_dict.return_value = mock_structure_instance
        mock_visualize_clusters.return_value = MagicMock()
        mock_vis_graph.return_value = MagicMock()
        mock_vis_lattice.return_value = MagicMock()
        mock_graph.return_value = MagicMock()
        mock_connect.return_value = (MagicMock(), [0])
        
        # Create args object with the new options
        args = MagicMock()
        args.json_file = "test_clusters.json"
        args.output = "test_output"
        args.show = True
        args.dpi = 300
        args.cluster_index = None  # Setting concrete values instead of MagicMock
        args.rotation = "45x,30y,0z"
        args.type = "cluster"  # Default to cluster type
        args.use_3d = False
        
        # Create mock data for the JSON file
        data = {
            "material_id": "mp-123",
            "formula": "Fe2O3",
            "clusters": [{
                "sites": [], 
                "size": 2,
                "average_distance": 2.5,
                "centroid": [0.0, 0.0, 0.0],
                "elements": ["Fe", "Fe"]
            }],
            "structure": {"mock": "structure"}
        }
        
        # Mock open and json.load to return our data
        with patch("builtins.open", mock_open(read_data=json.dumps(data))) as mock_file:
            with patch("json.load", return_value=data):
                # Test cluster visualization
                args.type = "cluster"
                visualize_command(args)
                mock_visualize_clusters.assert_called_with(
                    mock_structure_instance, 
                    [{'size': 2, 'average_distance': 2.5, 'centroid': [0.0, 0.0, 0.0], 
                    'elements': ['Fe', 'Fe'], 'sites': []}],
                    cluster_index=None,
                    rotation="45x,30y,0z"
                )
                
                # Test graph visualization
                args.type = "graph"
                mock_visualize_clusters.reset_mock()
                visualize_command(args)
                mock_vis_graph.assert_called_once()
                
                # Test all visualizations
                args.type = "all"
                mock_visualize_clusters.reset_mock()
                mock_vis_graph.reset_mock()
                
                # Mock os.path.exists to return False so it generates a new conventional structure
                with patch('os.path.exists', return_value=False):
                    with patch('cluster_finder.cli.generate_lattice_with_clusters') as mock_gen_lattice:
                        mock_gen_lattice.return_value = (mock_structure_instance, "P1", {})
                        visualize_command(args)
                
                # Verify that all visualization functions were called
                mock_visualize_clusters.assert_called_once()
                mock_vis_graph.assert_called_once()
                mock_vis_lattice.assert_called_once()
    
    def test_batch_command(self, tmp_path):
        """Test batch command."""
        # Create test directory structure
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create test files
        (input_dir / "test1.cif").write_text("test")
        (input_dir / "test2.cif").write_text("test")
        
        # Create args object
        args = MagicMock()
        args.input_dir = str(input_dir)
        args.pattern = "*.cif"
        args.elements = ["Fe"]
        args.radius = 3.5
        args.output = str(tmp_path / "output")
        
        # Mock dependencies
        with patch("cluster_finder.cli.find_command") as mock_find:
            batch_command(args)
            
            # Check that find_command was called for each file
            assert mock_find.call_count == 2
    
    def test_summary_command(self, tmp_path):
        """Test summary command."""
        # Create test data with total_magnetization field to avoid skipping
        test_data = {
            "material_id": "test-123",
            "formula": "Fe8",
            "num_clusters": 1,
            "total_magnetization": 5.0,
            "clusters": [{"sites": [], "size": 2}]
        }
        
        # Create test files
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        output_file = tmp_path / "summary.txt"
        
        # Create args object with retrieve-missing=False
        args = MagicMock()
        args.input_file = str(json_file)
        args.output = str(output_file)
        args.retrieve_missing = False
        args.api_key = None
        
        # Run command
        with patch("cluster_finder.cli.cluster_summary_stat") as mock_summary:
            mock_summary.return_value = "Test Summary"
            summary_command(args)
            
            # Check that summary was generated
            assert os.path.exists(str(output_file))
            with open(str(output_file)) as f:
                assert f.read() == "Test Summary"
            
            # Verify it was called with the processed compounds
            mock_summary.assert_called_once()
            
        # Test with missing total_magnetization field
        missing_data = {
            "material_id": "test-123",
            "formula": "Fe8",
            "num_clusters": 1,
            "clusters": [{"sites": [], "size": 2}]
        }
        
        # Create test file with missing field
        missing_file = tmp_path / "missing.json"
        with open(missing_file, 'w') as f:
            json.dump(missing_data, f)
        
        # Create args to try retrieving missing data
        args.input_file = str(missing_file)
        args.retrieve_missing = True
        
        # Run command with get_mp_property mocked
        with patch("cluster_finder.cli.get_mp_property") as mock_get_prop:
            mock_get_prop.return_value = 3.0
            with patch("cluster_finder.cli.cluster_summary_stat") as mock_summary:
                mock_summary.return_value = "Test Summary with Retrieved Data"
                
                # Execute the command (it should try to retrieve missing data)
                summary_command(args)
                
                # Check if the property was retrieved
                mock_get_prop.assert_called_with("test-123", "total_magnetization", None)
    
    @patch('cluster_finder.cli.sys.exit')
    @patch('cluster_finder.cli.get_parser')
    @patch('cluster_finder.cli.find_command')
    @patch('cluster_finder.cli.analyze_command')
    @patch('cluster_finder.cli.visualize_command')
    @patch('cluster_finder.cli.batch_command')
    @patch('cluster_finder.cli.summary_command')
    def test_main(self, mock_summary, mock_batch, mock_vis, mock_analyze, mock_find, 
                  mock_parser, mock_exit):
        """Test main function with mocked dependencies."""
        # Set up parser mock
        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance
        
        # Test with each command
        for command in ['find', 'analyze', 'visualize', 'batch', 'summary']:
            mock_args = MagicMock()
            mock_args.command = command
            mock_parser_instance.parse_args.return_value = mock_args
            
            main()
            
            # Check that the appropriate command was called
            if command == 'find':
                mock_find.assert_called_once()
            elif command == 'analyze':
                mock_analyze.assert_called_once()
            elif command == 'visualize':
                mock_vis.assert_called_once()
            elif command == 'batch':
                mock_batch.assert_called_once()
            elif command == 'summary':
                mock_summary.assert_called_once()
            
            # Reset mocks for next iteration
            mock_find.reset_mock()
            mock_analyze.reset_mock()
            mock_vis.reset_mock()
            mock_batch.reset_mock()
            mock_summary.reset_mock()
        
        # Test with no command
        mock_args.command = None
        main()
        mock_parser_instance.print_help.assert_called()
        mock_exit.assert_called_with(1)
        
        # Test with invalid command
        mock_args.command = 'invalid'
        main()
        mock_parser_instance.print_help.assert_called()
        mock_exit.assert_called_with(1)
        
        # Test with exception
        mock_find.side_effect = Exception("Test error")
        mock_args.command = 'find'
        main()
        mock_exit.assert_called_with(1)