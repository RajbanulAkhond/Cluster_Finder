"""
Tests for command-line interface of the cluster_finder package.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
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
    
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.SpacegroupAnalyzer')
    @patch('cluster_finder.cli.cluster_compounds_dataframe')
    @patch('cluster_finder.cli.export_csv_data')
    def test_analyze_command(self, mock_export, mock_df, mock_spacegroup, mock_structure):
        """Test analyze command with mocked dependencies."""
        # Set up mocks
        mock_structure.from_dict.return_value = MagicMock()
        mock_spacegroup.return_value.get_space_group_symbol.return_value = "P1"
        mock_spacegroup.return_value.get_point_group_symbol.return_value = "1"
        
        mock_df.return_value = MagicMock()
        
        # Create args object
        args = MagicMock()
        args.json_file = "test_clusters.json"
        args.output = "test_output"
        args.format = 'both'
        
        # Create a temporary directory and test JSON file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON file
            json_path = os.path.join(tmpdir, "test_clusters.json")
            data = {
                "formula": "Fe8",
                "num_clusters": 1,
                "clusters": [{"sites": [], "size": 2}],
                "structure": {"mock": "structure"}
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f)
            
            # Change args to point to our test file
            args.json_file = json_path
            
            # Run the command
            analyze_command(args)
            
            # Check that the mock functions were called
            mock_structure.from_dict.assert_called_once()
            mock_spacegroup.assert_called_once()
            mock_df.assert_called_once()
            mock_export.assert_called_once()
    
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.visualize_clusters_in_compound')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_command(self, mock_show, mock_savefig, mock_visualize, mock_structure):
        """Test visualize command with mocked dependencies."""
        # Set up mocks
        mock_structure.from_dict.return_value = MagicMock()
        mock_visualize.return_value = MagicMock()
        
        # Create args object
        args = MagicMock()
        args.json_file = "test_clusters.json"
        args.output = "test_output"
        args.show = True
        args.dpi = 300
        
        # Create a temporary directory and test JSON file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON file
            json_path = os.path.join(tmpdir, "test_clusters.json")
            data = {
                "clusters": [{"sites": [], "size": 2}],
                "structure": {"mock": "structure"}
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f)
            
            # Change args to point to our test file
            args.json_file = json_path
            
            # Run the command
            visualize_command(args)
            
            # Check that the mock functions were called
            mock_structure.from_dict.assert_called_once()
            mock_visualize.assert_called_once()
            mock_savefig.assert_called_once_with("test_output_clusters.png", dpi=300)
            mock_show.assert_called_once()
    
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
        # Create test data
        test_data = {
            "formula": "Fe8",
            "num_clusters": 1,
            "clusters": [{"sites": [], "size": 2}]
        }
        
        # Create test files
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        # Create args object
        args = MagicMock()
        args.input_file = str(json_file)
        args.output = str(tmp_path / "summary.txt")
        
        # Run command
        with patch("cluster_finder.cli.cluster_summary_stat") as mock_summary:
            mock_summary.return_value = "Test Summary"
            summary_command(args)
            
            # Check that summary was generated
            assert os.path.exists(args.output)
            with open(args.output) as f:
                assert f.read() == "Test Summary"
    
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