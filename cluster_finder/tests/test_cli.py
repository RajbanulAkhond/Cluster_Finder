"""
Tests for command-line interface of the cluster_finder package.
"""

import os
import sys
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from pymatgen.core.structure import Structure

from cluster_finder.cli import (
    get_parser,
    find_command,
    analyze_command,
    visualize_command,
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
    
    @patch('cluster_finder.cli.Structure')
    @patch('cluster_finder.cli.create_connectivity_matrix')
    @patch('cluster_finder.cli.structure_to_graph')
    @patch('cluster_finder.cli.find_clusters')
    @patch('cluster_finder.cli.analyze_clusters')
    @patch('cluster_finder.cli.visualize_clusters_in_compound')
    @patch('matplotlib.pyplot.savefig')
    def test_find_command(self, mock_savefig, mock_visualize, mock_analyze, 
                         mock_find, mock_graph, mock_connect, mock_structure):
        """Test find command with mocked dependencies."""
        # Set up mocks
        mock_structure.from_file.return_value = MagicMock()
        mock_structure.from_file.return_value.composition.reduced_formula = "Fe8"
        mock_structure.from_file.return_value.as_dict.return_value = {"mock": "structure"}
        
        mock_connect.return_value = (MagicMock(), [0, 1, 2, 3])
        mock_graph.return_value = MagicMock()
        mock_find.return_value = ["cluster1"]
        
        mock_cluster = {
            "sites": ["site1", "site2"],
            "size": 2,
            "average_distance": 3.0,
            "centroid": [0, 0, 0]
        }
        mock_analyze.return_value = [mock_cluster]
        
        mock_visualize.return_value = MagicMock()
        
        # Create args object
        args = MagicMock()
        args.structure_file = "test.cif"
        args.elements = ["Fe", "Co"]
        args.radius = 3.5
        args.min_size = 2
        args.output = "test_output"
        args.no_vis = False
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to the temp directory
            original_dir = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Run the command
                find_command(args)
                
                # Check that the output file was created
                assert os.path.exists("test_output_clusters.json")
                
                # Check JSON content
                with open("test_output_clusters.json", 'r') as f:
                    data = json.load(f)
                    assert data["structure_file"] == "test.cif"
                    assert data["formula"] == "Fe8"
                    assert data["num_clusters"] == 1
                
                # Check that visualization was called
                mock_visualize.assert_called_once()
                mock_savefig.assert_called_once()
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
            mock_savefig.assert_called_once()
            mock_show.assert_called_once()
    
    @patch('cluster_finder.cli.sys.exit')
    @patch('cluster_finder.cli.get_parser')
    @patch('cluster_finder.cli.find_command')
    @patch('cluster_finder.cli.analyze_command')
    @patch('cluster_finder.cli.visualize_command')
    def test_main(self, mock_vis, mock_analyze, mock_find, mock_parser, mock_exit):
        """Test main function with mocked dependencies."""
        # Set up parser mock
        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance
        
        # Test with 'find' command
        mock_args = MagicMock()
        mock_args.command = 'find'
        mock_parser_instance.parse_args.return_value = mock_args
        
        main()
        mock_find.assert_called_once()
        
        # Test with 'analyze' command
        mock_find.reset_mock()
        mock_args.command = 'analyze'
        main()
        mock_analyze.assert_called_once()
        
        # Test with 'visualize' command
        mock_analyze.reset_mock()
        mock_args.command = 'visualize'
        main()
        mock_vis.assert_called_once()
        
        # Test with no command
        mock_vis.reset_mock()
        mock_args.command = None
        main()
        mock_parser_instance.print_help.assert_called()
        mock_exit.assert_called_with(1) 