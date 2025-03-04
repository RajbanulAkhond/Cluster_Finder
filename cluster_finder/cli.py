"""
Command-line interface for cluster_finder package.

This module provides a command-line interface to use the functionality
of the cluster_finder package.
"""

import os
import sys
import argparse
import json
from typing import List, Optional
from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .core.graph import create_connectivity_matrix, structure_to_graph
from .core.clusters import (
    find_clusters, 
    analyze_clusters
)
from .visualization.visualize import (
    visualize_graph,
    visualize_clusters_in_compound
)
from .io.fileio import (
    export_structure_to_cif,
    export_csv_data,
    import_csv_data
)
from .core.utils import cluster_summary_stat
from .utils.helpers import get_transition_metals
from .analysis.dataframe import cluster_compounds_dataframe
from .analysis.postprocess import rank_clusters


def get_parser():
    """
    Create the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Cluster Finder - Find and analyze atomic clusters in crystal structures'
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Find clusters in a structure file
    find_parser = subparsers.add_parser('find', help='Find clusters in a structure file')
    find_parser.add_argument('structure_file', help='Structure file (CIF, POSCAR, etc.)')
    find_parser.add_argument('--elements', '-e', nargs='+', help='Elements to consider for clusters (default: all transition metals)')
    find_parser.add_argument('--radius', '-r', type=float, default=3.5, help='Maximum atom-to-atom distance for cluster search (default: 3.5 Å)')
    find_parser.add_argument('--min-size', '-s', type=int, default=2, help='Minimum cluster size (default: 2)')
    find_parser.add_argument('--output', '-o', help='Output file prefix (default: based on input filename)')
    find_parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    find_parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json', help='Output format (default: json)')
    
    # Analyze clusters from a previous run
    analyze_parser = subparsers.add_parser('analyze', help='Analyze clusters from a previous run')
    analyze_parser.add_argument('json_file', help='JSON file with cluster data')
    analyze_parser.add_argument('--output', '-o', help='Output file prefix')
    analyze_parser.add_argument('--format', choices=['csv', 'json', 'both'], default='csv', help='Output format (default: csv)')
    
    # Visualize clusters
    vis_parser = subparsers.add_parser('visualize', help='Visualize clusters')
    vis_parser.add_argument('json_file', help='JSON file with cluster data')
    vis_parser.add_argument('--output', '-o', help='Output file prefix')
    vis_parser.add_argument('--show', action='store_true', help='Show visualization (requires GUI)')
    vis_parser.add_argument('--dpi', type=int, default=300, help='DPI for saved images (default: 300)')
    
    # Batch process multiple structures
    batch_parser = subparsers.add_parser('batch', help='Process multiple structure files')
    batch_parser.add_argument('input_dir', help='Directory containing structure files')
    batch_parser.add_argument('--pattern', default='*.cif', help='Glob pattern for input files (default: *.cif)')
    batch_parser.add_argument('--elements', '-e', nargs='+', help='Elements to consider for clusters')
    batch_parser.add_argument('--radius', '-r', type=float, default=3.5, help='Maximum atom-to-atom distance')
    batch_parser.add_argument('--output', '-o', help='Output directory')
    
    # Generate summary statistics
    summary_parser = subparsers.add_parser('summary', help='Generate summary statistics')
    summary_parser.add_argument('input_file', help='JSON or CSV file with cluster data')
    summary_parser.add_argument('--output', '-o', help='Output file for summary')
    
    return parser


def validate_input_file(file_path: str, expected_format: Optional[str] = None) -> bool:
    """
    Validate input file existence and format.
    
    Parameters:
        file_path (str): Path to the input file
        expected_format (str, optional): Expected file format (extension)
        
    Returns:
        bool: True if file is valid
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if expected_format:
        if not file_path.lower().endswith(expected_format.lower()):
            raise ValueError(f"Invalid file format. Expected {expected_format}, got {os.path.splitext(file_path)[1]}")
    
    return True


def find_command(args):
    """Find clusters in a structure."""
    try:
        # Validate input file
        validate_input_file(args.structure_file)
        
        # Load structure
        structure = Structure.from_file(args.structure_file)
        print(f"\nFound {len(structure)} sites in {structure.composition.reduced_formula}")
        
        # Use default elements if none provided
        elements = args.elements or get_transition_metals()
        
        # Create connectivity matrix
        matrix, indices = create_connectivity_matrix(structure, elements, args.radius)
        
        # Create graph
        graph = structure_to_graph(matrix)
        
        # Find clusters
        clusters = find_clusters(structure, graph, indices, args.min_size)
        
        # Analyze clusters
        analyzed_clusters = analyze_clusters(clusters, structure.lattice) if clusters else []
        
        # Convert analyzed clusters to JSON-serializable format
        json_clusters = []
        for cluster in analyzed_clusters:
            json_cluster = {
                "size": cluster["size"],
                "average_distance": float(cluster["average_distance"]),
                "centroid": [float(x) for x in cluster["centroid"]],
                "elements": [site.specie.symbol for site in cluster["sites"]],
                "sites": [site.as_dict() for site in cluster["sites"]]
            }
            json_clusters.append(json_cluster)
        
        # Prepare output data
        result = {
            "formula": structure.composition.reduced_formula,
            "structure": structure.as_dict(),
            "clusters": json_clusters,
            "num_clusters": len(analyzed_clusters)
        }
        
        # Determine output path
        output_base = args.output or os.path.splitext(args.structure_file)[0]
        os.makedirs(os.path.dirname(output_base) or '.', exist_ok=True)
        
        # Save results
        if args.format in ['json', 'both']:
            json_file = f"{output_base}_clusters.json"
            with open(json_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved cluster data to {json_file}")
        
        if args.format in ['csv', 'both']:
            df = cluster_compounds_dataframe([{
                "material_id": os.path.basename(args.structure_file),
                "formula": result["formula"],
                "structure": structure,  # Pass the Structure object directly
                "clusters": analyzed_clusters,  # Pass the original clusters with PeriodicSite objects
                "num_clusters": result["num_clusters"]
            }])
            if df is not None:  # Check if DataFrame was created successfully
                csv_file = f"{output_base}_clusters.csv"
                export_csv_data(df, csv_file)
                print(f"Saved cluster data to {csv_file}")
        
        # Visualize if requested
        if not args.no_vis and analyzed_clusters:
            import matplotlib.pyplot as plt
            fig = visualize_clusters_in_compound(structure, analyzed_clusters)
            png_file = f"{output_base}_clusters.png"
            plt.savefig(png_file, dpi=300)
            print(f"Saved cluster visualization to {png_file}")
        
        print(f"\nFound {len(analyzed_clusters)} clusters in {structure.composition.reduced_formula}")
        
    except Exception as e:
        print(f"Error in find command: {str(e)}", file=sys.stderr)
        sys.exit(1)


def analyze_command(args):
    """Execute the analyze command to further analyze clusters."""
    try:
        # Validate input file
        validate_input_file(args.json_file, '.json')
        
        # Load data
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        
        # Create structure from dict
        structure = Structure.from_dict(data["structure"])
        
        # Analyze space group
        analyzer = SpacegroupAnalyzer(structure)
        space_group = analyzer.get_space_group_symbol()
        point_group = analyzer.get_point_group_symbol()
        
        # Create compound object
        compound = {
            "material_id": os.path.basename(args.json_file).split('_')[0],
            "formula": data["formula"],
            "clusters": data["clusters"],
            "structure": structure
        }
        
        # Create DataFrame
        df = cluster_compounds_dataframe([compound])
        
        # Determine output prefix
        output_prefix = args.output or os.path.splitext(os.path.basename(args.json_file))[0]
        
        # Save results
        if args.format in ['csv', 'both']:
            csv_file = f"{output_prefix}_analysis.csv"
            export_csv_data(df, csv_file)
            print(f"Saved analysis to {csv_file}")
        
        if args.format in ['json', 'both']:
            json_file = f"{output_prefix}_analysis.json"
            with open(json_file, 'w') as f:
                json.dump({
                    "formula": data["formula"],
                    "space_group": space_group,
                    "point_group": point_group,
                    "num_clusters": len(data["clusters"]),
                    "clusters": data["clusters"]
                }, f, indent=2)
            print(f"Saved analysis to {json_file}")
        
        # Print summary
        print(f"\nStructure Analysis for {data['formula']}")
        print(f"Space Group: {space_group}")
        print(f"Point Group: {point_group}")
        print(f"Number of Clusters: {data['num_clusters']}")
        
    except Exception as e:
        print(f"Error in analyze command: {str(e)}", file=sys.stderr)
        sys.exit(1)


def visualize_command(args):
    """Execute the visualize command to create visualizations."""
    try:
        # Validate input file
        validate_input_file(args.json_file, '.json')
        
        # Load data
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        
        # Create structure from dict
        structure = Structure.from_dict(data["structure"])
        
        # Determine output prefix
        output_prefix = args.output or os.path.splitext(os.path.basename(args.json_file))[0]
        
        # Create visualization
        try:
            import matplotlib.pyplot as plt
            fig = visualize_clusters_in_compound(structure, data["clusters"])
            
            if args.show:
                plt.show()
            
            plt.savefig(f"{output_prefix}_clusters.png", dpi=args.dpi)
            print(f"Saved cluster visualization to {output_prefix}_clusters.png")
        except Exception as e:
            print(f"Error creating visualization: {e}")
            
    except Exception as e:
        print(f"Error in visualize command: {str(e)}", file=sys.stderr)
        sys.exit(1)


def batch_command(args):
    """Execute batch processing of multiple structures."""
    try:
        # Validate input directory
        if not os.path.isdir(args.input_dir):
            raise NotADirectoryError(f"Directory not found: {args.input_dir}")
        
        # Create output directory
        output_dir = args.output or os.path.join(args.input_dir, 'cluster_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all matching files
        all_results = []
        for file_path in Path(args.input_dir).glob(args.pattern):
            print(f"\nProcessing {file_path}...")
            
            try:
                # Create args for find_command
                find_args = argparse.Namespace(
                    structure_file=str(file_path),
                    elements=args.elements,
                    radius=args.radius,
                    min_size=2,
                    output=os.path.join(output_dir, file_path.stem),
                    no_vis=True,
                    format='both'
                )
                
                # Run find command
                find_command(find_args)
                
                # Load results for summary
                with open(f"{find_args.output}_clusters.json") as f:
                    result = json.load(f)
                    all_results.append(result)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Generate summary
        if all_results:
            summary_file = os.path.join(output_dir, "batch_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(cluster_summary_stat(all_results, all_results))
            print(f"\nSaved batch summary to {summary_file}")
            
    except Exception as e:
        print(f"Error in batch command: {str(e)}", file=sys.stderr)
        sys.exit(1)


def summary_command(args):
    """Generate summary statistics for cluster data."""
    try:
        # Validate input file
        validate_input_file(args.input_file)
        
        # Load data based on file format
        if args.input_file.lower().endswith('.json'):
            with open(args.input_file) as f:
                data = json.load(f)
                compounds = [data] if isinstance(data, dict) else data
        else:  # CSV format
            df = import_csv_data(args.input_file)
            compounds = df.to_dict('records')
        
        # Generate summary
        summary = cluster_summary_stat(compounds, compounds)
        
        # Output summary
        if args.output:
            with open(args.output, 'w') as f:
                f.write(summary)
            print(f"Saved summary to {args.output}")
        else:
            print(summary)
            
    except Exception as e:
        print(f"Error in summary command: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the command-line interface."""
    parser = get_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    try:
        if args.command == 'find':
            find_command(args)
        elif args.command == 'analyze':
            analyze_command(args)
        elif args.command == 'visualize':
            visualize_command(args)
        elif args.command == 'batch':
            batch_command(args)
        elif args.command == 'summary':
            summary_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 