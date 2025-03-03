"""
Command-line interface for cluster_finder package.

This module provides a command-line interface to use the functionality
of the cluster_finder package.
"""

import os
import sys
import argparse
import json
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .core.structure import create_connectivity_matrix, structure_to_graph
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
    export_csv_data
)
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
    
    # Analyze clusters from a previous run
    analyze_parser = subparsers.add_parser('analyze', help='Analyze clusters from a previous run')
    analyze_parser.add_argument('json_file', help='JSON file with cluster data')
    analyze_parser.add_argument('--output', '-o', help='Output file prefix')
    
    # Visualize clusters
    vis_parser = subparsers.add_parser('visualize', help='Visualize clusters')
    vis_parser.add_argument('json_file', help='JSON file with cluster data')
    vis_parser.add_argument('--output', '-o', help='Output file prefix')
    vis_parser.add_argument('--show', action='store_true', help='Show visualization (requires GUI)')
    
    return parser


def find_command(args):
    """Find clusters in a structure."""
    # Load structure
    structure = Structure.from_file(args.structure_file)
    print(f"\nFound {len(structure)} sites in {structure.composition.reduced_formula}")

    # Create connectivity matrix
    matrix, indices = create_connectivity_matrix(structure, args.elements, args.radius)
    
    # Create graph
    graph = structure_to_graph(matrix)
    
    # Find clusters
    clusters = find_clusters(graph, indices, args.min_size)
    
    # Analyze clusters
    analyzed_clusters = analyze_clusters(clusters, structure)
    
    # Prepare output data
    result = {
        "formula": structure.composition.reduced_formula,
        "clusters": [{
            "size": cluster["size"],
            "average_distance": float(cluster["average_distance"]),  # Convert numpy float to native float
            "centroid": [float(x) for x in cluster["centroid"]],  # Convert numpy array to list of floats
            "elements": [site.specie.symbol for site in cluster["sites"]]
        } for cluster in analyzed_clusters]
    }

    # Save results
    output_base = args.output or "clusters"
    json_file = f"{output_base}_clusters.json"
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved cluster data to {json_file}")

    # Visualize if requested
    if not args.no_vis:
        import matplotlib.pyplot as plt
        fig = visualize_clusters_in_compound(structure, analyzed_clusters)
        png_file = f"{output_base}_clusters.png"
        plt.savefig(png_file)
        print(f"Saved cluster visualization to {png_file}")

    print(f"\nFound {len(analyzed_clusters)} clusters in {structure.composition.reduced_formula}")


def analyze_command(args):
    """
    Execute the analyze command to further analyze clusters.
    
    Parameters:
        args (Namespace): Command-line arguments
    """
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
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = os.path.splitext(os.path.basename(args.json_file))[0]
    
    # Save DataFrame
    export_csv_data(df, f"{output_prefix}_analysis.csv")
    print(f"Saved analysis to {output_prefix}_analysis.csv")
    
    # Print summary
    print(f"\nStructure Analysis for {data['formula']}")
    print(f"Space Group: {space_group}")
    print(f"Point Group: {point_group}")
    print(f"Number of Clusters: {data['num_clusters']}")


def visualize_command(args):
    """
    Execute the visualize command to create visualizations.
    
    Parameters:
        args (Namespace): Command-line arguments
    """
    # Load data
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Create structure from dict
    structure = Structure.from_dict(data["structure"])
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = os.path.splitext(os.path.basename(args.json_file))[0]
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        fig = visualize_clusters_in_compound(structure, data["clusters"])
        
        if args.show:
            plt.show()
        
        plt.savefig(f"{output_prefix}_clusters.png", dpi=300)
        print(f"Saved cluster visualization to {output_prefix}_clusters.png")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = get_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == 'find':
        find_command(args)
    elif args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 