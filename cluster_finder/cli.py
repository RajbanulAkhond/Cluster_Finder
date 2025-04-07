"""
Command-line interface for cluster_finder package.

This module provides a command-line interface to use the functionality
of the cluster_finder package.
"""

import os
import sys
import re
import argparse
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

from .core.graph import create_connectivity_matrix, structure_to_graph
from .core.clusters import (
    find_clusters, 
    analyze_clusters
)
from .core.structure import generate_lattice_with_clusters
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
from .utils.helpers import get_transition_metals, get_mp_property
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
    analyze_parser.add_argument('--export-conventional', action='store_true', 
                               help='Export conventional structure as CIF file')
    
    # Get a material from Materials Project by ID
    get_material_parser = subparsers.add_parser('get-material', help='Get a material from Materials Project by ID and export as CIF')
    get_material_parser.add_argument('material_id', help='Materials Project ID (e.g., mp-149)')
    get_material_parser.add_argument('--output', '-o', help='Output file path (default: material_id.cif)')
    get_material_parser.add_argument('--api-key', help='Materials Project API key (optional, will use environment variable if not provided)')
    get_material_parser.add_argument('--conventional', action='store_true', help='Export conventional cell instead of primitive cell')
    get_material_parser.add_argument('--no-analysis', action='store_true', help='Skip structure analysis information')
    
    # Visualize clusters
    vis_parser = subparsers.add_parser('visualize', help='Visualize clusters')
    vis_parser.add_argument('json_file', help='JSON file with cluster data')
    vis_parser.add_argument('--output', '-o', help='Output file prefix')
    vis_parser.add_argument('--show', action='store_true', help='Show visualization (requires GUI)')
    vis_parser.add_argument('--dpi', type=int, default=300, help='DPI for saved images (default: 300)')
    vis_parser.add_argument('--cluster-index', '-i', type=int, help='Index of specific cluster to visualize (0-based)')
    vis_parser.add_argument('--rotation', type=str, default="45x,30y,0z", 
                           help='Rotation parameters for visualization (default: "45x,30y,0z")')
    vis_parser.add_argument('--type', choices=['cluster', 'graph', 'lattice', 'all'], default='cluster',
                          help='Type of visualization: cluster (default), graph (connectivity), lattice (conventional cell), or all')
    vis_parser.add_argument('--3d', dest='use_3d', action='store_true', help='Use 3D visualization for graph (only applies to graph type)')
    
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
    summary_parser.add_argument('--retrieve-missing', action='store_true', 
                              help='Retrieve missing properties (e.g., total_magnetization) from Materials Project')
    summary_parser.add_argument('--api-key', help='Materials Project API key (used if retrieve-missing is enabled)')
    
    # Rank clusters based on various criteria
    rank_parser = subparsers.add_parser('rank', help='Rank clusters based on geometry, symmetry, and stability')
    rank_parser.add_argument('input_file', help='CSV file with cluster data')
    rank_parser.add_argument('--output', '-o', help='Output file prefix for ranked data')
    rank_parser.add_argument('--api-key', help='Materials Project API key (optional, will use environment variable if not provided)')
    rank_parser.add_argument('--top', type=int, default=None, help='Show only top N ranked clusters')
    rank_parser.add_argument('--format', choices=['csv', 'json', 'both'], default='csv', help='Output format (default: csv)')
    rank_parser.add_argument('--custom-props', nargs='+', help='Custom properties to include in ranking (e.g., formation_energy_per_atom density)')
    rank_parser.add_argument('--prop-weights', nargs='+', help='Weights for custom properties (format: prop_name:weight, e.g., formation_energy_per_atom:-1.0)')
    rank_parser.add_argument('--no-default-ranking', action='store_true', help='Disable default ranking criteria')
    
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
        
        # Extract material_id from filename if it looks like an MP ID
        filename_base = os.path.basename(args.structure_file)
        material_id = os.path.splitext(filename_base)[0]
        # If the filename looks like an MP id (mp-XXXXX), use it directly
        if not re.match(r'mp-\d+', material_id):
            # Otherwise, use the filename as a generic ID
            material_id = filename_base
        
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
            "material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "structure": structure.as_dict(),
            "clusters": json_clusters,
            "num_clusters": len(analyzed_clusters)
            # No default total_magnetization
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
                "material_id": material_id,
                "formula": result["formula"],
                "structure": structure,  # Pass the Structure object directly
                "clusters": analyzed_clusters,  # Pass the original clusters with PeriodicSite objects
                "num_clusters": result["num_clusters"]
                # No default total_magnetization
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
        
        # Reconstruct cluster objects from JSON data
        clusters = []
        for cluster_dict in data["clusters"]:
            # Create PeriodicSite objects from site dictionaries
            sites = []
            for site_dict in cluster_dict.get("sites", []):
                if isinstance(site_dict, dict) and "species" in site_dict:
                    try:
                        sites.append(Structure.from_dict({"lattice": structure.lattice.as_dict(), 
                                                         "sites": [site_dict]}).sites[0])
                    except Exception as e:
                        print(f"Warning: Could not reconstruct site: {e}")
            
            # Build a proper cluster object
            cluster = {
                "size": cluster_dict["size"],
                "average_distance": cluster_dict["average_distance"],
                "centroid": cluster_dict["centroid"],
                "elements": cluster_dict["elements"],
                "sites": sites
            }
            clusters.append(cluster)
        
        # Use generate_lattice_with_clusters function for point group analysis
        conventional_structure, space_group_symbol, point_groups = generate_lattice_with_clusters(structure, clusters)
        
        # Create compound object
        compound = {
            "material_id": os.path.basename(args.json_file).split('_')[0],
            "formula": data["formula"],
            "clusters": clusters,
            "structure": structure,
            "conventional_structure": conventional_structure,
            "space_group": space_group_symbol,
            "point_groups": point_groups
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
                    "space_group": space_group_symbol,
                    "point_groups": point_groups,
                    "conventional_structure": conventional_structure.as_dict(),
                    "num_clusters": len(data["clusters"]),
                    "clusters": data["clusters"]
                }, f, indent=2)
            print(f"Saved analysis to {json_file}")
        
        # Export conventional structure if requested
        if args.export_conventional:
            conventional_cif_file = f"{output_prefix}_conventional.cif"
            export_structure_to_cif(conventional_structure, conventional_cif_file)
            print(f"Saved conventional structure to {conventional_cif_file}")
        
        # Print summary
        print(f"\nStructure Analysis for {data['formula']}")
        print(f"Space Group: {space_group_symbol}")
        
        # Print point groups information
        print("Point Groups:")
        for cluster_label, point_group in point_groups.items():
            print(f"  {cluster_label}: {point_group}")
        
        print(f"Number of Clusters: {len(data['clusters'])}")
        
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
        
        # Reconstruct cluster objects from JSON
        clusters = []
        for cluster_dict in data["clusters"]:
            # Create PeriodicSite objects from site dictionaries
            sites = []
            for site_dict in cluster_dict.get("sites", []):
                if isinstance(site_dict, dict) and "species" in site_dict:
                    try:
                        sites.append(Structure.from_dict({"lattice": structure.lattice.as_dict(), 
                                                         "sites": [site_dict]}).sites[0])
                    except Exception as e:
                        print(f"Warning: Could not reconstruct site: {e}")
            
            # Build a proper cluster object
            cluster = {
                "size": cluster_dict["size"],
                "average_distance": cluster_dict["average_distance"],
                "centroid": cluster_dict["centroid"],
                "elements": cluster_dict["elements"],
                "sites": sites
            }
            clusters.append(cluster)
        
        # Determine output prefix
        output_prefix = args.output or os.path.splitext(os.path.basename(args.json_file))[0]
        
        # Import needed visualization modules
        import matplotlib.pyplot as plt
        from .visualization.visualize import visualize_cluster_lattice
        
        # Check cluster index bounds
        if args.cluster_index is not None and (args.cluster_index < 0 or args.cluster_index >= len(clusters)):
            print(f"Warning: Cluster index {args.cluster_index} is out of bounds (0-{len(clusters)-1}). Using default.")
            args.cluster_index = None
        
        # Validate visualization type choice
        vis_types = []
        if args.type == 'all':
            vis_types = ['cluster', 'graph', 'lattice']
        else:
            vis_types = [args.type]
        
        # Create the requested visualizations
        for vis_type in vis_types:
            try:
                if vis_type == 'cluster':
                    # Cluster visualization
                    fig = visualize_clusters_in_compound(
                        structure, 
                        clusters, 
                        cluster_index=args.cluster_index, 
                        rotation=args.rotation
                    )
                    if fig:
                        cluster_idx = args.cluster_index if args.cluster_index is not None else 0
                        png_file = f"{output_prefix}_cluster_{cluster_idx+1}.png"
                        fig.savefig(png_file, dpi=args.dpi)
                        print(f"Saved cluster visualization to {png_file}")
                        
                        if args.show:
                            plt.figure(fig.number)
                            plt.show(block=False)
                    
                elif vis_type == 'graph':
                    # Graph visualization - need to reconstruct connectivity
                    # Get transition metal elements from clusters
                    tm_elements = set()
                    for cluster in clusters:
                        for site in cluster["sites"]:
                            if hasattr(site, 'specie'):
                                tm_elements.add(site.specie.symbol)
                    
                    # Create connectivity matrix
                    matrix, indices = create_connectivity_matrix(structure, list(tm_elements))
                    
                    # Create graph
                    graph = structure_to_graph(matrix)
                    
                    # Create visualization
                    fig = visualize_graph(
                        graph, 
                        structure, 
                        indices, 
                        material_id=data.get("material_id", ""), 
                        formula=data.get("formula", ""),
                        use_3d=args.use_3d
                    )
                    
                    # Save graph visualization
                    dim = "3d" if args.use_3d else "2d"
                    png_file = f"{output_prefix}_graph_{dim}.png"
                    fig.savefig(png_file, dpi=args.dpi)
                    print(f"Saved graph visualization to {png_file}")
                    
                    if args.show:
                        plt.figure(fig.number)
                        plt.show(block=False)
                
                elif vis_type == 'lattice':
                    # Lattice visualization - first need to get or create conventional structure
                    # Check if we already have a conventional structure from previous analysis
                    conv_structure_file = f"{output_prefix}_conventional.cif"
                    try:
                        if os.path.exists(conv_structure_file):
                            conventional_structure = Structure.from_file(conv_structure_file)
                            print(f"Using existing conventional structure from {conv_structure_file}")
                        else:
                            # Generate conventional structure with clusters
                            conventional_structure, _, _ = generate_lattice_with_clusters(structure, clusters)
                            print("Generated conventional structure with clusters")
                        
                        # Create lattice visualization
                        fig = visualize_cluster_lattice(conventional_structure, rot=args.rotation)
                        
                        # Save lattice visualization
                        png_file = f"{output_prefix}_lattice.png"
                        fig.savefig(png_file, dpi=args.dpi)
                        print(f"Saved lattice visualization to {png_file}")
                        
                        if args.show:
                            plt.figure(fig.number)
                            plt.show(block=False)
                    
                    except Exception as e:
                        print(f"Warning: Could not create lattice visualization: {e}")
                        
            except Exception as e:
                print(f"Error creating {vis_type} visualization: {e}")
        
        # Show all figures at once if requested and not already shown
        if args.show:
            plt.show()
            
    except Exception as e:
        print(f"Error in visualize command: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
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
        
        # Process compounds to ensure they have required fields
        processed_compounds = []
        for compound in compounds:
            # Make a copy to avoid modifying the original
            processed = dict(compound)
            
            # Ensure material_id is present
            if "material_id" not in processed:
                # Try to get it from the filename if JSON
                if args.input_file.lower().endswith('.json'):
                    base_name = os.path.basename(args.input_file)
                    # Check if filename follows mp-XXXXX pattern
                    mp_match = re.search(r'(mp-\d+)', base_name)
                    if mp_match:
                        processed["material_id"] = mp_match.group(1)
                    else:
                        # Use the filename without extension as a fallback
                        processed["material_id"] = os.path.splitext(base_name)[0]
                else:
                    # Generate a generic ID
                    processed["material_id"] = f"material_{compounds.index(compound) + 1}"
            
            # Handle missing total_magnetization based on retrieve-missing flag
            if "total_magnetization" not in processed:
                if args.retrieve_missing:
                    try:
                        print(f"Retrieving total_magnetization for {processed['material_id']}...")
                        # This is the key line that needed fixing - make sure we call get_mp_property
                        # when retrieve_missing is True, regardless of material ID format
                        processed["total_magnetization"] = get_mp_property(
                            processed["material_id"], 
                            "total_magnetization", 
                            args.api_key
                        )
                        print(f"Retrieved total_magnetization: {processed['total_magnetization']}")
                    except Exception as e:
                        print(f"Warning: Could not retrieve total_magnetization for {processed['material_id']}: {e}")
                        # Skip this compound if we can't retrieve the required property
                        continue
                else:
                    # Skip this compound if missing required property and not retrieving
                    print(f"Warning: Missing total_magnetization for {processed['material_id']} and retrieve-missing not enabled")
                    continue
            
            # Process clusters if they exist but need conversion
            if "clusters" in processed and isinstance(processed["clusters"], list):
                # Ensure clusters have all required fields
                for cluster in processed["clusters"]:
                    if isinstance(cluster, dict):
                        if "size" not in cluster:
                            # Try to determine size from sites if available
                            if "sites" in cluster and isinstance(cluster["sites"], list):
                                cluster["size"] = len(cluster["sites"])
                            else:
                                cluster["size"] = 0
            
            processed_compounds.append(processed)
        
        if not processed_compounds:
            print("Warning: No valid compounds to process after handling missing properties")
            return
        
        # Generate summary using the processed compounds
        summary = cluster_summary_stat(processed_compounds, processed_compounds)
        
        # Output summary
        if args.output:
            with open(args.output, 'w') as f:
                f.write(summary)
            print(f"Saved summary to {args.output}")
        else:
            print(summary)
            
    except Exception as e:
        print(f"Error in summary command: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def rank_command(args):
    """Rank clusters based on geometry, symmetry, and stability (energy above hull)."""
    try:
        # Validate input file
        validate_input_file(args.input_file, '.csv')
        
        print(f"\nRanking clusters from {args.input_file}...")
        
        # Parse custom property weights if provided
        prop_weights = None
        if args.prop_weights:
            prop_weights = {}
            for weight_str in args.prop_weights:
                if ':' in weight_str:
                    prop_name, weight = weight_str.split(':')
                    try:
                        prop_weights[prop_name] = float(weight)
                    except ValueError:
                        print(f"Warning: Invalid weight format for {prop_name}: {weight}. Using default weight.")
        
        print("This may take some time if retrieving data from the Materials Project API.")
        
        # Rank the clusters with the enhanced capabilities
        ranked_df = rank_clusters(
            data_source=args.input_file,
            api_key=args.api_key,
            custom_props=args.custom_props,
            prop_weights=prop_weights,
            include_default_ranking=not args.no_default_ranking
        )
        
        # Limit to top N if specified
        if args.top and args.top > 0:
            ranked_df = ranked_df.head(args.top)
        
        # Determine output prefix
        output_prefix = args.output or os.path.splitext(os.path.basename(args.input_file))[0] + "_ranked"
        
        # Save results
        if args.format in ['csv', 'both']:
            csv_file = f"{output_prefix}_clusters.csv"
            export_csv_data(ranked_df, csv_file)
            print(f"Saved ranked data to {csv_file}")
        
        if args.format in ['json', 'both']:
            json_file = f"{output_prefix}_clusters.json"
            with open(json_file, 'w') as f:
                # Convert DataFrame to dict for JSON serialization, handling potential non-serializable objects
                json_data = []
                for _, row in ranked_df.iterrows():
                    row_dict = {col: row[col] for col in ranked_df.columns if not isinstance(row[col], (np.ndarray, list))}
                    # Convert special types
                    for col in ranked_df.columns:
                        if isinstance(row[col], np.ndarray):
                            row_dict[col] = row[col].tolist()
                        elif isinstance(row[col], list) and col != 'average_distance':
                            row_dict[col] = str(row[col])
                    json_data.append(row_dict)
                json.dump(json_data, f, indent=2)
            print(f"Saved ranked data to {json_file}")
        
        # Display summary of ranking
        print(f"\nRanked {len(ranked_df)} clusters based on:")
        if not args.no_default_ranking:
            print("- Geometric properties (minimum average distance)")
            print("- Symmetry (point group and space group order)")
            print("- Stability (energy above hull)")
        
        # Show custom properties if any
        if args.custom_props:
            print("- Custom properties:", ", ".join(args.custom_props))
        
        # Show top 5 ranked clusters
        print("\nTop ranked clusters:")
        display_columns = ['material_id', 'formula', 'rank_score']
        if 'energy_above_hull' in ranked_df.columns:
            display_columns.append('energy_above_hull')
        if 'highest_point_group' in ranked_df.columns:
            display_columns.append('highest_point_group')
        if 'max_point_group_order' in ranked_df.columns:
            display_columns.append('max_point_group_order')
        
        # Add custom properties to display columns
        if args.custom_props:
            for prop in args.custom_props:
                if prop in ranked_df.columns:
                    display_columns.append(prop)
        
        with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            print(ranked_df[display_columns].head(5))
        
    except Exception as e:
        print(f"Error in rank command: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_material_command(args):
    """Execute the get-material command to retrieve a structure from Materials Project."""
    try:
        print(f"\nRetrieving material {args.material_id} from Materials Project...")
        
        # Initialize MPRester with the provided API key or from environment variable
        with MPRester(args.api_key) as mpr:
            try:
                # Get the structure from Materials Project
                structure = mpr.get_structure_by_material_id(args.material_id)
                
                # Convert to conventional cell if requested
                if args.conventional:
                    analyzer = SpacegroupAnalyzer(structure)
                    structure = analyzer.get_conventional_standard_structure()
                    print(f"Retrieved conventional structure for {args.material_id}")
                else:
                    print(f"Retrieved primitive structure for {args.material_id}")
                
                # Determine output path
                output_path = args.output or f"{args.material_id}.cif"
                
                # Export structure to CIF
                export_structure_to_cif(structure, output_path)
                print(f"Structure exported to {output_path}")
                
                # Display structure analysis information
                if not args.no_analysis:
                    analyzer = SpacegroupAnalyzer(structure)
                    space_group = analyzer.get_space_group_symbol()
                    point_group = analyzer.get_point_group_symbol()
                    formula = structure.composition.reduced_formula
                    
                    print("\nStructure Information:")
                    print(f"Formula: {formula}")
                    print(f"Space Group: {space_group}")
                    print(f"Point Group: {point_group}")
                    print(f"Number of Sites: {len(structure)}")
                    print(f"Elements: {', '.join(sorted([str(e) for e in structure.composition.elements]))}")
                    
                    # Try to get additional properties if available
                    try:
                        energy_above_hull = get_mp_property(args.material_id, 'energy_above_hull', args.api_key)
                        formation_energy = get_mp_property(args.material_id, 'formation_energy_per_atom', args.api_key)
                        band_gap = get_mp_property(args.material_id, 'band_gap', args.api_key)
                        
                        print("\nMaterial Properties:")
                        print(f"Energy Above Hull: {energy_above_hull:.4f} eV/atom")
                        print(f"Formation Energy: {formation_energy:.4f} eV/atom")
                        print(f"Band Gap: {band_gap:.4f} eV")
                    except Exception as e:
                        # Not all properties may be available, so we'll just ignore any errors
                        pass
            
            except Exception as e:
                print(f"Error retrieving structure: {str(e)}", file=sys.stderr)
                print("Make sure you have a valid Materials Project API key set as MAPI_KEY environment variable "
                      "or provided through the --api-key argument.", file=sys.stderr)
                sys.exit(1)
                
    except Exception as e:
        print(f"Error in get_material command: {str(e)}", file=sys.stderr)
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
        elif args.command == 'rank':
            rank_command(args)
        elif args.command == 'get-material':
            get_material_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()