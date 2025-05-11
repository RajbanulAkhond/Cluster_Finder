"""
Command-line interface for cluster_finder package.

This module provides a modern Typer-based CLI for the cluster_finder package.
"""
import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
import matplotlib.pyplot as plt

# Add check for LZMA support early in the CLI
try:
    import lzma
except ImportError:
    from rich.console import Console
    console = Console()
    console.print("\n[bold yellow]⚠️  Warning: Python LZMA support is missing![/bold yellow]")
    console.print("[yellow]This may cause errors when using compression features required by dependencies like pymatgen.[/yellow]")
    console.print("[yellow]See installation instructions for fixing this issue in the documentation or setup.py.[/yellow]\n")

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_MODERN_CLI = True
except ImportError:
    print("Error: This package requires typer and rich packages. Install them using:")
    print("pip install typer rich")
    sys.exit(1)

from .core.graph import create_connectivity_matrix, structure_to_graph
from .core.clusters import (
    find_clusters, 
    analyze_clusters
)
from .core.structure import generate_lattice_with_clusters
from .visualization.visualize import (
    visualize_graph,
    visualize_clusters_in_compound,
    visualize_cluster_lattice
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
from .utils.config_utils import load_config, get_element_combinations
from .analysis.analysis import run_analysis
from .analysis.batch import run_batch_analysis

def validate_input_file(file_path: str, expected_format: Optional[str] = None) -> bool:
    """
    Validate input file existence and format.
    
    Args:
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

# Create Typer app
app = typer.Typer(help="Cluster Finder CLI")
analyze_app = typer.Typer(help="Advanced analysis commands")
app.add_typer(analyze_app, name="analyze")

console = Console()

@app.command("find")
def find_command(
    structure_file: str = typer.Argument(..., help="Structure file (CIF, POSCAR, etc.)"),
    elements: Optional[List[str]] = typer.Option(None, "--elements", "-e", help="Elements to consider for clusters (default: all transition metals)"),
    radius: float = typer.Option(3.5, "--radius", "-r", help="Maximum atom-to-atom distance for cluster search (default: 3.5 Å)"),
    min_size: int = typer.Option(2, "--min-size", "-s", help="Minimum cluster size (default: 2)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file prefix (default: based on input filename)"),
    no_vis: bool = typer.Option(False, "--no-vis", help="Disable visualization"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv, or both")
):
    """Find clusters in a structure."""
    try:
        validate_input_file(structure_file)
        
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(f"\nFound {len(structure)} sites in {structure.composition.reduced_formula}")
        
        # Use default elements if none provided
        elements_list = elements or get_transition_metals()
        
        # Create connectivity matrix
        matrix, indices = create_connectivity_matrix(structure, elements_list, radius)
        
        # Create graph
        graph = structure_to_graph(matrix)
        
        # Find clusters
        clusters = find_clusters(structure, graph, indices, min_size)
        
        # Analyze clusters
        analyzed_clusters = analyze_clusters(clusters, structure.lattice) if clusters else []
        
        # Extract material_id from filename
        filename_base = os.path.basename(structure_file)
        material_id = os.path.splitext(filename_base)[0]
        if not re.match(r'mp-\d+', material_id):
            material_id = filename_base
        
        # Convert clusters to JSON format
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
        }
        
        # Determine output path
        output_base = output or os.path.splitext(structure_file)[0]
        os.makedirs(os.path.dirname(output_base) or '.', exist_ok=True)
        
        # Save results
        if format in ['json', 'both']:
            json_file = f"{output_base}_clusters.json"
            with open(json_file, "w") as f:
                json.dump(result, f, indent=2)
            console.print(f"Saved cluster data to {json_file}")
        
        if format in ['csv', 'both']:
            df = cluster_compounds_dataframe([{
                "material_id": material_id,
                "formula": result["formula"],
                "structure": structure,
                "clusters": analyzed_clusters,
                "num_clusters": result["num_clusters"]
            }])
            if df is not None:
                csv_file = f"{output_base}_clusters.csv"
                export_csv_data(df, csv_file)
                console.print(f"Saved cluster data to {csv_file}")
        
        # Visualize if requested
        if not no_vis and analyzed_clusters:
            fig = visualize_clusters_in_compound(structure, analyzed_clusters)
            png_file = f"{output_base}_clusters.png"
            fig.savefig(png_file, dpi=300)
            console.print(f"Saved cluster visualization to {png_file}")
        
        console.print(f"\n[green]Found {len(analyzed_clusters)} clusters in {structure.composition.reduced_formula}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("analyze")
def analyze_command(
    json_file: str = typer.Argument(..., help="JSON file with cluster data"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file prefix"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, json, or both"),
    export_conventional: bool = typer.Option(False, "--export-conventional", help="Export conventional structure as CIF file")
):
    """Analyze clusters from a previous run."""
    try:
        validate_input_file(json_file, '.json')
        
        # Load data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create structure from dict
        structure = Structure.from_dict(data["structure"])
        
        # Reconstruct clusters
        clusters = []
        for cluster_dict in data["clusters"]:
            sites = []
            for site_dict in cluster_dict.get("sites", []):
                if isinstance(site_dict, dict) and "species" in site_dict:
                    try:
                        sites.append(Structure.from_dict({"lattice": structure.lattice.as_dict(), 
                                                       "sites": [site_dict]}).sites[0])
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not reconstruct site: {e}[/yellow]")
            
            cluster = {
                "size": cluster_dict["size"],
                "average_distance": cluster_dict["average_distance"],
                "centroid": cluster_dict["centroid"],
                "elements": cluster_dict["elements"],
                "sites": sites
            }
            clusters.append(cluster)
        
        # Generate lattice analysis
        conventional_structure, space_group_symbol, point_groups = generate_lattice_with_clusters(structure, clusters)
        
        compound = {
            "material_id": os.path.basename(json_file).split('_')[0],
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
        output_prefix = output or os.path.splitext(os.path.basename(json_file))[0]
        
        # Save results
        if format in ['csv', 'both']:
            csv_file = f"{output_prefix}_analysis.csv"
            export_csv_data(df, csv_file)
            console.print(f"Saved analysis to {csv_file}")
        
        if format in ['json', 'both']:
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
            console.print(f"Saved analysis to {json_file}")
        
        # Export conventional structure if requested
        if export_conventional:
            conventional_cif_file = f"{output_prefix}_conventional.cif"
            export_structure_to_cif(conventional_structure, conventional_cif_file)
            console.print(f"Saved conventional structure to {conventional_cif_file}")
        
        # Print summary
        console.print(f"\n[green]Structure Analysis for {data['formula']}[/green]")
        console.print(f"Space Group: {space_group_symbol}")
        console.print("Point Groups:")
        for cluster_label, point_group in point_groups.items():
            console.print(f"  {cluster_label}: {point_group}")
        console.print(f"Number of Clusters: {len(data['clusters'])}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("visualize")
def visualize_command(
    json_file: str = typer.Argument(..., help="JSON file with cluster data"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file prefix"),
    show: bool = typer.Option(False, "--show", help="Show visualization (requires GUI)"),
    dpi: int = typer.Option(300, "--dpi", help="DPI for saved images (default: 300)"),
    cluster_index: Optional[int] = typer.Option(None, "--cluster-index", "-i", help="Index of specific cluster to visualize (0-based)"),
    rotation: str = typer.Option("45x,30y,0z", "--rotation", help="Rotation parameters for visualization"),
    type: str = typer.Option("cluster", "--type", "-t", help="Type of visualization: cluster, graph, lattice, or all"),
    use_3d: bool = typer.Option(False, "--3d", help="Use 3D visualization for graph")
):
    """Visualize clusters."""
    try:
        validate_input_file(json_file, '.json')
        
        # Load data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create structure from dict
        structure = Structure.from_dict(data["structure"])
        
        # Reconstruct clusters
        clusters = []
        for cluster_dict in data["clusters"]:
            sites = []
            for site_dict in cluster_dict.get("sites", []):
                if isinstance(site_dict, dict) and "species" in site_dict:
                    try:
                        sites.append(Structure.from_dict({"lattice": structure.lattice.as_dict(), 
                                                       "sites": [site_dict]}).sites[0])
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not reconstruct site: {e}[/yellow]")
            
            cluster = {
                "size": cluster_dict["size"],
                "average_distance": cluster_dict["average_distance"],
                "centroid": cluster_dict["centroid"],
                "elements": cluster_dict["elements"],
                "sites": sites
            }
            clusters.append(cluster)
        
        # Determine output prefix
        output_prefix = output or os.path.splitext(os.path.basename(json_file))[0]
        
        # Check cluster index bounds
        if cluster_index is not None and (cluster_index < 0 or cluster_index >= len(clusters)):
            console.print(f"[yellow]Warning: Cluster index {cluster_index} is out of bounds (0-{len(clusters)-1}). Using default.[/yellow]")
            cluster_index = None
        
        # Create the requested visualizations
        vis_types = ['cluster', 'graph', 'lattice'] if type == 'all' else [type]
        
        for vis_type in vis_types:
            try:
                if vis_type == 'cluster':
                    fig = visualize_clusters_in_compound(
                        structure, 
                        clusters, 
                        cluster_index=cluster_index,
                        rotation=rotation
                    )
                    if fig:
                        cluster_idx = cluster_index if cluster_index is not None else 0
                        png_file = f"{output_prefix}_cluster_{cluster_idx+1}.png"
                        fig.savefig(png_file, dpi=dpi)
                        console.print(f"Saved cluster visualization to {png_file}")
                        
                        if show:
                            plt.figure(fig.number)
                            plt.show(block=False)
                
                elif vis_type == 'graph':
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
                        use_3d=use_3d
                    )
                    
                    dim = "3d" if use_3d else "2d"
                    png_file = f"{output_prefix}_graph_{dim}.png"
                    fig.savefig(png_file, dpi=dpi)
                    console.print(f"Saved graph visualization to {png_file}")
                    
                    if show:
                        plt.figure(fig.number)
                        plt.show(block=False)
                
                elif vis_type == 'lattice':
                    conv_structure_file = f"{output_prefix}_conventional.cif"
                    try:
                        if os.path.exists(conv_structure_file):
                            conventional_structure = Structure.from_file(conv_structure_file)
                            console.print(f"Using existing conventional structure from {conv_structure_file}")
                        else:
                            conventional_structure, _, _ = generate_lattice_with_clusters(structure, clusters)
                            console.print("Generated conventional structure with clusters")
                        
                        fig = visualize_cluster_lattice(conventional_structure, rot=rotation)
                        png_file = f"{output_prefix}_lattice.png"
                        fig.savefig(png_file, dpi=dpi)
                        console.print(f"Saved lattice visualization to {png_file}")
                        
                        if show:
                            plt.figure(fig.number)
                            plt.show(block=False)
                    
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not create lattice visualization: {e}[/yellow]")
            
            except Exception as e:
                console.print(f"[red]Error creating {vis_type} visualization: {e}[/red]")
        
        # Show all figures at once if requested
        if show:
            plt.show()
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("batch")
def batch_command(
    input_dir: str = typer.Argument(..., help="Directory containing structure files"),
    pattern: str = typer.Option("*.cif", "--pattern", "-p", help="Glob pattern for input files"),
    elements: Optional[List[str]] = typer.Option(None, "--elements", "-e", help="Elements to consider for clusters"),
    radius: float = typer.Option(3.5, "--radius", "-r", help="Maximum atom-to-atom distance"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory")
):
    """Process multiple structure files."""
    try:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Directory not found: {input_dir}")
        
        # Create output directory
        output_dir = output or os.path.join(input_dir, 'cluster_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all matching files
        all_results = []
        for file_path in Path(input_dir).glob(pattern):
            console.print(f"\nProcessing {file_path}...")
            
            try:
                # Create args for find_command
                find_args = argparse.Namespace(
                    structure_file=str(file_path),
                    elements=elements,
                    radius=radius,
                    min_size=2,
                    output=os.path.join(output_dir, file_path.stem),
                    no_vis=True,
                    format='both'
                )
                
                # Run find command
                find_command(
                    structure_file=find_args.structure_file,
                    elements=find_args.elements,
                    radius=find_args.radius,
                    min_size=find_args.min_size,
                    output=find_args.output,
                    no_vis=find_args.no_vis,
                    format=find_args.format
                )
                
                # Load results for summary
                with open(f"{find_args.output}_clusters.json") as f:
                    result = json.load(f)
                    all_results.append(result)
                    
            except Exception as e:
                console.print(f"[red]Error processing {file_path}: {e}[/red]")
                continue
        
        # Generate summary
        if all_results:
            summary_file = os.path.join(output_dir, "batch_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(cluster_summary_stat(all_results, all_results))
            console.print(f"\nSaved batch summary to {summary_file}")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("summary")
def summary_command(
    input_file: str = typer.Argument(..., help="JSON or CSV file with cluster data"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for summary"),
    retrieve_missing: bool = typer.Option(False, "--retrieve-missing", help="Retrieve missing properties from Materials Project"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Materials Project API key")
):
    """Generate summary statistics."""
    try:
        validate_input_file(input_file)
        
        # Load data based on file format
        if input_file.lower().endswith('.json'):
            with open(input_file) as f:
                data = json.load(f)
                compounds = [data] if isinstance(data, dict) else data
        else:  # CSV format
            df = import_csv_data(input_file)
            compounds = df.to_dict('records')
        
        # Process compounds to ensure they have required fields
        processed_compounds = []
        for compound in compounds:
            processed = dict(compound)
            
            # Ensure material_id is present
            if "material_id" not in processed:
                # Try to get it from the filename if JSON
                if input_file.lower().endswith('.json'):
                    base_name = os.path.basename(input_file)
                    # Check if filename follows mp-XXXXX pattern
                    mp_match = re.search(r'(mp-\d+)', base_name)
                    if mp_match:
                        processed["material_id"] = mp_match.group(1)
                    else:
                        processed["material_id"] = os.path.splitext(base_name)[0]
                else:
                    processed["material_id"] = f"material_{compounds.index(compound) + 1}"
            
            # Handle missing total_magnetization based on retrieve-missing flag
            if "total_magnetization" not in processed:
                if retrieve_missing:
                    try:
                        console.print(f"Retrieving total_magnetization for {processed['material_id']}...")
                        processed["total_magnetization"] = get_mp_property(
                            processed["material_id"], 
                            "total_magnetization", 
                            api_key
                        )
                        console.print(f"Retrieved total_magnetization: {processed['total_magnetization']}")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not retrieve total_magnetization for {processed['material_id']}: {e}[/yellow]")
                        continue
                else:
                    console.print(f"[yellow]Warning: Missing total_magnetization for {processed['material_id']} and retrieve-missing not enabled[/yellow]")
                    continue
            
            # Process clusters if they exist but need conversion
            if "clusters" in processed and isinstance(processed["clusters"], list):
                for cluster in processed["clusters"]:
                    if isinstance(cluster, dict):
                        if "size" not in cluster:
                            if "sites" in cluster and isinstance(cluster["sites"], list):
                                cluster["size"] = len(cluster["sites"])
                            else:
                                cluster["size"] = 0
            
            processed_compounds.append(processed)
        
        if not processed_compounds:
            console.print("[yellow]Warning: No valid compounds to process after handling missing properties[/yellow]")
            return
        
        # Generate summary
        summary = cluster_summary_stat(processed_compounds, processed_compounds)
        
        # Output summary
        if output:
            with open(output, 'w') as f:
                f.write(summary)
            console.print(f"Saved summary to {output}")
        else:
            console.print(summary)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("rank")
def rank_command(
    input_file: str = typer.Argument(..., help="CSV file with cluster data"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file prefix for ranked data"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Materials Project API key"),
    top: Optional[int] = typer.Option(None, "--top", "-n", help="Show only top N ranked clusters"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, json, or both"),
    custom_props: Optional[List[str]] = typer.Option(None, "--custom-props", help="Custom properties for ranking"),
    prop_weights: Optional[List[str]] = typer.Option(None, "--prop-weights", help="Weights for custom properties (format: prop:weight)"),
    no_default_ranking: bool = typer.Option(False, "--no-default-ranking", help="Disable default ranking criteria")
):
    """Rank clusters based on geometry, symmetry, and stability."""
    try:
        validate_input_file(input_file)
        
        # Load data
        df = import_csv_data(input_file)
        
        # Parse property weights if provided
        custom_weights = {}
        if prop_weights:
            for weight_str in prop_weights:
                prop, weight = weight_str.split(':')
                custom_weights[prop] = float(weight)
        
        # Rank clusters
        ranked_df = rank_clusters(
            df,
            custom_props=custom_props,
            prop_weights=custom_weights if custom_weights else None,
            top_n=top,
            include_default_ranking=not no_default_ranking
        )
        
        # Save results if output path provided
        if output:
            if format in ['csv', 'both']:
                export_csv_data(ranked_df, f"{output}_ranked.csv")
                console.print(f"Saved ranked data to {output}_ranked.csv")
                
            if format in ['json', 'both']:
                with open(f"{output}_ranked.json", 'w') as f:
                    json.dump(ranked_df.to_dict('records'), f, indent=2)
                console.print(f"Saved ranked data to {output}_ranked.json")
        
        # Display top results
        display_columns = ['material_id', 'formula', 'rank_score']
        if 'energy_above_hull' in ranked_df.columns:
            display_columns.append('energy_above_hull')
        if 'highest_point_group' in ranked_df.columns:
            display_columns.append('highest_point_group')
        if 'max_point_group_order' in ranked_df.columns:
            display_columns.append('max_point_group_order')
        
        # Add custom properties to display columns
        if custom_props:
            for prop in custom_props:
                if prop in ranked_df.columns:
                    display_columns.append(prop)
        
        with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            console.print(ranked_df[display_columns].head(5))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("get-material")
def get_material_command(
    material_id: str = typer.Argument(..., help="Materials Project ID (e.g., mp-149)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (default: material_id.cif)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Materials Project API key"),
    conventional: bool = typer.Option(False, "--conventional", help="Export conventional cell instead of primitive cell"),
    no_analysis: bool = typer.Option(False, "--no-analysis", help="Skip structure analysis information")
):
    """Get a material from Materials Project by ID and export as CIF."""
    try:
        console.print(f"\nRetrieving material {material_id} from Materials Project...")
        
        with MPRester(api_key) as mpr:
            try:
                structure = mpr.get_structure_by_material_id(material_id)
                
                # Convert to conventional cell if requested
                if conventional:
                    analyzer = SpacegroupAnalyzer(structure)
                    structure = analyzer.get_conventional_standard_structure()
                    console.print(f"Retrieved conventional structure for {material_id}")
                else:
                    console.print(f"Retrieved primitive structure for {material_id}")
                
                # Determine output path
                output_path = output or f"{material_id}.cif"
                
                # Export structure to CIF
                export_structure_to_cif(structure, output_path)
                console.print(f"Structure exported to {output_path}")
                
                # Display structure analysis information
                if not no_analysis:
                    analyzer = SpacegroupAnalyzer(structure)
                    space_group = analyzer.get_space_group_symbol()
                    point_group = analyzer.get_point_group_symbol()
                    formula = structure.composition.reduced_formula
                    
                    console.print("\nStructure Information:")
                    console.print(f"Formula: {formula}")
                    console.print(f"Space Group: {space_group}")
                    console.print(f"Point Group: {point_group}")
                    console.print(f"Number of Sites: {len(structure)}")
                    console.print(f"Elements: {', '.join(sorted([str(e) for e in structure.composition.elements]))}")
                    
                    try:
                        energy_above_hull = get_mp_property(material_id, 'energy_above_hull', api_key)
                        formation_energy = get_mp_property(material_id, 'formation_energy_per_atom', api_key)
                        band_gap = get_mp_property(material_id, 'band_gap', api_key)
                        
                        console.print("\nMaterial Properties:")
                        console.print(f"Energy Above Hull: {energy_above_hull:.4f} eV/atom")
                        console.print(f"Formation Energy: {formation_energy:.4f} eV/atom")
                        console.print(f"Band Gap: {band_gap:.4f} eV")
                    except Exception:
                        pass
            
            except Exception as e:
                console.print(f"[red]Error retrieving structure: {str(e)}[/red]")
                console.print("[yellow]Make sure you have a valid Materials Project API key set as MAPI_KEY environment variable "
                              "or provided through the --api-key argument.[/yellow]")
                raise typer.Exit(code=1)
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command("version")
def version():
    """Show the version of cluster_finder."""
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("cluster_finder").version
        console.print(f"[bold green]Cluster Finder version: {version}[/bold green]")
    except Exception:
        console.print("[yellow]Could not determine Cluster Finder version[/yellow]")

# Advanced analysis commands
@analyze_app.command("batch")
def advanced_batch_analysis(
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Materials Project API key"),
    output_dir: Path = typer.Option("results", "--output-dir", "-o", help="Directory to save outputs"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    tms: Optional[List[str]] = typer.Option(None, "--tms", help="Specific transition metals to analyze (comma-separated list)", callback=lambda x: x[0].split(',') if x else None),
    anions: Optional[List[str]] = typer.Option(None, "--anions", help="Specific anions to analyze (comma-separated list)", callback=lambda x: x[0].split(',') if x else None),
    max_workers: int = typer.Option(4, "--max-workers", "-w", help="Maximum number of parallel system analyses"),
    save_pdf: bool = typer.Option(True, "--no-pdf", help="Do not save PDF reports"),
    save_csv: bool = typer.Option(True, "--no-csv", help="Do not save CSV files"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Number of parallel jobs"),
    use_mpi: bool = typer.Option(False, "--mpi", help="Enable multiprocessing to use all available CPU cores"),
    mpi_cores: Optional[int] = typer.Option(None, "--mpi-cores", "-m", help="Number of CPU cores to use for multiprocessing"),
    show_systems: bool = typer.Option(False, "--show-systems", help="Only show the systems that would be analyzed without running"),
    summary_only: bool = typer.Option(False, "--summary-only", help="Only show the summary table, not individual system results"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Retry previously failed systems from a batch run"),
    load_summary: Optional[str] = typer.Option(None, "--load-summary", help="Path to a previous batch_summary.json to continue from"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress and debug information")
):
    """Run advanced batch analysis with additional options for parallel processing."""
    try:
        from cluster_finder.utils.logger import setup_logging
        
        # Configure logging based on verbose flag
        setup_logging(verbose=verbose)

        # Load configuration
        config_dict = load_config(config) if config else None

        # Handle previous results if retrying
        previous_results = {}
        if load_summary and retry_failed:
            try:
                with open(load_summary, 'r') as f:
                    previous_summary = json.load(f)
                    previous_results = previous_summary.get("results", {})
                console.print(f"[bold blue]Loaded previous summary from {load_summary}[/bold blue]")
            except Exception as e:
                console.print(f"[bold red]Error loading previous summary: {e}[/bold red]")

        # Update config with specific elements if provided
        if tms or anions:
            if config_dict is None:
                config_dict = {}
            if tms:
                config_dict['transition_metals'] = tms
            if anions:
                config_dict['anions'] = anions

        # Get systems to analyze
        systems = get_element_combinations(config_dict)

        if not systems:
            console.print("[red]Error: No valid systems to analyze. Please check your TM and anion selections.[/red]")
            raise typer.Exit(code=1)

        # Filter failed systems if retrying
        if retry_failed and previous_results:
            failed_systems = []
            for system_name, result in previous_results.items():
                if result.get("status") == "error":
                    elements = system_name.split("-")
                    if len(elements) == 2:
                        failed_systems.append(elements)
            if failed_systems:
                systems = failed_systems
                console.print(f"[bold yellow]Will retry {len(systems)} failed systems[/bold yellow]")
            else:
                console.print("[bold yellow]No failed systems found to retry[/bold yellow]")

        # Show systems to analyze
        console.print(f"[bold green]Preparing analysis for {len(systems)} TM-anion systems[/bold green]")

        table = Table(title="Systems to Analyze")
        table.add_column("System", style="cyan")
        table.add_column("Transition Metal", style="green")
        table.add_column("Anion", style="yellow")

        for system in systems:
            table.add_row(f"{system[0]}-{system[1]}", system[0], system[1])

        console.print(table)

        if show_systems:
            console.print("[yellow]This was a dry run. Use without --show-systems to execute analysis.[/yellow]")
            return

        # Run batch analysis
        result = run_batch_analysis(
            api_key=api_key,
            output_dir=output_dir,
            config_path=config,
            specific_tms=tms,
            specific_anions=anions,
            max_workers=max_workers,
            save_pdf=save_pdf,
            save_csv=save_csv,
            n_jobs_per_analysis=n_jobs,
            verbose=verbose,
            use_mpi=use_mpi,
            mpi_cores=mpi_cores
        )

        # Print results
        if summary_only:
            console.print("[bold green]Batch Analysis Summary[/bold green]")
            summary_table = Table(title="Analysis Results")
            summary_table.add_column("Parameter", style="cyan")
            summary_table.add_column("Value", style="green")

            summary_table.add_row("Total Systems", str(len(systems)))
            summary_table.add_row("Completed Systems", str(result.get("completed_systems", 0)))
            summary_table.add_row("Failed Systems", str(result.get("failed_systems", 0)))
            summary_table.add_row("Total Time (seconds)", f"{result.get('total_time_seconds', 0):.2f}")
            summary_table.add_row("Output Directory", str(output_dir))

            console.print(summary_table)
            return

        # Show detailed results if not summary only
        results_table = Table(title="System-Specific Results")
        results_table.add_column("System", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Compounds", style="yellow")
        results_table.add_column("With Clusters", style="yellow")
        results_table.add_column("Time (s)", style="blue")

        for system_name, system_result in result.get("results", {}).items():
            status = system_result.get("status", "unknown")
            status_style = "green" if status == "completed" else "red"

            results_table.add_row(
                system_name,
                f"[{status_style}]{status}[/{status_style}]",
                str(system_result.get("compounds_count", "N/A")),
                str(system_result.get("compounds_with_clusters_count", "N/A")),
                f"{system_result.get('time_taken', 0):.2f}"
            )

        console.print(results_table)

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@analyze_app.command("config")
def show_config(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the configuration")
):
    """Display or export the current configuration."""
    try:
        config = load_config(config_path) if config_path else {}
        
        if output:
            with open(output, 'w') as f:
                json.dump(config, f, indent=2)
            console.print(f"[green]Configuration saved to {output}[/green]")
        else:
            console.print("[bold]Current Configuration:[/bold]")
            console.print(json.dumps(config, indent=2))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

def main():
    """Main entry point for the command-line interface."""
    app()