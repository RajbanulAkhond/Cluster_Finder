#!/usr/bin/env python
"""
This script demonstrates the full workflow of the cluster_finder package:
1. Search Materials Project for compounds with Nb and Cl
2. Process compounds to identify clusters
3. Generate lattices with clusters and create supercells
4. Classify dimensionality of compounds
5. Add materials properties and rank clusters
6. Generate visualizations and save final output to PDF and CSV files
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pymatgen.core.structure import Structure
import networkx as nx
import ast
import logging
from typing import List, Dict, Any, Optional

# Import cluster_finder package
import cluster_finder as cf
from cluster_finder.utils.helpers import (
    search_transition_metal_compounds,
    get_transition_metals,
    get_mp_properties_batch  # Import the new batch function
)
from cluster_finder.core.clusters import get_compounds_with_clusters
from cluster_finder.core.structure import (
    generate_lattice_with_clusters,
    generate_supercell
)
from cluster_finder.analysis.postprocess import (
    classify_dimensionality,
    rank_clusters
)
from cluster_finder.analysis.dataframe import (
    cluster_compounds_dataframe,
    postprocessed_clusters_dataframe
)
from cluster_finder.visualization.visualize import (
    visualize_clusters_in_compound,
    visualize_cluster_lattice,
    visualize_graph
)
from cluster_finder.io.fileio import export_csv_data

# Import rich formatting utilities
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler

# Import logger configuration but create our own console and configure logging
from cluster_finder.utils.logger import get_logger
from cluster_finder.utils.async_utils import get_api_key

# Create a dedicated console for this script with full output enabled
console = Console(highlight=True)

# Configure logging with rich handler for better output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)

# Get logger for this script
logger = get_logger("nb_cl_analysis")

# Define constants
API_KEY = get_api_key()
ELEMENTS = ["Nb", "Cl"]
OUTPUT_PDF = "Nb_Cl_analysis_results.pdf"
RAW_OUTPUT_CSV = "Nb_Cl_analysis_results_raw.csv"
SUMMARY_OUTPUT_CSV = "Nb_Cl_analysis_results_summary.csv"

def main():
    """Main function to execute the workflow."""
    console.print(Panel.fit("Starting [bold cyan]Nb-Cl[/bold cyan] Compounds Analysis", 
                          border_style="green", title="Cluster Finder Analysis"))
    
    # 1. Search for compounds with Nb and Cl
    console.print(f"[bold]Searching for compounds with elements:[/bold] {', '.join(ELEMENTS)}")
    
    # Create progress context to track long-running operations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True  # This helps clear the progress display when done
    ) as progress:
        search_task = progress.add_task("[cyan]Searching Materials Project database...", total=None)
        
        # This already uses the batch API internally after our changes
        compounds = search_transition_metal_compounds(
            elements=ELEMENTS,
            api_key=API_KEY,
            min_elements=2,
            max_elements=4,
            min_magnetization=0.01,  # Set the lower bound of magnetization
            max_magnetization=5     # Set the upper bound of magnetization
        )
        progress.update(search_task, completed=True)
    
    console.print(f"Found [bold green]{len(compounds)}[/bold green] compounds containing {', '.join(ELEMENTS)}")
    
    # Get list of transition metals for filtering
    transition_metals = get_transition_metals()
    console.print(f"Using transition metals: [italic]{', '.join(transition_metals)}[/italic]")
    
    # 2. Process compounds to find clusters
    console.print("\n[bold]Processing compounds to identify clusters...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        cluster_task = progress.add_task("[cyan]Finding clusters in compounds...", total=None)
        compounds_with_clusters = get_compounds_with_clusters(
            compounds,
            transition_metals,
            primary_transition_metal=ELEMENTS[0]  # Ensure clusters contain the primary TM (Nb)
        )
        progress.update(cluster_task, completed=True)
    
    # Filter to keep only compounds that have clusters
    compounds_with_clusters = [comp for comp in compounds_with_clusters if comp["clusters"]]
    console.print(f"Found [bold green]{len(compounds_with_clusters)}[/bold green] compounds with clusters")
 
    # Create a dataframe from compounds using the function from dataframe.py
    console.print("\n[bold]Creating compounds dataframe...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        df_task = progress.add_task("[cyan]Processing compound data...", total=None)
        compounds_df = cluster_compounds_dataframe(compounds_with_clusters, compound_system="-".join(ELEMENTS), verbose=True)
        progress.update(df_task, completed=True)
    
    # Post-process the ranked dataframe to add additional calculated properties
    console.print("\n[bold]Post-processing compounds dataframe...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        postproc_task = progress.add_task("[cyan]Calculating additional properties...", total=None)
        processed_df = postprocessed_clusters_dataframe(compounds_df)
        progress.update(postproc_task, completed=True)

    # Batch retrieve additional properties for all materials at once
    console.print("\n[bold]Retrieving additional materials properties in batch...[/bold]")
    
    # Extract all unique material IDs
    material_ids = processed_df['material_id'].unique().tolist()
    
    # Define properties to retrieve
    properties_to_get = ["energy_above_hull", "formation_energy_per_atom", "band_gap", "total_magnetization"]
    
    # Show what properties we're getting
    properties_table = Table(title="Properties to Retrieve")
    properties_table.add_column("Property", style="cyan")
    properties_table.add_column("Description", style="green")
    property_descriptions = {
        "energy_above_hull": "Thermodynamic stability (eV/atom)",
        "formation_energy_per_atom": "Formation energy (eV/atom)",
        "band_gap": "Electronic band gap (eV)",
        "total_magnetization": "Total magnetic moment (μB)"
    }
    for prop in properties_to_get:
        properties_table.add_row(prop, property_descriptions.get(prop, ""))
    console.print(properties_table)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        props_task = progress.add_task(f"[cyan]Fetching properties for {len(material_ids)} materials...", total=None)
        # Use our new batch function to get all properties at once
        properties_dict = get_mp_properties_batch(material_ids, properties_to_get, API_KEY)
        progress.update(props_task, completed=True)
    
    # Add properties to the dataframe
    prop_counts = {}
    for property_name in properties_to_get:
        # Create a temporary dictionary for mapping material_id to property value
        property_map = {}
        for material_id, props in properties_dict.items():
            if property_name in props:
                property_map[material_id] = props[property_name]
        
        # Update the dataframe with the property values
        if property_map:
            processed_df[property_name] = processed_df['material_id'].map(property_map)
            prop_counts[property_name] = len(property_map)
    
    # Show how many properties were added
    props_result = Table(title="Retrieved Properties")
    props_result.add_column("Property", style="cyan")
    props_result.add_column("Materials with Data", style="green", justify="right")
    for prop, count in prop_counts.items():
        props_result.add_row(prop, str(count))
    console.print(props_result)

    # Rank clusters using the existing rank_clusters function
    console.print("\n[bold]Ranking clusters...[/bold]")
    
    # Handle the ranking without a progress bar since it produces log messages
    # This prevents the overlap issue
    console.print("[cyan]Running cluster ranking calculations... (this may produce log messages)[/cyan]")
    
    # Execute the rank_clusters function directly
    ranked_df = rank_clusters(
        data_source=processed_df,
        api_key=API_KEY,  # API key still needed for internal functions
        custom_props=["symmetry", "energy_above_hull"],
        prop_weights={
            #"formation_energy_per_atom": -1.0,  # Lower formation energy is better
            "symmetry": 0.0,  # Point group has no effect on ranking
            "energy_above_hull": -2.0  # Lower energy above hull is better
        },
        include_default_ranking=True  # Include default criteria like min_avg_distance, point_group_order, etc.
    )
    console.print("[green]✓[/green] Cluster ranking completed")

    # Create a summary dataframe by dropping specific columns
    summary_df = ranked_df.drop(['magnetization', 'conventional_cluster_lattice', 'cluster_sites','point_groups_dict',
                                'max_point_group_order','highest_point_group','space_group_order'], 
                                axis=1, errors='ignore')
    
    # Save the summary dataframe to CSV
    console.print(f"\n[bold]Saving summary results to {SUMMARY_OUTPUT_CSV}...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        csv_task = progress.add_task("[cyan]Exporting data to CSV...", total=None)
        export_csv_data(summary_df, SUMMARY_OUTPUT_CSV)
        progress.update(csv_task, completed=True)
    
    # Create a dictionary to store processed data for visualization - more efficiently
    console.print("\n[bold]Processing data for visualization...[/bold]")
    processed_data = {}

    top_compounds = ranked_df.head(10)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        vis_task = progress.add_task("[cyan]Preparing visualization data...", total=len(top_compounds))
        
        for idx, row in top_compounds.iterrows():
            progress.update(vis_task, advance=1, description=f"[cyan]Processing {idx+1}/{len(top_compounds)} compounds...")
            
            # Access row data correctly (row is a tuple in some pandas versions)
            # Convert row to Series to avoid tuple indexing issue
            row_data = pd.Series(row)
            material_id = row_data['material_id']
            
            # Get the compound data from compounds_df for correct structure and clusters
            compound_data = compounds_df[compounds_df['material_id'] == material_id]
            if compound_data.empty:
                logger.warning(f"Could not find compound data for {material_id}")
                continue
                
            # Extract structure data
            structure_data = compound_data.iloc[0]['structure']
            
            # Convert structure if needed - more efficiently
            try:
                # More direct conversion to Structure object
                if isinstance(structure_data, dict):
                    structure = Structure.from_dict(structure_data)
                elif isinstance(structure_data, str):
                    try:
                        # Try to parse as JSON string
                        import json
                        structure_dict = json.loads(structure_data)
                        structure = Structure.from_dict(structure_dict)
                    except json.JSONDecodeError:
                        try:
                            # Fallback to ast.literal_eval
                            structure_dict = ast.literal_eval(structure_data)
                            structure = Structure.from_dict(structure_dict)
                        except:
                            # Last resort - try as JSON string
                            structure = Structure.from_str(structure_data, fmt="json")
                else:
                    structure = structure_data
                    
                # Get cluster data - more directly
                clusters = []
                
                # Directly use cluster_sites and related data from the row
                # This avoids complex conditional logic
                cluster_sites_data = row_data['cluster_sites']
                cluster_sizes = row_data['cluster_sizes']
                average_distance = row_data['average_distance']
                
                # Handle different types of data (string vs. list)
                if isinstance(cluster_sites_data, str):
                    cluster_sites_data = ast.literal_eval(cluster_sites_data)
                if isinstance(cluster_sizes, str):
                    cluster_sizes = ast.literal_eval(cluster_sizes)
                if isinstance(average_distance, str):
                    average_distance = ast.literal_eval(average_distance)
                
                # Reconstruct clusters list - more efficiently with minimal conversions
                clusters = []
                for i, (size, avg_dist, sites) in enumerate(zip(cluster_sizes, average_distance, cluster_sites_data)):
                    # Convert site dictionaries to Site objects
                    from pymatgen.core.sites import PeriodicSite
                    sites_objects = [PeriodicSite.from_dict(site) for site in sites]
                        
                    clusters.append({
                        'size': size,
                        'average_distance': avg_dist,
                        'sites': sites_objects,
                        'label': f'X{i}'
                    })
                
                # Get graph data if available - unchanged
                graph = None
                if 'graph' in compound_data.columns:
                    graph = compound_data.iloc[0]['graph']
                
                # Get transition metal indices for visualization - unchanged
                transition_metals = get_transition_metals()
                tm_indices = [i for i, site in enumerate(structure) 
                            if site.specie.symbol in transition_metals]
                
                # Store essential data for visualization - skip unnecessary properties
                processed_data[material_id] = {
                    "structure": structure,
                    "clusters": clusters,
                    "graph": graph,
                    "formula": row['formula'],
                    "space_group": row['space_group'],
                    "point_groups": row['point_groups'],
                    "dimensionality": row['predicted_dimentionality'],
                    "energy_above_hull": row.get('energy_above_hull'),
                    "formation_energy_per_atom": row.get('formation_energy_per_atom'),
                    "rank_score": row.get('rank_score'),
                    "tm_indices": tm_indices
                }
                
                # Generate a supercell for visualization - more efficiently
                try:
                    # Generate lattice structure directly
                    lattice_structure, space_group, _ = generate_lattice_with_clusters(structure, clusters)
                    processed_data[material_id]["lattice_structure"] = lattice_structure
                    processed_data[material_id]["space_group"] = space_group
                    
                    # Generate a separate supercell for dimensionality analysis only - don't use for visualization
                    dimensionality_supercell = generate_supercell(lattice_structure, supercell_matrix=(10, 10, 10))
                    processed_data[material_id]["supercell"] = dimensionality_supercell  # Save with key 'supercell' for visualization
                    
                    # Get singular values for dimensionality visualization
                    _, singular_values = classify_dimensionality(dimensionality_supercell)
                    processed_data[material_id]["singular_values"] = singular_values
                    
                except Exception as e:
                    logger.error(f"Error processing supercell for {material_id}: {e}")
                    
            except Exception as e:
                logger.error(f"Error processing structure data for {material_id}: {e}")
    
    # Create cluster statistics dataframe for visualizations - more efficiently
    console.print("\n[bold]Creating visualization statistics...[/bold]")
    cluster_stats_df = pd.DataFrame([
        {
            "material_id": material_id,
            "formula": data["formula"],
            "num_clusters": len(data["clusters"]),
            "cluster_sizes": [cluster["size"] for cluster in data["clusters"]] if data["clusters"] else [],
            "average_distance": float(np.mean([cluster["average_distance"] for cluster in data["clusters"]])) if data["clusters"] else 0,
            "rank_score": data.get("rank_score", 0)  # Include rank score here
        }
        for material_id, data in processed_data.items()
    ])
    
    # Show summary of what we'll visualize
    if not cluster_stats_df.empty:
        stats_table = Table(title="Compounds for Visualization")
        stats_table.add_column("Material ID", style="cyan")
        stats_table.add_column("Formula", style="green")
        stats_table.add_column("Clusters", justify="right")
        stats_table.add_column("Avg. Distance (Å)", justify="right")
        
        for i, (_, row) in enumerate(cluster_stats_df.iterrows()):
            stats_table.add_row(
                row["material_id"],
                row["formula"],
                str(row["num_clusters"]),
                f"{row['average_distance']:.3f}"
            )
        console.print(stats_table)
    
    # Create a PDF to store all visualizations and results in the requested order
    console.print(f"\n[bold]Generating PDF report to {OUTPUT_PDF}...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        pdf_task = progress.add_task("[cyan]Creating PDF report...", total=None)
        
        with PdfPages(OUTPUT_PDF) as pdf:
                # Create a title page
                plt.figure(figsize=(8.5, 11))
                plt.text(0.5, 0.5, f"Analysis of {'-'.join(ELEMENTS)} Compounds\nCluster Finder Results", 
                        horizontalalignment='center', verticalalignment='center', fontsize=24)
                plt.axis('off')
                pdf.savefig()
                plt.close()
                
                # 1. Summary Cluster Statistics
                if not cluster_stats_df.empty:
                    # Create a summary page for all cluster statistics
                    plt.figure(figsize=(8.5, 11))
                    plt.text(0.5, 0.95, "Cluster Statistics Summary", 
                            horizontalalignment='center', fontsize=18)
                    plt.text(0.5, 0.90, f"Total compounds analyzed: {len(compounds)}", 
                            horizontalalignment='center', fontsize=14)
                    plt.text(0.5, 0.85, f"Number of compounds with clusters: {len(compounds_with_clusters)}",
                            horizontalalignment='center', fontsize=14)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                    
                    # Create a figure for cluster size and distance distributions
                    plt.figure(figsize=(8.5, 11))
                    
                    # Plot cluster size distribution
                    plt.subplot(211)
                    # Flatten all cluster sizes into a single list
                    all_cluster_sizes = []
                    for sizes in cluster_stats_df["cluster_sizes"]:
                        all_cluster_sizes.extend(sizes)
                    
                    if all_cluster_sizes:
                        plt.hist(all_cluster_sizes, bins=range(min(all_cluster_sizes), max(all_cluster_sizes)+2), align='left')
                        plt.xlabel("Cluster Size (number of atoms)")
                        plt.ylabel("Count")
                        plt.title("Distribution of Individual Cluster Sizes")
                    else:
                        plt.text(0.5, 0.5, "No cluster size data available", 
                                horizontalalignment='center', fontsize=14)
                        plt.title("Distribution of Individual Cluster Sizes")
                    
                    # Plot average distance distribution
                    plt.subplot(212)
                    plt.hist(cluster_stats_df["average_distance"], bins=20)
                    plt.xlabel("Average Distance (Å)")
                    plt.ylabel("Count")
                    plt.title("Distribution of Average Cluster Distances")
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                # Add point group and space group distribution plots
                if 'point_groups' in ranked_df.columns and 'space_group' in ranked_df.columns:
                    plt.figure(figsize=(8.5, 11))
                    
                    # First subplot: Point group distribution
                    plt.subplot(211)
                    
                    # Extract actual point group symbols
                    all_point_groups = []
                    for _, row in ranked_df.iterrows():
                        try:
                            pg_data = row['point_groups']
                            if isinstance(pg_data, str):
                                pg_data = ast.literal_eval(pg_data)
                            if isinstance(pg_data, dict):
                                for point_group_symbol in pg_data.values():
                                    if point_group_symbol and point_group_symbol != 'None':
                                        all_point_groups.append(point_group_symbol)
                        except Exception as e:
                            logger.warning(f"Error processing point groups: {e}")
                    
                    if all_point_groups:
                        pg_counts = pd.Series(all_point_groups).value_counts()
                        x_labels = [str(name) for name in pg_counts.index]
                        bars = plt.bar(range(len(pg_counts)), pg_counts.values)
                        plt.xticks(range(len(pg_counts)), x_labels, rotation=45, ha='right')
                        plt.xlabel("Point Group")
                        plt.ylabel("Count")
                        plt.title("Distribution of Cluster Point Groups")
                        
                        # Add count labels above bars
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}',
                                    ha='center', va='bottom')
                    
                    # Second subplot: Space group distribution
                    plt.subplot(212)
                    sg_counts = ranked_df['space_group'].value_counts().head(15)
                    bars = plt.bar(sg_counts.index.astype(str), sg_counts.values)
                    plt.xlabel("Space Group")
                    plt.ylabel("Count")
                    plt.title("Distribution of Compound Space Groups (Top 15)")
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add count labels above bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}',
                                ha='center', va='bottom')
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                # Add ranked compounds summary table
                if not ranked_df.empty:
                    plt.figure(figsize=(8.5, 11))
                    plt.text(0.5, 0.95, "Ranked Compounds Summary", 
                            horizontalalignment='center', fontsize=18)
                    
                    # Create table with key information
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    
                    # Select key columns for display
                    display_columns = [
                        "material_id", "formula", "num_clusters",
                        "predicted_dimentionality", "energy_above_hull"
                    ]
                    
                    if all(col in ranked_df.columns for col in display_columns):
                        display_df = ranked_df[display_columns].head(10).copy()
                        
                        # Format numbers
                        if 'energy_above_hull' in display_df.columns:
                            display_df['energy_above_hull'] = display_df['energy_above_hull'].round(3)
                        
                        # Rename columns for better display
                        column_labels = [
                            "MP ID", "Formula", "# Clusters",
                            "Dim.", "E above hull"
                        ]
                        
                        # Create and format table
                        table = ax.table(
                            cellText=display_df.values,
                            colLabels=column_labels,
                            loc='center',
                            cellLoc='center'
                        )
                        
                        # Format table appearance
                        table.auto_set_font_size(False)
                        table.set_fontsize(9)
                        table.scale(1, 1.8)
                        
                        # Apply custom column widths
                        col_widths = [0.15, 0.2, 0.12, 0.08, 0.15]
                        for i, width in enumerate(col_widths):
                            for j in range(len(display_df) + 1):
                                cell = table[j, i]
                                cell.set_width(width)
                        
                        # Style header row
                        for i in range(len(column_labels)):
                            table[0, i].set_text_props(weight='bold', color='white')
                            table[0, i].set_facecolor('#4472C4')
                        
                        # Alternate row colors
                        for i in range(1, len(display_df) + 1):
                            if i % 2 == 0:
                                for j in range(len(column_labels)):
                                    table[i, j].set_facecolor('#E6F0FF')
                        
                        plt.title("Top 10 Ranked Compounds", pad=20)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                
                # Add individual compound visualizations
                for material_id, data in processed_data.items():
                    try:
                        formula = data["formula"]
                        structure = data["structure"]
                        clusters = data["clusters"]
                        graph = data.get("graph")
                        tm_indices = data.get("tm_indices", [])
                        
                        # Basic compound information
                        plt.figure(figsize=(8.5, 11))
                        plt.text(0.5, 0.95, f"Compound Details: {formula} ({material_id})", 
                                horizontalalignment='center', fontsize=18)
                        
                        # Add property details
                        compound_row = ranked_df[ranked_df["material_id"] == material_id]
                        if not compound_row.empty:
                            plt.text(0.5, 0.85, f"Space Group: {data['space_group']}", 
                                    horizontalalignment='center', fontsize=14)
                            plt.text(0.5, 0.80, f"Dimensionality: {data['dimensionality']}", 
                                    horizontalalignment='center', fontsize=14)
                            
                            energy_above_hull = data.get('energy_above_hull')
                            if energy_above_hull is not None:
                                plt.text(0.5, 0.75, f"Energy Above Hull: {energy_above_hull:.3f} eV/atom", 
                                        horizontalalignment='center', fontsize=14)
                            
                            formation_energy = data.get('formation_energy_per_atom')
                            if formation_energy is not None:
                                plt.text(0.5, 0.70, f"Formation Energy: {formation_energy:.3f} eV/atom", 
                                        horizontalalignment='center', fontsize=14)
                        
                        # Add cluster statistics
                        cluster_stats = {
                            "num_clusters": len(clusters),
                            "avg_size": np.mean([cluster["size"] for cluster in clusters]) if clusters else 0,
                            "avg_distance": np.mean([cluster["average_distance"] for cluster in clusters]) if clusters else 0
                        }
                        
                        plt.text(0.5, 0.55, "Cluster Statistics:", horizontalalignment='center', fontsize=16)
                        stat_text = f"Number of Clusters: {cluster_stats['num_clusters']}\n\n" \
                                  f"Average Distance: {cluster_stats['avg_distance']:.3f} Å\n"
                        plt.text(0.5, 0.35, stat_text, horizontalalignment='center', fontsize=14)
                        plt.axis('off')
                        pdf.savefig()
                        plt.close()
                        
                        # Add connectivity graph visualization
                        if len(tm_indices) > 1:
                            try:
                                # Create a networkx graph if one doesn't exist
                                if not graph:
                                    nx_graph = nx.Graph()
                                    for i, idx1 in enumerate(tm_indices):
                                        for j, idx2 in enumerate(tm_indices):
                                            if i < j:
                                                distance = structure.sites[idx1].distance(structure.sites[idx2])
                                                if distance < 3.5:  # Example cutoff
                                                    nx_graph.add_edge(i, j)
                                    if nx_graph.edges:
                                        graph = nx_graph
                                
                                # Visualize the graph
                                if isinstance(graph, nx.Graph) and len(graph.edges) > 0:
                                    graph_fig = visualize_graph(
                                        graph=graph,
                                        structure=structure,
                                        tm_indices=tm_indices,
                                        material_id=material_id,
                                        formula=formula,
                                        use_3d=False
                                    )
                                    if graph_fig:
                                        pdf.savefig(graph_fig)
                                        plt.close(graph_fig)
                                
                                elif isinstance(graph, dict) and 'edges' in graph:
                                    nx_graph = nx.Graph()
                                    for edge in graph['edges']:
                                        nx_graph.add_edge(edge[0], edge[1])
                                    if nx_graph.edges:
                                        graph_fig = visualize_graph(
                                            graph=nx_graph,
                                            structure=structure,
                                            tm_indices=tm_indices,
                                            material_id=material_id,
                                            formula=formula,
                                            use_3d=False
                                        )
                                        if graph_fig:
                                            pdf.savefig(graph_fig)
                                            plt.close(graph_fig)
                            except Exception as e:
                                logger.warning(f"Error visualizing connectivity graph for {formula}: {e}")
                        
                        # Add cluster visualization
                        try:
                            cluster_fig = visualize_clusters_in_compound(structure, clusters)
                            if cluster_fig:
                                pdf.savefig(cluster_fig)
                                plt.close(cluster_fig)
                        except Exception as e:
                            logger.warning(f"Error visualizing clusters for {formula}: {e}")
                        
                        # Add cluster lattice visualization
                        try:
                            if "lattice_structure" in data:
                                lattice_structure = data["lattice_structure"]
                                space_group = data["space_group"]
                                lattice_fig = visualize_cluster_lattice(lattice_structure)
                                if lattice_fig:
                                    plt.title(f"Cluster Lattice for {formula} (Space Group: {space_group})")
                                    pdf.savefig(lattice_fig)
                                    plt.close(lattice_fig)
                        except Exception as e:
                            logger.warning(f"Error visualizing cluster lattice for {formula}: {e}")
                        
                        # Add dimensionality visualization
                        try:
                            if "supercell" in data and "dimensionality" in data and "singular_values" in data:
                                supercell = data["supercell"]
                                dimensionality = data["dimensionality"]
                                singular_values = data["singular_values"]
                                
                                dim_fig = plt.figure(figsize=(8, 6))
                                ax = dim_fig.add_subplot(111, projection='3d')
                                
                                # Plot only a subset of points for larger structures
                                coords = np.array([site.coords for site in supercell.sites])
                                if len(coords) > 100:
                                    sample_size = min(100, len(coords))
                                    indices = np.random.choice(len(coords), sample_size, replace=False)
                                    coords = coords[indices]
                                
                                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', s=30)
                                
                                # Plot singular value vectors
                                if hasattr(supercell, 'lattice'):
                                    origin = np.mean(coords, axis=0)
                                    scale = np.max(singular_values) * 5
                                    
                                    for i in range(min(2, len(singular_values))):
                                        vector = np.zeros(3)
                                        vector[i] = scale * singular_values[i]
                                        ax.quiver(origin[0], origin[1], origin[2],
                                                vector[0], vector[1], vector[2],
                                                color=['r', 'g'][i], label=f'SV{i+1}')
                                
                                ax.set_xlabel('X (Å)')
                                ax.set_ylabel('Y (Å)')
                                ax.set_zlabel('Z (Å)')
                                ax.set_title(f'Dimensionality: {formula} ({dimensionality})')
                                ax.legend()
                                
                                pdf.savefig(dim_fig)
                                plt.close(dim_fig)
                        except Exception as e:
                            logger.warning(f"Error creating dimensionality visualization for {formula}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing visualizations for {material_id}: {e}")
                        
                        # Make sure all figures are closed
                        plt.close('all')
                
                # Clean up at the end of each compound's visualization to prevent memory issues
                plt.close('all')
    
    # Final completion message with styled output
    complete_panel = Panel.fit(
        f"[bold green]Analysis complete![/bold green]\n"
        f"Results saved to [cyan]{OUTPUT_PDF}[/cyan] and [cyan]{SUMMARY_OUTPUT_CSV}[/cyan]\n"
        f"[italic]Note: Only top 10 compounds were visualized to improve performance.[/italic]",
        title="Cluster Finder Analysis Complete",
        border_style="green"
    )
    console.print(complete_panel)

if __name__ == "__main__":
    main()