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

# Import cluster_finder package
import cluster_finder as cf
from cluster_finder.utils.helpers import (
    search_transition_metal_compounds,
    get_transition_metals
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

# Define constants
API_KEY = "6rcVBNjGRVyfiGPYaLy2xVJNB9X8cN8q"
ELEMENTS = ["Ta", "O"]
OUTPUT_PDF = "ta_o_analysis_results.pdf"
RAW_OUTPUT_CSV = "ta_o_analysis_results_raw.csv"
SUMMARY_OUTPUT_CSV = "ta_o_analysis_results_summary.csv"

def main():
    """Main function to execute the workflow."""
    print("Starting compounds analysis...")
    
    # 1. Search for compounds with Nb and Cl
    print(f"Searching for compounds with elements: {ELEMENTS}")
    compounds = search_transition_metal_compounds(
        elements=ELEMENTS,
        api_key=API_KEY,
        min_elements=2,
        max_elements=4,
        min_magnetization=0.1
    )
    print(f"Found {len(compounds)} compounds containing {ELEMENTS}")
    
    # Get list of transition metals for filtering
    transition_metals = get_transition_metals()
    print(f"Using transition metals: {transition_metals}")
    
    # 2. Process compounds to find clusters
    print("Processing compounds to identify clusters...")
    compounds_with_clusters = get_compounds_with_clusters(compounds, transition_metals)
    
    # Filter to keep only compounds that have clusters
    compounds_with_clusters = [comp for comp in compounds_with_clusters if comp["clusters"]]
    print(f"Found {len(compounds_with_clusters)} compounds with clusters")
 
    # Create a dataframe from compounds using the function from dataframe.py
    print("Creating compounds dataframe...")
    compounds_df = cluster_compounds_dataframe(compounds_with_clusters, compound_system="-".join(ELEMENTS), verbose=True)
    
    # Post-process the ranked dataframe to add additional calculated properties
    print("Post-processing compounds dataframe...")
    processed_df = postprocessed_clusters_dataframe(compounds_df)

    # Rank clusters using the existing rank_clusters function
    print("Ranking clusters...")
    ranked_df = rank_clusters(
        data_source=processed_df,
        api_key=API_KEY,
        custom_props=["symmetry", "energy_above_hull"],
        prop_weights={
            #"formation_energy_per_atom": -1.0,  # Lower formation energy is better
            "symmetry": 0.0,  # Point group has no effect on ranking
            "energy_above_hull": -2.0  # Lower energy above hull is better
        },
        include_default_ranking=True  # Include default criteria like min_avg_distance, point_group_order, etc.
    )

    # Save the full result to CSV
    #print(f"Saving results to {RAW_OUTPUT_CSV}...")
    #export_csv_data(ranked_df, RAW_OUTPUT_CSV)

    # Create a summary dataframe by dropping specific columns
    summary_df = ranked_df.drop(['conventional_cluster_lattice', 'cluster_sites','point_groups_dict','max_point_group_order','highest_point_group','space_group_order'], axis=1, errors='ignore')
    # Save the summary dataframe to CSV
    print(f"Saving summary results to {SUMMARY_OUTPUT_CSV}...")
    export_csv_data(summary_df, SUMMARY_OUTPUT_CSV)
    
    # Create a dictionary to store processed data for visualization - more efficiently
    print("Processing data for visualization...")
    processed_data = {}

    top_compounds = ranked_df.head(10)
    
    for idx, row in top_compounds.iterrows():
        # Access row data correctly (row is a tuple in some pandas versions)
        # Convert row to Series to avoid tuple indexing issue
        row_data = pd.Series(row)
        material_id = row_data['material_id']
        
        # Get the compound data from compounds_df for correct structure and clusters
        compound_data = compounds_df[compounds_df['material_id'] == material_id]
        if compound_data.empty:
            print(f"Warning: Could not find compound data for {material_id}")
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
                print(f"Error processing supercell for {material_id}: {e}")
                
        except Exception as e:
            print(f"Error processing structure data for {material_id}: {e}")
    
    # Create cluster statistics dataframe for visualizations - more efficiently
    print("Creating visualization statistics...")
    cluster_stats_df = pd.DataFrame([
        {
            "material_id": material_id,
            "formula": data["formula"],
            "num_clusters": len(data["clusters"]),
            "cluster_sizes": [cluster["size"] for cluster in data["clusters"]] if data["clusters"] else [],
            "average_distance": np.mean([cluster["average_distance"] for cluster in data["clusters"]]) if data["clusters"] else 0,
            "rank_score": data.get("rank_score", 0)  # Include rank score here
        }
        for material_id, data in processed_data.items()
    ])
    
    # Create a PDF to store all visualizations and results in the requested order
    print("Generating PDF report...")
    with PdfPages(OUTPUT_PDF) as pdf:
        # Create a title page
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.5, f"Analysis of {'-'.join(ELEMENTS)} Compounds\nCluster Finder Results", 
                 horizontalalignment='center', verticalalignment='center', fontsize=24)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 1. FIRST: Summary Cluster Statistics
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
            
            # Plot cluster size distribution - counting each individual cluster size instead of averaging
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
            
            # Add combined point group and space group distribution plots on one page
            if 'point_groups' in ranked_df.columns and 'space_group' in ranked_df.columns:
                plt.figure(figsize=(8.5, 11))
                
                # First subplot: Point group distribution
                plt.subplot(211)
                
                # Extract actual point group symbols (not X0, X1, etc.)
                all_point_groups = []
                
                # Process each compound's point groups data
                for _, row in ranked_df.iterrows():
                    try:
                        # Get point group data
                        pg_data = row['point_groups']
                        
                        # Handle different formats
                        if isinstance(pg_data, str):
                            try:
                                pg_data = ast.literal_eval(pg_data)
                            except (ValueError, SyntaxError) as e:
                                print(f"Error parsing point groups: {e}, value was: {pg_data}")
                                continue
                        
                        # Extract point group symbols from the dictionary where keys are X0, X1, etc.
                        # and values are the actual point group symbols like Ci, D3h, etc.
                        if isinstance(pg_data, dict):
                            for cluster_label, point_group_symbol in pg_data.items():
                                if point_group_symbol and point_group_symbol != 'None':
                                    all_point_groups.append(point_group_symbol)
                                    
                    except Exception as e:
                        print(f"Error processing point groups: {e}")
                
                if all_point_groups:
                    # Count occurrences of each point group
                    pg_counts = pd.Series(all_point_groups).value_counts()
                    
                    # Plot point group distribution
                    if not pg_counts.empty:
                        # Ensure the point group names are properly displayed
                        x_labels = [str(name) for name in pg_counts.index]
                        
                        # Create the bar plot with clear x labels
                        bars = plt.bar(range(len(pg_counts)), pg_counts.values)
                        
                        # Set the x-tick positions and labels
                        plt.xticks(range(len(pg_counts)), x_labels, rotation=45, ha='right')
                        
                        plt.xlabel("Point Group")
                        plt.ylabel("Count")
                        plt.title("Distribution of Cluster Point Groups")
                        
                        # Add count labels above bars
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}',
                                    ha='center', va='bottom', rotation=0)
                
                # Second subplot: Space group distribution
                plt.subplot(212)
                
                # Count occurrences of each space group
                sg_counts = ranked_df['space_group'].value_counts().head(15)  # Show top 15 space groups
                
                # Plot space group distribution
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
                            ha='center', va='bottom', rotation=0)
                
                # Adjust layout to prevent overlap
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            else:
                if 'point_groups' not in ranked_df.columns:
                    print("'point_groups' column not found in the dataframe")
                if 'space_group' not in ranked_df.columns:
                    print("'space_group' column not found in the dataframe")
        
        # 2. SECOND: Ranked Clusters Dataframe - improving table formatting
        if not ranked_df.empty:
            # Create a figure for dimensionality distribution
            plt.figure(figsize=(8.5, 11))
            
            # Plot dimensionality distribution
            dim_counts = ranked_df["predicted_dimentionality"].value_counts()
            plt.subplot(111)
            plt.pie(dim_counts, labels=dim_counts.index, autopct='%1.1f%%')
            plt.title("Dimensionality Distribution of Compounds")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Add ranked dataframe summary visualization - improving table formatting
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.95, "Ranked Compounds Summary", 
                    horizontalalignment='center', fontsize=18)
            
            # Add the ranked dataframe to the PDF - with better formatting
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Select key columns for display
            display_columns = ["material_id", "formula", "magnetization", "num_clusters", "predicted_dimentionality", "energy_above_hull"]
            
            if all(col in ranked_df.columns for col in display_columns):
                # Get top 10 compounds for display
                display_df = ranked_df[display_columns].head(10).copy()
                
                # Format numbers to prevent overlap and make table more readable
                display_df['magnetization'] = display_df['magnetization'].round(2)
                
                if 'energy_above_hull' in display_df.columns:
                    display_df['energy_above_hull'] = display_df['energy_above_hull'].round(3)
                
                # Rename columns for better display
                column_labels = [
                    "MP ID", 
                    "Formula",
                    "Mag.",
                    "# Clusters",
                    "Dim.",
                    "E above hull"
                ]
                
                # Create the table with improved formatting
                table = ax.table(
                    cellText=display_df.values,
                    colLabels=column_labels,
                    loc='center',
                    cellLoc='center'
                )
                
                # Format table appearance
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                
                # Adjust column widths based on content
                table.scale(1, 1.8)  # Scale the entire table
                
                # Apply custom column widths (adjust total to 1.0)
                col_widths = [0.15, 0.2, 0.1, 0.12, 0.08, 0.15]
                for i, width in enumerate(col_widths):
                    for j in range(len(display_df) + 1):  # +1 for header row
                        cell = table[j, i]
                        cell.set_width(width)
                
                # Style header row
                for i in range(len(column_labels)):
                    table[0, i].set_text_props(weight='bold', color='white')
                    table[0, i].set_facecolor('#4472C4')
                
                # Alternate row colors for better readability
                for i in range(1, len(display_df) + 1):
                    if i % 2 == 0:
                        for j in range(len(column_labels)):
                            table[i, j].set_facecolor('#E6F0FF')
                
                plt.title("Top 10 Ranked Compounds", pad=20)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        # 4. FOURTH: Individual Compound Details - more efficiently with fewer compounds
        print("Generating individual compound visualizations...")
        for material_id, data in processed_data.items():
            formula = data["formula"]
            structure = data["structure"]
            clusters = data["clusters"]
            graph = data.get("graph")
            tm_indices = data.get("tm_indices", [])
            
            # Basic compound information
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.95, f"Compound Details: {formula} ({material_id})", 
                    horizontalalignment='center', fontsize=18)
            
            # Add property details if available
            compound_row = ranked_df[ranked_df["material_id"] == material_id]
            if not compound_row.empty:
                plt.text(0.5, 0.85, f"Space Group: {data['space_group']}", 
                        horizontalalignment='center', fontsize=14)
                plt.text(0.5, 0.80, f"Dimensionality: {data['dimensionality']}", 
                        horizontalalignment='center', fontsize=14)
                
                energy_above_hull = data.get('energy_above_hull')
                if energy_above_hull is not None:
                    plt.text(0.5, 0.75, f"Energy Above Hull: {energy_above_hull} eV/atom", 
                            horizontalalignment='center', fontsize=14)
                
                formation_energy = data.get('formation_energy_per_atom')
                if formation_energy is not None:
                    plt.text(0.5, 0.70, f"Formation Energy: {formation_energy} eV/atom", 
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
            try:
                # Create a networkx graph from the transition metal sites
                if len(tm_indices) > 1:
                    # Create a new graph if one doesn't exist
                    if not graph:
                        nx_graph = nx.Graph()
                        
                        # Add edges between transition metal sites that are close enough
                        # This is a simplified approach - in a real scenario you'd use proper
                        # bonding criteria from the cluster_finder package
                        for i, idx1 in enumerate(tm_indices):
                            for j, idx2 in enumerate(tm_indices):
                                if i < j:  # Avoid double counting
                                    distance = structure.sites[idx1].distance(structure.sites[idx2])
                                    # Create an edge if atoms are within 3.5 Å (example cutoff)
                                    if distance < 3.5:
                                        nx_graph.add_edge(i, j)
                        
                        # Only proceed if the graph has edges
                        if nx_graph.edges:
                            graph = nx_graph
                    
                    # Use the visualize_graph function if we have a graph
                    if isinstance(graph, nx.Graph) and len(graph.edges) > 0:
                        graph_fig = visualize_graph(
                            graph=graph,
                            structure=structure,
                            tm_indices=tm_indices,
                            material_id=material_id,
                            formula=formula,
                            use_3d=False  # Use 2D layout for PDF
                        )
                        if graph_fig:
                            pdf.savefig(graph_fig)
                            plt.close(graph_fig)
                    
                    # If graph is in a different format, try to convert it
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
                print(f"Error visualizing connectivity graph for {formula}: {e}")
            
            # Add cluster visualization - less verbose but more efficient
            try:
                cluster_fig = visualize_clusters_in_compound(structure, clusters)
                if cluster_fig:
                    pdf.savefig(cluster_fig)
                    plt.close(cluster_fig)
            except Exception as e:
                print(f"Error visualizing clusters for {formula}: {e}")
            
            # Cluster lattice visualization - less verbose but more efficient
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
                print(f"Error visualizing cluster lattice for {formula}: {e}")
            
            # Dimensionality visualization - more efficient with smaller supercell size
            try:
                if "supercell" in data and "dimensionality" in data and "singular_values" in data:
                    supercell = data["supercell"]
                    dimensionality = data["dimensionality"]
                    singular_values = data["singular_values"]
                    
                    dim_fig = plt.figure(figsize=(8, 6))
                    ax = dim_fig.add_subplot(111, projection='3d')
                    
                    # Plot only a subset of points for larger structures to improve performance
                    coords = np.array([site.coords for site in supercell.sites])
                    if len(coords) > 100:  # If more than 100 points, sample them
                        sample_size = min(100, len(coords))
                        indices = np.random.choice(len(coords), sample_size, replace=False)
                        coords = coords[indices]
                    
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', s=30)
                    
                    # Plot singular value vectors
                    if hasattr(supercell, 'lattice'):
                        origin = np.mean(coords, axis=0)
                        scale = np.max(singular_values) * 5  # Scale for visibility
                        
                        # Plot only the important singular value vectors
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
                print(f"Error creating dimensionality visualization for {formula}: {e}")
    
    print("Analysis complete!")
    print(f"Results saved to {OUTPUT_PDF} and {SUMMARY_OUTPUT_CSV}")
    print("Note: Only top 10 compounds were visualized to improve performance.")

if __name__ == "__main__":
    main()