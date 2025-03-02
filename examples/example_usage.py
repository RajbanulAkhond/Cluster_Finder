#!/usr/bin/env python
"""
Comprehensive example usage of the cluster_finder package.

This script demonstrates the full range of functionality in the
cluster_finder package including:
- Structure loading and manipulation
- Cluster finding and analysis
- Visualization of clusters and structures
- File I/O operations
- Utility functions and API integration
- Command-line interface examples

For real usage, replace "YOUR_API_KEY" with your Materials Project API key.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cluster_finder as cf

# Define API key - replace with your own key
API_KEY = "YOUR_API_KEY"  # Replace with your Materials Project API key

# Create output directory for saving results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def divider(title):
    """Print a section divider with title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def example_load_structure():
    """Example of loading structures."""
    divider("1. LOADING STRUCTURES")
    
    # 1.1 Load from CIF file
    print("1.1 Loading from included sample CIF file...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, '..', 'tests', 'data')
    sample_cif = os.path.join(test_data_dir, 'sample_structure.cif')
    
    if os.path.exists(sample_cif):
        structure = Structure.from_file(sample_cif)
        print(f"Loaded structure: {structure.composition.reduced_formula}")
        print(f"Number of sites: {len(structure)}")
    else:
        # Create a simple cubic structure if sample file not found
        print("Sample CIF not found, creating a simple cubic Fe structure...")
        from pymatgen.core.lattice import Lattice
        
        lattice = Lattice.cubic(3.0)  # 3 Å cube
        structure = Structure(
            lattice=lattice,
            species=['Fe', 'Fe', 'Fe', 'Fe', 'Fe', 'Fe', 'Fe', 'Fe'],
            coords=[
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ],
            coords_are_cartesian=False
        )
    
    # 1.2 Load from Materials Project (if API key provided)
    if API_KEY != "YOUR_API_KEY":
        print("\n1.2 Loading from Materials Project...")
        with MPRester(API_KEY) as mpr:
            # Get a material with Fe-Co
            mp_structure = mpr.get_structure_by_material_id("mp-568345")
            print(f"Loaded MP structure: {mp_structure.composition.reduced_formula}")
    else:
        print("\n1.2 Skipping Materials Project example (no API key provided)")
    
    # 1.3 Create supercell
    print("\n1.3 Creating a supercell...")
    supercell = cf.generate_supercell(structure, (2, 2, 2))
    print(f"Original structure sites: {len(structure)}")
    print(f"Supercell structure sites: {len(supercell)}")
    
    return structure


def example_find_clusters(structure):
    """Example of finding and analyzing clusters."""
    divider("2. FINDING AND ANALYZING CLUSTERS")
    
    # 2.1 Find non-equivalent positions
    print("2.1 Finding non-equivalent transition metal positions...")
    transition_metals = cf.get_transition_metals()
    print(f"Transition metals considered: {', '.join(transition_metals[:5])}...")
    
    unique_sites = cf.find_non_equivalent_positions(structure, transition_metals)
    print(f"Found {len(unique_sites)} unique transition metal sites")
    
    # 2.2 Create connectivity matrix
    print("\n2.2 Creating connectivity matrix...")
    connectivity_matrix, tm_indices = cf.create_connectivity_matrix(
        structure, 
        transition_metals,
        cutoff=3.5  # Adjust based on your needs
    )
    
    print(f"Connectivity matrix shape: {connectivity_matrix.shape}")
    print(f"Number of transition metal sites: {len(tm_indices)}")
    
    # 2.3 Create graph and find clusters
    print("\n2.3 Creating graph and finding clusters...")
    graph = cf.structure_to_graph(connectivity_matrix)
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    clusters = cf.find_clusters(structure, graph, tm_indices, min_cluster_size=2)
    print(f"Found {len(clusters)} clusters with minimum size 2")
    
    # 2.4 Analyze clusters
    print("\n2.4 Analyzing clusters...")
    analyzed_clusters = cf.analyze_clusters(clusters, structure.lattice)
    
    print("Cluster details:")
    for i, cluster in enumerate(analyzed_clusters):
        elements = [site.specie.symbol for site in cluster["sites"]]
        print(f"  Cluster {i+1}: {len(elements)} atoms ({', '.join(elements)}), "
              f"avg. distance: {cluster['average_distance']:.3f} Å")
    
    # 2.5 Classify dimensionality
    print("\n2.5 Classifying structure dimensionality...")
    dimensionality = cf.classify_dimensionality(structure)
    print(f"Structure dimensionality: {dimensionality}")
    
    # 2.6 Identify unique clusters
    print("\n2.6 Identifying unique clusters...")
    unique_clusters = cf.identify_unique_clusters(analyzed_clusters)
    print(f"Number of unique cluster types: {len(unique_clusters)}")
    
    return analyzed_clusters


def example_visualize_clusters(structure, clusters):
    """Example of visualizing clusters."""
    divider("3. VISUALIZING CLUSTERS")
    
    # 3.1 Visualize connectivity graph
    print("3.1 Visualizing connectivity graph...")
    connectivity_matrix, tm_indices = cf.create_connectivity_matrix(
        structure, 
        cf.get_transition_metals()
    )
    graph = cf.structure_to_graph(connectivity_matrix)
    
    fig1 = cf.visualize_graph(
        graph, 
        structure, 
        tm_indices, 
        material_id="example", 
        formula=structure.composition.reduced_formula
    )
    
    plt.figure(fig1.number)
    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, "connectivity_graph.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Saved connectivity graph to {graph_path}")
    plt.close(fig1)
    
    # 3.2 Visualize clusters in compound
    print("\n3.2 Visualizing clusters in compound...")
    if clusters:
        fig2 = cf.visualize_clusters_in_compound(structure, clusters)
        plt.figure(fig2.number)
        plt.tight_layout()
        clusters_path = os.path.join(OUTPUT_DIR, "clusters_visualization.png")
        plt.savefig(clusters_path, dpi=300)
        print(f"Saved clusters visualization to {clusters_path}")
        plt.close(fig2)
    else:
        print("No clusters to visualize")
    
    # 3.3 Visualize structure with rotation
    print("\n3.3 Visualizing structure with custom rotation...")
    # Create a rotation matrix (45° around y-axis)
    rotation_matrix = cf.generate_rotation_matrix(
        axis=np.array([0, 1, 0]),
        angle=np.pi/4
    )
    
    fig3 = cf.visualize_cluster_lattice(structure, rotation_matrix)
    plt.figure(fig3.number)
    plt.tight_layout()
    lattice_path = os.path.join(OUTPUT_DIR, "rotated_structure.png")
    plt.savefig(lattice_path, dpi=300)
    print(f"Saved rotated structure visualization to {lattice_path}")
    plt.close(fig3)


def example_file_operations(structure, clusters):
    """Example of file I/O operations."""
    divider("4. FILE I/O OPERATIONS")
    
    # 4.1 Export structure to CIF
    print("4.1 Exporting structure to CIF...")
    cif_path = os.path.join(OUTPUT_DIR, "exported_structure.cif")
    cf.export_structure_to_cif(structure, cif_path)
    print(f"Exported structure to {cif_path}")
    
    # 4.2 Generate lattice with clusters
    print("\n4.2 Generating lattice with cluster centroids...")
    if clusters:
        cluster_structure = cf.generate_lattice_with_clusters(structure, clusters)
        cluster_cif_path = os.path.join(OUTPUT_DIR, "cluster_centroids.cif")
        cf.export_structure_to_cif(cluster_structure, cluster_cif_path)
        print(f"Exported cluster centroids structure to {cluster_cif_path}")
    else:
        print("No clusters to generate lattice")
    
    # 4.3 Export to CSV and read back
    print("\n4.3 Exporting and importing cluster data with CSV...")
    # Create a sample compound dictionary
    if clusters:
        compound = {
            "material_id": "example",
            "formula": structure.composition.reduced_formula,
            "total_magnetization": 8.0,  # Example value
            "clusters": clusters,
            "structure": structure
        }
        
        # Create DataFrame
        df = cf.cluster_compounds_dataframe([compound], "Example System")
        
        # Export to CSV
        csv_path = os.path.join(OUTPUT_DIR, "cluster_data.csv")
        cf.export_csv_data(df, csv_path)
        print(f"Exported cluster data to {csv_path}")
        
        # Import and post-process
        processed_df = cf.postprocess_clusters(csv_path)
        print("Post-processed DataFrame:")
        print(f"  Columns: {', '.join(processed_df.columns)}")
        print(f"  Rows: {len(processed_df)}")
        
        # Show the first few rows
        if not processed_df.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print("\nDataFrame summary:")
            print(processed_df.head(1).T)  # Transpose for better display
    else:
        print("No clusters to export")


def example_advanced_analysis(structure, clusters):
    """Example of advanced analysis functionality."""
    divider("5. ADVANCED ANALYSIS")
    
    # 5.1 Analyze using spacegroup
    print("5.1 Analyzing structure symmetry...")
    analyzer = SpacegroupAnalyzer(structure)
    space_group = analyzer.get_space_group_symbol()
    point_group = analyzer.get_point_group_symbol()
    
    print(f"Space group: {space_group}")
    print(f"Point group: {point_group}")
    print(f"Point group order: {cf.get_point_group_order(point_group)}")
    
    # 5.2 Rank clusters by properties
    print("\n5.2 Ranking clusters by properties...")
    if clusters and len(clusters) >= 2:
        # Create multiple compounds for ranking
        compounds = []
        # Original compound
        compounds.append({
            "material_id": "example1",
            "formula": structure.composition.reduced_formula,
            "total_magnetization": 8.0,
            "clusters": clusters,
            "structure": structure
        })
        
        # Modified version with different avg distance
        modified_clusters = clusters.copy()
        for cluster in modified_clusters:
            # Slightly modify the average distance
            cluster["average_distance"] = cluster["average_distance"] * 0.9
        
        compounds.append({
            "material_id": "example2",
            "formula": structure.composition.reduced_formula,
            "total_magnetization": 4.0,
            "clusters": modified_clusters,
            "structure": structure
        })
        
        # Create and rank DataFrame
        df = cf.cluster_compounds_dataframe(compounds, "Example System")
        ranked_df = cf.rank_clusters(df)
        
        print("Ranked DataFrame:")
        pd.set_option('display.max_columns', 5)
        print(ranked_df[["material_id", "min_avg_distance", "point_group_order"]])
    else:
        print("Need at least 2 clusters for meaningful ranking")
    
    # 5.3 Simulate CLI usage
    print("\n5.3 Simulating CLI commands:")
    print("  Finding clusters in a structure file:")
    print("    $ cluster-finder find structure.cif --elements Fe Co Ni --radius 3.2")
    print("\n  Analyzing clusters from a previous run:")
    print("    $ cluster-finder analyze structure_clusters.json")
    print("\n  Visualizing clusters:")
    print("    $ cluster-finder visualize structure_clusters.json --show")


def example_materials_project_integration():
    """Example of Materials Project API integration."""
    divider("6. MATERIALS PROJECT INTEGRATION")
    
    if API_KEY == "YOUR_API_KEY":
        print("Skipping Materials Project examples (no API key provided)")
        return
    
    # 6.1 Calculate metal-metal distances in pure elements
    print("6.1 Calculating metal-metal distances in pure elements...")
    metals = ["Fe", "Co", "Ni", "Mn"]
    distances = cf.calculate_metal_distances(metals, API_KEY)
    
    print("Metal-metal distances in pure elements:")
    for metal, distance in distances.items():
        if distance:
            print(f"  {metal}: {distance:.3f} Å")
        else:
            print(f"  {metal}: Not found")
    
    # 6.2 Search for transition metal compounds
    print("\n6.2 Searching for transition metal compounds...")
    compounds = cf.search_transition_metal_compounds(
        ["Fe", "Co"],
        API_KEY,
        min_elements=2,
        max_elements=3,
        min_magnetization=5.0,
        include_fields=["band_gap", "formation_energy_per_atom"]
    )
    
    print(f"Found {len(compounds)} compounds with Fe and Co")
    
    # Show details of a few compounds
    if compounds:
        for i, compound in enumerate(compounds[:3]):
            print(f"\nCompound {i+1}:")
            print(f"  Material ID: {compound.material_id}")
            print(f"  Formula: {compound.formula_pretty}")
            print(f"  Magnetization: {compound.total_magnetization:.2f} μB")
            if hasattr(compound, "band_gap"):
                print(f"  Band Gap: {compound.band_gap:.3f} eV")
            if hasattr(compound, "formation_energy_per_atom"):
                print(f"  Formation Energy: {compound.formation_energy_per_atom:.3f} eV/atom")


def main():
    """Main function demonstrating the package usage."""
    start_time = time.time()
    print("Cluster Finder Comprehensive Example\n" + "="*80)
    
    try:
        # Example 1: Loading structures
        structure = example_load_structure()
        
        # Example 2: Finding clusters
        clusters = example_find_clusters(structure)
        
        # Example 3: Visualizing clusters
        example_visualize_clusters(structure, clusters)
        
        # Example 4: File operations
        example_file_operations(structure, clusters)
        
        # Example 5: Advanced analysis
        example_advanced_analysis(structure, clusters)
        
        # Example 6: Materials Project integration
        example_materials_project_integration()
        
        # Done!
        elapsed_time = time.time() - start_time
        divider("EXAMPLE COMPLETED")
        print(f"All examples completed in {elapsed_time:.2f} seconds")
        print(f"Output files can be found in: {OUTPUT_DIR}")
        
    except Exception as e:
        print("\n" + "!"*80)
        print(f"Error: {e}")
        print("!"*80)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 