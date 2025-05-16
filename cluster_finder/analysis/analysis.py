#!/usr/bin/env python
"""
Parameterized analysis module for running cluster finder analysis on various TM-anion systems.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
import networkx as nx
import ast
import time
import io
import sys
import re
import tempfile
import shutil
import glob
import warnings
import signal
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import atexit
import gc
import multiprocessing

# Import cluster_finder package
from cluster_finder.utils.helpers import (
    search_transition_metal_compounds,
    get_transition_metals,
    get_mp_properties_batch
)
from cluster_finder.utils.logger import get_logger, setup_logging, console
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
from cluster_finder.utils.config_utils import load_config

def setup_analysis_logger(log_dir: Path, system_name: str) -> None:
    """Set up a logger for the current analysis run."""
    # Create the log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the logger for this system
    logger = logging.getLogger(f"cluster_finder.{system_name}")
    
    # Get log level from root logger (respects quiet/verbose mode)
    logger.setLevel(logging.getLogger().getEffectiveLevel())
    
    # Create file handler for this specific analysis
    log_file = log_dir / f"{system_name}_analysis.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always keep detailed logs in file
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

def cleanup_joblib_resources():
    """
    Clean up joblib and multiprocessing resources to prevent leaked resources.
    
    This function performs cleanup of:
    - Temporary folders created by joblib
    - Semaphores and other resources tracked by multiprocessing
    - Forces garbage collection to release memory
    """
    logger = get_logger()
    try:
        # Clean up temporary joblib memmapping folders
        import tempfile
        import shutil
        import glob
        import warnings
        import sys
        import os
        import atexit
        import signal
        import time
        
        # Suppress specific resource tracker warnings
        warnings.filterwarnings("ignore", category=UserWarning, message=".*resource_tracker.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*loky.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*sempahores might leak.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Some resources might leak.*")
        
        # Clean up loky backend first
        try:
            from joblib.externals.loky import get_reusable_executor
            executor = get_reusable_executor(timeout=1)
            executor.shutdown(wait=True, kill_workers=True)
        except Exception as e:
            logger.debug(f"Error shutting down loky executor: {e}")
            
        # Find and clean up any joblib temp folders in the system temp directory
        tmp_patterns = [
            os.path.join(tempfile.gettempdir(), "joblib_memmapping_folder_*"),
            os.path.join(tempfile.gettempdir(), "loky-*"),
            os.path.join("/tmp", "joblib_memmapping_folder_*"),
            os.path.join("/tmp", "loky-*"),
            os.path.join(os.path.expanduser("~"), "joblib_memmapping_folder_*")
        ]
        
        for pattern in tmp_patterns:
            for folder in glob.glob(pattern):
                try:
                    if os.path.exists(folder):
                        logger.debug(f"Cleaning up joblib temporary folder: {folder}")
                        # Try multiple times with exponential backoff
                        for attempt in range(3):
                            try:
                                shutil.rmtree(folder, ignore_errors=True)
                                if not os.path.exists(folder):
                                    break
                            except Exception:
                                time.sleep(0.5 * (2 ** attempt))
                except Exception as e:
                    logger.debug(f"Failed to clean up folder {folder}: {e}")
        
        # Reset joblib's internal state
        try:
            from joblib.parallel import Parallel, parallel_backend
            Parallel()._managed_pool = None
            with parallel_backend('threading', n_jobs=1):  # Force switch to threading backend
                pass
        except Exception as e:
            logger.debug(f"Error resetting joblib state: {e}")

        # Clean up multiprocessing resources
        try:
            import multiprocessing as mp
            if hasattr(mp, '_cleanup'):
                mp._cleanup()
            
            # Handle resource tracker
            if hasattr(mp, 'resource_tracker'):
                tracker_mod = mp.resource_tracker
                if hasattr(tracker_mod, '_resource_tracker') and tracker_mod._resource_tracker is not None:
                    try:
                        # Try to stop the resource tracker cleanly
                        tracker = tracker_mod._resource_tracker
                        if hasattr(tracker, '_stop'):
                            tracker._stop = True
                        # Clear any remaining resources
                        if hasattr(tracker, '_resources'):
                            tracker._resources.clear()
                        # Reset the tracker
                        tracker_mod._resource_tracker = None
                    except Exception as e:
                        logger.debug(f"Error cleaning resource tracker: {e}")
        except Exception as e:
            logger.debug(f"Error cleaning multiprocessing resources: {e}")

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Small sleep to allow processes to terminate
        time.sleep(0.5)
        
    except Exception as cleanup_err:
        logger.debug(f"Error during joblib resource cleanup: {cleanup_err}")

# Register cleanup function to run at exit
atexit.register(cleanup_joblib_resources)

def process_compound_visualization(idx, row_data, logger):
    """Process visualization data for a single compound."""
    try:
        material_id = row_data['material_id']
        formula = row_data['formula']
        
        
        # Convert the structure data to a Structure object
        structure_data = row_data['structure']
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
        
        # Get cluster data
        clusters = []
        cluster_sites_data = row_data['cluster_sites']
        cluster_sizes = row_data['cluster_sizes']
        average_distance = row_data['average_distance']
        
        # Handle string data that needs to be evaluated
        if isinstance(cluster_sites_data, str):
            cluster_sites_data = ast.literal_eval(cluster_sites_data)
        if isinstance(cluster_sizes, str):
            cluster_sizes = ast.literal_eval(cluster_sizes)
        if isinstance(average_distance, str):
            average_distance = ast.literal_eval(average_distance)
        
        # Reconstruct clusters list
        for i, (size, avg_dist, sites) in enumerate(zip(cluster_sizes, average_distance, cluster_sites_data)):
            # Convert site dictionaries to Site objects
            sites_objects = [PeriodicSite.from_dict(site) for site in sites]
            clusters.append({
                'size': size,
                'average_distance': avg_dist,
                'sites': sites_objects,
                'label': f'X{i}'
            })
        
        # Get graph data if available
        graph = None
        if 'graph' in row_data:
            graph = row_data['graph']
        
        # Get transition metal indices for visualization
        transition_metals = get_transition_metals()
        tm_indices = [i for i, site in enumerate(structure) 
                    if site.specie.symbol in transition_metals]
        
        # Generate visualization data
        visualization_data = {
            "structure": structure,
            "clusters": clusters,
            "graph": graph,
            "formula": formula,
            "space_group": row_data['space_group'],
            "point_groups": row_data['point_groups'],
            "dimensionality": row_data['predicted_dimentionality'],
            "energy_above_hull": row_data.get('energy_above_hull'),
            "formation_energy_per_atom": row_data.get('formation_energy_per_atom'),
            "rank_score": row_data.get('rank_score'),
            "tm_indices": tm_indices
        }
        
        # Generate lattice structure
        try:
            lattice_structure, space_group, _ = generate_lattice_with_clusters(structure, clusters)
            visualization_data["lattice_structure"] = lattice_structure
            visualization_data["space_group"] = space_group
            
            # Generate supercell for dimensionality analysis
            supercell = generate_supercell(lattice_structure, supercell_matrix=(10, 10, 10))
            visualization_data["supercell"] = supercell
            
            # Get singular values for dimensionality visualization
            _, singular_values = classify_dimensionality(supercell)
            visualization_data["singular_values"] = singular_values
            
        except Exception as e:
            logger.warning(f"Error processing supercell for {material_id}: {e}")
        
        return material_id, visualization_data
        
    except Exception as e:
        logger.error(f"Error processing structure data for {material_id}: {e}")
        return material_id, None

def run_analysis(
    primary_tm: str,
    anion: str,
    api_key: str,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    save_pdf: bool = True,
    save_csv: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run analysis for a single TM-anion system.
    
    Args:
        primary_tm: Primary transition metal
        anion: Anion element
        api_key: Materials Project API key
        output_dir: Base directory for outputs
        config: Configuration dictionary (optional)
        n_jobs: Number of parallel jobs (default: 1)
        save_pdf: Whether to save PDF report (default: True)
        save_csv: Whether to save CSV data (default: True)
        verbose: Show detailed logging information (default: False)
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    if not config:
        config = {}

    # Create system name and output directory
    system_name = f"{primary_tm}-{anion}"
    system_dir = output_dir / system_name
    system_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger with proper verbosity
    setup_analysis_logger(system_dir, system_name)
    logger = get_logger()
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
    logger.info(f"Starting analysis for {system_name} system")
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Get filter parameters from config
    element_filters = config.get('element_filters', {})
    property_filters = config.get('property_filters', {})
    analysis_params = config.get('analysis_params', {})
    ranking_weights = config.get('ranking_weights', {})
    mp_properties = config.get('mp_properties', [])
    
    # Define output file paths
    pdf_path = system_dir / f"{system_name}_analysis_results.pdf"
    raw_csv_path = system_dir / f"{system_name}_analysis_results_raw.csv"
    summary_csv_path = system_dir / f"{system_name}_analysis_results_summary.csv"
    
    # Define the elements list for this analysis
    elements = [primary_tm, anion]
    
    # Start timing
    start_time = time.time()
    
    # 1. Search for compounds with the specified elements
    logger.info(f"Searching for compounds with elements: {elements}")
    
    try:
        compounds = search_transition_metal_compounds(
            elements=elements,
            api_key=api_key,
            min_elements=element_filters.get('min_elements', 2),
            max_elements=element_filters.get('max_elements', 4),
            min_magnetization=property_filters.get('min_magnetization', 0.01),
            max_magnetization=property_filters.get('max_magnetization', 5.0)
        )
        logger.info(f"Found {len(compounds)} compounds containing {elements}")
    except Exception as e:
        logger.error(f"Error searching for compounds: {e}")
        return {"status": "error", "error": str(e), "system": system_name}
    
    # Get list of transition metals for filtering
    transition_metals = get_transition_metals()
    logger.info(f"Using transition metals: {transition_metals}")
    
    # 2. Process compounds to find clusters
    logger.info("Processing compounds to identify clusters...")
    try:
        compounds_with_clusters = get_compounds_with_clusters(
            compounds,
            transition_metals,
            primary_transition_metal=primary_tm
        )
        
        # Filter to keep only compounds that have clusters
        compounds_with_clusters = [comp for comp in compounds_with_clusters if comp["clusters"]]
        logger.info(f"Found {len(compounds_with_clusters)} compounds with clusters")
    except Exception as e:
        logger.error(f"Error processing compounds for clusters: {e}")
        return {"status": "error", "error": str(e), "system": system_name, "step": "cluster_processing"}
    
    # If no compounds with clusters were found, return early
    if not compounds_with_clusters:
        logger.warning(f"No compounds with clusters found for {system_name}")
        return {"status": "completed", "system": system_name, "compounds_count": 0}
    
    # 3. Create a dataframe from compounds
    logger.info("Creating compounds dataframe...")
    try:
        compounds_df = cluster_compounds_dataframe(
            compounds_with_clusters,
            compound_system=system_name,
            verbose=True
        )
        
        # Store structures before post-processing
        structure_dict = {row['material_id']: row['structure'] 
                        for _, row in compounds_df.iterrows()}
        
        # Post-process the dataframe to add additional calculated properties
        logger.info("Post-processing compounds dataframe...")
        processed_df = postprocessed_clusters_dataframe(compounds_df)
        
        # Restore structures after post-processing
        processed_df['structure'] = processed_df['material_id'].map(structure_dict)
        
    except Exception as e:
        logger.error(f"Error creating compounds dataframe: {e}")
        return {"status": "error", "error": str(e), "system": system_name, "step": "dataframe_creation"}
    
    # 4. Batch retrieve additional properties for all materials at once
    logger.info("Retrieving additional materials properties in batch...")
    try:
        # Extract all unique material IDs
        material_ids = processed_df['material_id'].unique().tolist()
        
        # Use our batch function to get all properties at once
        properties_dict = get_mp_properties_batch(material_ids, mp_properties, api_key)
        
        # Add properties to the dataframe
        for property_name in mp_properties:
            # Create a temporary dictionary for mapping material_id to property value
            property_map = {}
            for material_id, props in properties_dict.items():
                if property_name in props:
                    property_map[material_id] = props[property_name]
            
            # Update the dataframe with the property values
            if property_map:
                processed_df[property_name] = processed_df['material_id'].map(property_map)
                logger.info(f"Added {property_name} for {len(property_map)} materials")
    except Exception as e:
        logger.error(f"Error retrieving materials properties: {e}")
        # Continue anyway, as this is not critical
    
    # 5. Rank clusters using the existing rank_clusters function
    logger.info("Ranking clusters...")
    try:
        # Call rank_clusters directly - it now uses proper logging
        ranked_df = rank_clusters(
            data_source=processed_df,
            api_key=api_key,
            custom_props=["symmetry", "energy_above_hull"],
            prop_weights=ranking_weights,
            include_default_ranking=True
        )
        
        # Create a summary dataframe by dropping specific columns
        summary_df = ranked_df.drop([
            'magnetization', 'conventional_cluster_lattice', 'cluster_sites',
            'point_groups_dict', 'max_point_group_order', 'highest_point_group',
            'space_group_order','structure'
        ], axis=1, errors='ignore')
        
        # Save the summary dataframe to CSV if requested
        if save_csv:
            logger.info(f"Saving summary results to {summary_csv_path}...")
            export_csv_data(summary_df, summary_csv_path)
    except Exception as e:
        logger.error(f"Error ranking clusters: {e}")
        return {"status": "error", "error": str(e), "system": system_name, "step": "cluster_ranking"}
    
    # 6. Process data for visualization using parallel processing
    # The number of compounds to process
    top_n = analysis_params.get('top_n_compounds', 10)
    top_compounds = ranked_df.head(top_n)
    
    # Only process visualization data if we're saving PDF reports
    processed_data = {}
    if save_pdf:
        logger.info(f"Processing data for visualization (top {top_n} compounds)...")
        
        # Import joblib for parallel processing
        try:
            from joblib import Parallel, delayed
            
            # Process compounds in parallel
            logger.info(f"Starting parallel processing with {n_jobs} workers")
            # Convert the dataframe to a list of tuples for parallel processing
            top_compounds_list = list(top_compounds.iterrows())
            
            # Run the parallel processing with shared logger
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_compound_visualization)(idx, row, logger) 
                for idx, row in top_compounds_list
            )
            
            # Convert results to a dictionary
            for material_id, data in results:
                if data is not None:
                    processed_data[material_id] = data
            
        except ImportError:
            logger.warning("joblib not available, falling back to sequential processing")
            # Fall back to sequential processing if joblib is not available
            for idx, row in top_compounds.iterrows():
                material_id, data = process_compound_visualization(idx, row, logger)
                if data is not None:
                    processed_data[material_id] = data
    else:
        logger.info("Skipping visualization data processing (--no-pdf option was used)")
    
    # 7. Generate visualization and save to PDF if requested
    if save_pdf and processed_data:
        logger.info(f"Generating PDF report...")
        
        try:
            # Create cluster statistics dataframe for visualizations
            cluster_stats_df = pd.DataFrame([
                {
                    "material_id": material_id,
                    "formula": data["formula"],
                    "num_clusters": len(data["clusters"]),
                    "cluster_sizes": [cluster["size"] for cluster in data["clusters"]] if data["clusters"] else [],
                    "average_distance": float(np.mean([cluster["average_distance"] for cluster in data["clusters"]])) if data["clusters"] else 0,
                    "rank_score": data.get("rank_score", 0)
                }
                for material_id, data in processed_data.items()
            ])
            
            with PdfPages(pdf_path) as pdf:
                # Create a title page
                plt.figure(figsize=(8.5, 11))
                plt.text(0.5, 0.5, f"Analysis of {system_name} Compounds\nCluster Finder Results", 
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
                        
            logger.info(f"PDF report saved to {pdf_path}")

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
    
    # Calculate and log the total time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Clean up any joblib resources to prevent memory leaks
    cleanup_joblib_resources()
    
    # Return summary data
    return {
        "status": "completed",
        "system": system_name,
        "compounds_count": len(compounds),
        "compounds_with_clusters_count": len(compounds_with_clusters),
        "time_taken": elapsed_time,
        "output_dir": str(system_dir),
        "outputs": {
            "pdf": str(pdf_path) if save_pdf else None,
            "summary_csv": str(summary_csv_path) if save_csv else None
        }
    }