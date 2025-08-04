#!/usr/bin/env python3
"""
Normalized Periodic Table Heatmap for Cluster Analysis

This script creates periodic table heatmaps showing cluster formation efficiency 
(clusters per compound analyzed) by normalizing data from merged_results_all_prop.csv 
using compound counts from summary.csv.

Uses the periodic_trends plotting functionality for professional visualizations.

Author: Cluster Finder Analysis Tool
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict, Counter
import os
from pathlib import Path
import csv
from typing import List, Dict, Tuple, Any, Set
import warnings
import ast  # Add this import for safe evaluation of string literals
import matplotlib.patheffects
from scipy import stats
import json
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import re

# Import periodic_trends functionality
from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    LogColorMapper,
    ColorBar,
    BasicTicker,
)
from bokeh.plotting import figure, output_file
from bokeh.io import show as show_
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge
from matplotlib.colors import Normalize, LogNorm, to_hex
from matplotlib.cm import plasma, inferno, magma, viridis, cividis, turbo, ScalarMappable

# Periodic table positions for transition metals (group, period)
TRANSITION_METALS = {
    'Sc': (3, 4), 'Ti': (4, 4), 'V': (5, 4), 'Cr': (6, 4), 'Mn': (7, 4), 
    'Fe': (8, 4), 'Co': (9, 4), 'Ni': (10, 4), 'Cu': (11, 4), 'Zn': (12, 4),
    'Y': (3, 5), 'Zr': (4, 5), 'Nb': (5, 5), 'Mo': (6, 5), 'Tc': (7, 5), 
    'Ru': (8, 5), 'Rh': (9, 5), 'Pd': (10, 5), 'Ag': (11, 5), 'Cd': (12, 5),
    'Hf': (4, 6), 'Ta': (5, 6), 'W': (6, 6), 'Re': (7, 6), 'Os': (8, 6), 
    'Ir': (9, 6), 'Pt': (10, 6), 'Au': (11, 6), 'Hg': (12, 6),
    'Rf': (4, 7), 'Db': (5, 7), 'Sg': (6, 7), 'Bh': (7, 7), 'Hs': (8, 7), 
    'Mt': (9, 7), 'Ds': (10, 7), 'Rg': (11, 7), 'Cn': (12, 7)
}

def parse_compound_system(compound_system):
    """Parse compound system string to extract metal and anion."""
    if '-' in compound_system:
        parts = compound_system.split('-')
        if len(parts) == 2:
            metal, anion = parts[0].strip(), parts[1].strip()
            if metal in TRANSITION_METALS:
                return metal, anion
    return None, None

def load_summary_data(summary_csv_path):
    """
    Load compound counts from summary.csv for normalization.
    
    Returns:
        dict: Nested dictionary with compound counts by metal and anion
    """
    try:
        df = pd.read_csv(summary_csv_path)
        print(f"Loaded summary data from {summary_csv_path}")
        
        compound_counts = defaultdict(lambda: defaultdict(int))
        metal_compound_totals = defaultdict(int)
        anion_compound_totals = defaultdict(int)
        
        for _, row in df.iterrows():
            system = row['System']
            metal = row['TM']
            anion = row['Anion']
            compounds = row['Compounds']
            
            if metal in TRANSITION_METALS and compounds > 0:
                compound_counts[metal][anion] = compounds
                metal_compound_totals[metal] += compounds
                anion_compound_totals[anion] += compounds
        
        print(f"Found compound data for {len(metal_compound_totals)} metals and {len(anion_compound_totals)} anions")
        return compound_counts, metal_compound_totals, anion_compound_totals
        
    except Exception as e:
        print(f"Error loading summary data: {e}")
        return {}, {}, {}

def parse_cluster_sizes(cluster_sizes_str):
    """
    Parse cluster_sizes string to extract numeric values.
    Handles various formats like "[3, 4, 5]", "3,4,5", or single numbers.
    Returns maximum value from the list.
    """
    if pd.isna(cluster_sizes_str):
        return None
    
    # Convert to string if not already
    sizes_str = str(cluster_sizes_str).strip()
    
    # Handle empty or invalid strings
    if not sizes_str or sizes_str.lower() in ['nan', 'none', '']:
        return None
    
    try:
        # Try to evaluate as a literal (for lists like "[3, 4, 5]")
        if sizes_str.startswith('[') and sizes_str.endswith(']'):
            sizes_list = ast.literal_eval(sizes_str)
            return max(sizes_list) if sizes_list else None
        
        # Try comma-separated values
        elif ',' in sizes_str:
            sizes_list = [float(x.strip()) for x in sizes_str.split(',') if x.strip()]
            return max(sizes_list) if sizes_list else None
        
        # Single number
        else:
            return float(sizes_str)
            
    except (ValueError, SyntaxError):
        # If all else fails, try to extract numbers from the string
        import re
        numbers = re.findall(r'-?\d+\.?\d*', sizes_str)
        if numbers:
            return max(float(x) for x in numbers)
        else:
            return None

def load_and_process_cluster_data(csv_path):
    """Load cluster data and process it to count instances and extract metrics by metal-anion pairs."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} cluster records from {csv_path}")
        
        # Data structures for counting instances
        cluster_data = defaultdict(lambda: defaultdict(int))
        metal_cluster_totals = defaultdict(int)
        anion_cluster_totals = defaultdict(int)
        
        # Data structures for additional metrics
        num_clusters_data = defaultdict(lambda: defaultdict(list))
        cluster_sizes_data = defaultdict(lambda: defaultdict(list))
        min_avg_distance_data = defaultdict(lambda: defaultdict(list))
        
        # Data structure for point groups
        point_groups_data = defaultdict(lambda: defaultdict(list))
        materials_by_point_group = defaultdict(lambda: defaultdict(list))
        
        for _, row in df.iterrows():
            metal, anion = parse_compound_system(row['compound_system'])
            if metal and anion:
                # Count each instance (row) where a cluster was found
                cluster_data[metal][anion] += 1
                metal_cluster_totals[metal] += 1
                anion_cluster_totals[anion] += 1
                
                # Collect additional metrics
                num_clusters_data[metal][anion].append(row['num_clusters'])
                
                # Parse cluster sizes and get maximum value
                max_cluster_size = parse_cluster_sizes(row['cluster_sizes'])
                if max_cluster_size is not None:
                    cluster_sizes_data[metal][anion].append(max_cluster_size)
                
                min_avg_distance_data[metal][anion].append(row['min_avg_distance'])
                
                # Process point groups if available
                if 'point_groups' in row and not pd.isna(row['point_groups']):
                    try:
                        # Convert string representation to dictionary
                        point_groups_dict = ast.literal_eval(str(row['point_groups']))
                        if isinstance(point_groups_dict, dict) and point_groups_dict:
                            # Store all point groups for this material
                            point_groups_data[metal][anion].append(point_groups_dict)
                            
                            # Store material info by point group for lookup
                            material_id = row.get('material_id', f"unknown_{_}")
                            cluster_size = max_cluster_size
                            
                            for cluster_id, point_group in point_groups_dict.items():
                                materials_by_point_group[point_group][cluster_size].append({
                                    'material_id': material_id,
                                    'metal': metal,
                                    'anion': anion,
                                    'cluster_id': cluster_id,
                                    'compound_system': row['compound_system'],
                                    'num_clusters': row['num_clusters']
                                })
                    except (ValueError, SyntaxError) as e:
                        # Skip invalid point group data
                        print(f"Warning: Could not parse point groups for row {_}: {e}")
        
        return (cluster_data, metal_cluster_totals, anion_cluster_totals, 
                num_clusters_data, cluster_sizes_data, min_avg_distance_data,
                point_groups_data, materials_by_point_group)
        
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return {}, {}, {}, {}, {}, {}, {}, {}

def calculate_normalized_data(cluster_data, compound_counts):
    """
    Calculate cluster formation efficiency (clusters per compound analyzed).
    
    Returns:
        dict: Normalized data showing clusters per compound
    """
    normalized_data = defaultdict(lambda: defaultdict(float))
    metal_efficiency = defaultdict(float)
    anion_efficiency = defaultdict(float)
    
    for metal in cluster_data:
        metal_total_clusters = 0
        metal_total_compounds = 0
        
        for anion in cluster_data[metal]:
            clusters = cluster_data[metal][anion]
            compounds = compound_counts[metal][anion]
            
            if compounds > 0:
                efficiency = clusters / compounds
                normalized_data[metal][anion] = efficiency
                metal_total_clusters += clusters
                metal_total_compounds += compounds
        
        if metal_total_compounds > 0:
            metal_efficiency[metal] = metal_total_clusters / metal_total_compounds
    
    # Calculate anion efficiency
    for anion in set().union(*(cluster_data[metal].keys() for metal in cluster_data)):
        anion_total_clusters = 0
        anion_total_compounds = 0
        
        for metal in cluster_data:
            if anion in cluster_data[metal] and anion in compound_counts[metal]:
                anion_total_clusters += cluster_data[metal][anion]
                anion_total_compounds += compound_counts[metal][anion]
        
        if anion_total_compounds > 0:
            anion_efficiency[anion] = anion_total_clusters / anion_total_compounds
    
    return normalized_data, metal_efficiency, anion_efficiency

def create_csv_for_periodic_trends(data_dict, output_path, data_type="efficiency"):
    """Create CSV file in format expected by periodic_trends plotter."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for element, value in data_dict.items():
            if element in TRANSITION_METALS and value > 0:
                writer.writerow([element, f"{value:.4f}"])
    print(f"Created {data_type} CSV for periodic_trends: {output_path}")

def plot_periodic_table_heatmap(
    csv_filename: str,
    output_filename: str,
    title: str,
    cmap: str = "plasma",
    log_scale: bool = False,
    width: int = 1050,
):
    """
    Create periodic table heatmap using the periodic_trends functionality.
    """
    from pandas import options
    options.mode.chained_assignment = None

    # Assign color palette
    cmap_dict = {
        "plasma": (plasma, "Plasma256"),
        "inferno": (inferno, "Inferno256"), 
        "magma": (magma, "Magma256"),
        "viridis": (viridis, "Viridis256"),
        "cividis": (cividis, "Cividis256"),
        "turbo": (turbo, "Turbo256")
    }
    
    if cmap not in cmap_dict:
        cmap = "plasma"
    
    mpl_cmap, bokeh_palette = cmap_dict[cmap]

    # Read data
    data_elements = []
    data_list = []
    with open(csv_filename, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data_elements.append(row[0])
            data_list.append(float(row[1]))

    if len(data_list) == 0:
        print(f"No data found in {csv_filename}")
        return None

    # Setup periods and groups
    period_label = ["1", "2", "3", "4", "5", "6", "7", "blank"]
    group_range = [str(x) for x in range(1, 19)]

    # Setup color mapping
    if log_scale and all(d > 0 for d in data_list):
        color_mapper = LogColorMapper(palette=bokeh_palette, low=min(data_list), high=max(data_list))
        norm = LogNorm(vmin=min(data_list), vmax=max(data_list))
    else:
        color_mapper = LinearColorMapper(palette=bokeh_palette, low=min(data_list), high=max(data_list))
        norm = Normalize(vmin=min(data_list), vmax=max(data_list))
    
    color_scale = ScalarMappable(norm=norm, cmap=mpl_cmap).to_rgba(data_list, alpha=None)
    
    # Set colors
    blank_color = "#c4c4c4"
    color_list = [blank_color] * len(elements)
    
    for i, data_element in enumerate(data_elements):
        element_entry = elements.symbol[elements.symbol.str.lower() == data_element.lower()]
        if not element_entry.empty:
            element_index = element_entry.index[0]
            color_list[element_index] = to_hex(color_scale[i])

    # Create plot
    source = ColumnDataSource(
        data=dict(
            group=[str(x) for x in elements["group"]],
            period=[str(y) for y in elements["period"]],
            sym=elements["symbol"],
            atomic_number=elements["atomic number"],
            type_color=color_list,
        )
    )

    p = figure(x_range=group_range, y_range=list(reversed(period_label)), tools="save", title=title)
    p.width = width
    p.outline_line_color = None
    p.background_fill_color = None
    p.border_fill_color = None
    p.toolbar_location = "above"
    p.rect("group", "period", 0.9, 0.9, source=source, alpha=0.65, color="type_color")
    p.axis.visible = False
    
    # Add text
    text_props = {"source": source, "angle": 0, "color": "black", "text_align": "left", "text_baseline": "middle"}
    x = dodge("group", -0.4, range=p.x_range)
    y = dodge("period", 0.3, range=p.y_range)
    p.text(x=x, y="period", text="sym", text_font_style="bold", text_font_size="16pt", **text_props)
    p.text(x=x, y=y, text="atomic_number", text_font_size="11pt", **text_props)

    # Add colorbar
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=10),
        border_line_color=None,
        label_standoff=12,
        location=(0, 0),
        orientation="vertical",
        scale_alpha=0.65,
        major_label_text_font_size="14pt",
    )
    p.add_layout(color_bar, "right")
    p.grid.grid_line_color = None

    # Save plot
    output_file(output_filename)
    show_(p)
    
    return p

def create_matplotlib_heatmaps(normalized_data, metal_efficiency, anion_efficiency, output_dir):
    """Create additional matplotlib-based visualizations."""
    
    # Combined interaction efficiency matrix
    top_metals = sorted(metal_efficiency.items(), key=lambda x: x[1], reverse=True)[:15]
    top_anions = sorted(anion_efficiency.items(), key=lambda x: x[1], reverse=True)[:8]
    
    metal_names = [metal for metal, _ in top_metals]
    anion_names = [anion for anion, _ in top_anions]
    
    interaction_matrix = np.zeros((len(metal_names), len(anion_names)))
    
    for i, metal in enumerate(metal_names):
        for j, anion in enumerate(anion_names):
            if metal in normalized_data and anion in normalized_data[metal]:
                interaction_matrix[i, j] = normalized_data[metal][anion]
    
    # Use the centralized interaction matrix function
    create_interaction_matrix(
        interaction_matrix, 
        metal_names, 
        anion_names, 
        'Cluster Formation Efficiency (Clusters per Compound)\nTop Metal-Anion Combinations',
        'Clusters per Compound',
        f"{output_dir}/efficiency_interaction_matrix",
        cmap="plasma",
        annotation_format="{:.2f}"
    )

def generate_comprehensive_report(cluster_data, compound_counts, normalized_data, 
                                metal_efficiency, anion_efficiency, output_dir):
    """Generate comprehensive analysis report."""
    
    report_path = f"{output_dir}/normalized_cluster_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Normalized Cluster Formation Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("This analysis shows cluster formation efficiency by normalizing\n")
        f.write("cluster counts using the number of compounds analyzed per system.\n")
        f.write("Efficiency = Total Clusters / Total Compounds Analyzed\n\n")
        
        # Overall statistics
        total_clusters = sum(sum(cluster_data[metal].values()) for metal in cluster_data)
        total_compounds = sum(sum(compound_counts[metal].values()) for metal in compound_counts if metal in cluster_data)
        overall_efficiency = total_clusters / total_compounds if total_compounds > 0 else 0
        
        f.write(f"OVERALL STATISTICS:\n")
        f.write(f"Total clusters found: {total_clusters:,}\n")
        f.write(f"Total compounds analyzed: {total_compounds:,}\n")
        f.write(f"Overall efficiency: {overall_efficiency:.3f} clusters/compound\n\n")
        
        # ALL metals by efficiency
        f.write("ALL METALS BY CLUSTER FORMATION EFFICIENCY:\n")
        f.write("=" * 60 + "\n")
        f.write("Rank  Metal  Efficiency  Total_Clusters  Total_Compounds\n")
        f.write("-" * 60 + "\n")
        for i, (metal, eff) in enumerate(sorted(metal_efficiency.items(), key=lambda x: x[1], reverse=True), 1):
            total_clusters_metal = sum(cluster_data[metal].values())
            total_compounds_metal = sum(compound_counts[metal].values()) if metal in compound_counts else 0
            f.write(f"{i:2d}.   {metal:2s}     {eff:6.3f}      {total_clusters_metal:6d}        {total_compounds_metal:6d}\n")
        
        # ALL anions by efficiency
        f.write("\nALL ANIONS BY CLUSTER FORMATION EFFICIENCY:\n")
        f.write("=" * 60 + "\n")
        f.write("Rank  Anion  Efficiency  Total_Clusters  Total_Compounds\n")
        f.write("-" * 60 + "\n")
        for i, (anion, eff) in enumerate(sorted(anion_efficiency.items(), key=lambda x: x[1], reverse=True), 1):
            # Calculate totals for this anion
            anion_clusters = sum(cluster_data[metal].get(anion, 0) for metal in cluster_data)
            anion_compounds = sum(compound_counts[metal].get(anion, 0) for metal in compound_counts)
            f.write(f"{i:2d}.   {anion:2s}      {eff:6.3f}      {anion_clusters:6d}        {anion_compounds:6d}\n")
        
        # Metal efficiency statistics
        f.write("\nMETAL EFFICIENCY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        metal_effs = list(metal_efficiency.values())
        f.write(f"Number of metals analyzed: {len(metal_effs)}\n")
        f.write(f"Mean efficiency: {np.mean(metal_effs):.3f}\n")
        f.write(f"Median efficiency: {np.median(metal_effs):.3f}\n")
        f.write(f"Standard deviation: {np.std(metal_effs):.3f}\n")
        f.write(f"Minimum efficiency: {np.min(metal_effs):.3f} ({min(metal_efficiency, key=metal_efficiency.get)})\n")
        f.write(f"Maximum efficiency: {np.max(metal_effs):.3f} ({max(metal_efficiency, key=metal_efficiency.get)})\n")
        
        # Anion efficiency statistics
        f.write("\nANION EFFICIENCY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        anion_effs = list(anion_efficiency.values())
        f.write(f"Number of anions analyzed: {len(anion_effs)}\n")
        f.write(f"Mean efficiency: {np.mean(anion_effs):.3f}\n")
        f.write(f"Median efficiency: {np.median(anion_effs):.3f}\n")
        f.write(f"Standard deviation: {np.std(anion_effs):.3f}\n")
        f.write(f"Minimum efficiency: {np.min(anion_effs):.3f} ({min(anion_efficiency, key=anion_efficiency.get)})\n")
        f.write(f"Maximum efficiency: {np.max(anion_effs):.3f} ({max(anion_efficiency, key=anion_efficiency.get)})\n")
        
        # Top metal-anion combinations
        f.write("\nTOP 25 METAL-ANION COMBINATIONS BY EFFICIENCY:\n")
        f.write("=" * 65 + "\n")
        f.write("Rank  System  Efficiency  Clusters  Compounds\n")
        f.write("-" * 65 + "\n")
        combinations = []
        for metal in normalized_data:
            for anion, eff in normalized_data[metal].items():
                clusters = cluster_data[metal].get(anion, 0)
                compounds = compound_counts[metal].get(anion, 0)
                combinations.append((f"{metal}-{anion}", eff, clusters, compounds))
        
        for i, (combo, eff, clusters, compounds) in enumerate(sorted(combinations, key=lambda x: x[1], reverse=True)[:25], 1):
            f.write(f"{i:2d}.   {combo:6s}    {eff:6.3f}     {clusters:5d}     {compounds:5d}\n")
        
        # Metal-anion combination statistics
        f.write("\nMETAL-ANION COMBINATION STATISTICS:\n")
        f.write("-" * 45 + "\n")
        combo_effs = [eff for _, eff, _, _ in combinations]
        f.write(f"Total combinations analyzed: {len(combinations)}\n")
        f.write(f"Mean efficiency: {np.mean(combo_effs):.3f}\n")
        f.write(f"Median efficiency: {np.median(combo_effs):.3f}\n")
        f.write(f"Standard deviation: {np.std(combo_effs):.3f}\n")
        f.write(f"Minimum efficiency: {np.min(combo_effs):.3f}\n")
        f.write(f"Maximum efficiency: {np.max(combo_effs):.3f}\n")
        
        # Efficiency distribution analysis
        f.write("\nEFFICIENCY DISTRIBUTION ANALYSIS:\n")
        f.write("-" * 45 + "\n")
        
        # Metal efficiency quartiles
        metal_q1, metal_q2, metal_q3 = np.percentile(metal_effs, [25, 50, 75])
        f.write("Metal Efficiency Quartiles:\n")
        f.write(f"  Q1 (25th percentile): {metal_q1:.3f}\n")
        f.write(f"  Q2 (50th percentile): {metal_q2:.3f}\n")
        f.write(f"  Q3 (75th percentile): {metal_q3:.3f}\n")
        
        # Anion efficiency quartiles
        anion_q1, anion_q2, anion_q3 = np.percentile(anion_effs, [25, 50, 75])
        f.write("\nAnion Efficiency Quartiles:\n")
        f.write(f"  Q1 (25th percentile): {anion_q1:.3f}\n")
        f.write(f"  Q2 (50th percentile): {anion_q2:.3f}\n")
        f.write(f"  Q3 (75th percentile): {anion_q3:.3f}\n")
        
        # High efficiency analysis
        f.write("\nHIGH EFFICIENCY ANALYSIS (>0.1 clusters/compound):\n")
        f.write("-" * 55 + "\n")
        high_eff_metals = [metal for metal, eff in metal_efficiency.items() if eff > 0.1]
        high_eff_anions = [anion for anion, eff in anion_efficiency.items() if eff > 0.1]
        high_eff_combos = [combo for combo, eff, _, _ in combinations if eff > 0.1]
        
        f.write(f"High-efficiency metals: {len(high_eff_metals)} ({len(high_eff_metals)/len(metal_effs)*100:.1f}%)\n")
        f.write(f"  {', '.join(high_eff_metals)}\n")
        f.write(f"High-efficiency anions: {len(high_eff_anions)} ({len(high_eff_anions)/len(anion_effs)*100:.1f}%)\n")
        f.write(f"  {', '.join(high_eff_anions)}\n")
        f.write(f"High-efficiency combinations: {len(high_eff_combos)} ({len(high_eff_combos)/len(combinations)*100:.1f}%)\n")
        
        f.write(f"\nDetailed data saved to CSV files for periodic table visualization.\n")
    
    print(f"Comprehensive report saved to {report_path}")

def calculate_metric_averages(metric_data, compound_counts, metric_name):
    """
    Calculate average metric values normalized by compound counts.
    
    Args:
        metric_data: Nested dict with lists of metric values by metal-anion
        compound_counts: Compound counts for normalization
        metric_name: Name of the metric for reporting
    
    Returns:
        Tuple of (normalized_metric_data, metal_averages, anion_averages)
    """
    normalized_metric_data = defaultdict(lambda: defaultdict(float))
    metal_averages = defaultdict(float)
    anion_averages = defaultdict(float)
    
    # Calculate per-combination averages
    for metal in metric_data:
        metal_total_metric = 0
        metal_total_compounds = 0
        
        for anion in metric_data[metal]:
            if metric_data[metal][anion]:  # Check if list is not empty
                avg_metric = np.mean(metric_data[metal][anion])
                compounds = compound_counts[metal][anion]
                
                if compounds > 0:
                    # Weight by number of compounds analyzed
                    normalized_metric_data[metal][anion] = avg_metric
                    metal_total_metric += avg_metric * compounds
                    metal_total_compounds += compounds
        
        if metal_total_compounds > 0:
            metal_averages[metal] = metal_total_metric / metal_total_compounds
    
    # Calculate anion averages
    for anion in set().union(*(metric_data[metal].keys() for metal in metric_data)):
        anion_total_metric = 0
        anion_total_compounds = 0
        
        for metal in metric_data:
            if anion in metric_data[metal] and metric_data[metal][anion]:
                avg_metric = np.mean(metric_data[metal][anion])
                compounds = compound_counts[metal].get(anion, 0)
                
                if compounds > 0:
                    anion_total_metric += avg_metric * compounds
                    anion_total_compounds += compounds
        
        if anion_total_compounds > 0:
            anion_averages[anion] = anion_total_metric / anion_total_compounds
    
    return normalized_metric_data, metal_averages, anion_averages

def create_metric_heatmaps_and_matrices(metric_data, compound_counts, metric_name, output_dir, 
                                      cmap="viridis", log_scale=False):
    """
    Create periodic table heatmaps and interaction matrices for a specific metric.
    
    Args:
        metric_data: Raw metric data by metal-anion
        compound_counts: Compound counts for normalization
        metric_name: Name of the metric (for file naming and titles)
        output_dir: Output directory path
        cmap: Colormap to use
        log_scale: Whether to use log scale
    """
    print(f"\nProcessing {metric_name} data...")
    
    # Calculate averages
    normalized_data, metal_averages, anion_averages = calculate_metric_averages(
        metric_data, compound_counts, metric_name)
    
    if not metal_averages:
        print(f"No data available for {metric_name}")
        return
    
    # Create CSV for periodic table
    metric_csv = output_dir / f"metal_{metric_name.lower().replace(' ', '_')}.csv"
    create_csv_for_periodic_trends(metal_averages, metric_csv, f"metal {metric_name}")
    
    # Create periodic table heatmap
    try:
        plot_periodic_table_heatmap(
            str(metric_csv),
            str(output_dir / f"metal_{metric_name.lower().replace(' ', '_')}_periodic_table.html"),
            f"Transition Metal Average {metric_name}",
            cmap=cmap,
            log_scale=log_scale
        )
        print(f"Created {metric_name} periodic table heatmap")
    except Exception as e:
        print(f"Error creating {metric_name} periodic table: {e}")
    
    # Create interaction matrix
    create_metric_interaction_matrix(normalized_data, metal_averages, anion_averages, 
                                   metric_name, output_dir, cmap)

def create_metric_interaction_matrix(normalized_data, metal_averages, anion_averages, 
                                   metric_name, output_dir, cmap="viridis"):
    """Create matplotlib interaction matrix for a specific metric."""
    
    # Get top metals and anions by average values
    top_metals = sorted(metal_averages.items(), key=lambda x: x[1], reverse=True)[:15]
    top_anions = sorted(anion_averages.items(), key=lambda x: x[1], reverse=True)[:8]
    
    metal_names = [metal for metal, _ in top_metals]
    anion_names = [anion for anion, _ in top_anions]
    
    interaction_matrix = np.zeros((len(metal_names), len(anion_names)))
    
    for i, metal in enumerate(metal_names):
        for j, anion in enumerate(anion_names):
            if metal in normalized_data and anion in normalized_data[metal]:
                interaction_matrix[i, j] = normalized_data[metal][anion]
    
    # Use the centralized interaction matrix function
    create_interaction_matrix(
        interaction_matrix, 
        metal_names, 
        anion_names, 
        f"{metric_name} Analysis\nTop Metal-Anion Combinations", 
        metric_name, 
        f"{output_dir}/{metric_name.lower().replace(' ', '_')}_interaction_matrix",
        cmap=cmap,
        annotation_format="{:.2f}" if "Distance" not in metric_name else "{:.3f}"
    )

def create_interaction_matrix(
    data_matrix, row_labels, col_labels, title, colorbar_label, 
    output_path, cmap="viridis", annotation_format="{:.2f}", figsize=(16, 12)
):
    """
    Centralized function to create interaction matrix plots with consistent styling.
    
    Args:
        data_matrix: 2D numpy array with values for the heatmap
        row_labels: List of labels for rows (y-axis)
        col_labels: List of labels for columns (x-axis)
        title: Title of the plot
        colorbar_label: Label for the colorbar
        output_path: Path to save the output files (without extension)
        cmap: Colormap to use (string name of matplotlib colormap)
        annotation_format: Format string for cell annotations
        figsize: Size of the figure (width, height) in inches
    """
    # Handle empty data
    if np.max(data_matrix) == 0:
        print(f"No data available for {title}")
        return
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colormap
    if cmap == "viridis":
        plt_cmap = plt.cm.viridis
    elif cmap == "plasma":
        plt_cmap = plt.cm.plasma
    elif cmap == "magma":
        plt_cmap = plt.cm.magma
    else:
        plt_cmap = plt.cm.viridis
    
    # Set up normalization - only consider non-zero values for vmin
    vmin = np.min(data_matrix[data_matrix > 0]) if np.any(data_matrix > 0) else 0
    vmax = np.max(data_matrix)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Turn off default grid
    ax.grid(False)
    
    # Use pcolormesh for precise grid alignment
    x = np.arange(len(col_labels) + 1)
    y = np.arange(len(row_labels) + 1)
    mesh = ax.pcolormesh(x, y, data_matrix, cmap=plt_cmap, norm=norm)
    
    # Add explicit grid lines that align perfectly with cell boundaries
    for i in range(len(row_labels) + 1):
        ax.axhline(y=i, color='white', linewidth=1.5, alpha=0.8)
    for j in range(len(col_labels) + 1):
        ax.axvline(x=j, color='white', linewidth=1.5, alpha=0.8)
    
    # Add annotations with improved contrast
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if data_matrix[i, j] > 0:
                # Format the text according to the provided format string
                text = annotation_format.format(data_matrix[i, j])
                
                text_color = 'white'
                outline_color = 'black'
   
                
                # Add text with outline for better visibility - centered in cells
                ax.text(j + 0.5, i + 0.5, text, 
                       ha='center', va='center', 
                       color=text_color, 
                       fontweight='bold', 
                       fontsize=18,
                       path_effects=[plt.matplotlib.patheffects.withStroke(
                           linewidth=3, foreground=outline_color)])
    
    # Customize plot appearance
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=18, fontweight='bold')
    ax.set_ylabel('', fontsize=18, fontweight='bold')
    
    # Center the x-tick labels in the cells
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=16)
    ax.set_xlabel('Anions', fontsize=18, fontweight='bold')
    
    # Center the y-tick labels in the cells
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=16)
    ax.set_ylabel('Transition Metals', fontsize=18, fontweight='bold')
    
    # Add colorbar with consistent styling
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label, fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # Save with high resolution
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()

def create_all_metric_visualizations(num_clusters_data, cluster_sizes_data, min_avg_distance_data,
                                   compound_counts, output_dir):
    """Create visualizations for all additional metrics."""
    
    print("\nCreating additional metric visualizations...")
    
    # Process each metric with appropriate settings
    metrics = [
        (num_clusters_data, "Number of Clusters", "plasma", False),
        (cluster_sizes_data, "Max Cluster Size", "viridis", False), 
        (min_avg_distance_data, "Min Avg Distance", "magma", False)
    ]
    
    for metric_data, metric_name, cmap, log_scale in metrics:
        create_metric_heatmaps_and_matrices(metric_data, compound_counts, metric_name, 
                                          output_dir, cmap, log_scale)

def create_comprehensive_distribution_plot(num_clusters_data, cluster_sizes_data, min_avg_distance_data, output_dir):
    """
    Create a publication-quality comprehensive violin plot showing distribution statistics 
    for all three metrics with statistical line segments and enhanced typography.
    """
    print("\nCreating comprehensive distribution analysis...")
    
    # Set publication-quality parameters
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.5,
        'grid.linewidth': 1.0,
        'lines.linewidth': 2.5
    })
    
    # Prepare data for plotting
    plot_data = []
    
    # Process num_clusters data
    for metal in num_clusters_data:
        for anion in num_clusters_data[metal]:
            for value in num_clusters_data[metal][anion]:
                plot_data.append({
                    'metric': 'Number of Clusters',
                    'value': value,
                    'metal': metal,
                    'anion': anion,
                    'system': f'{metal}-{anion}'
                })
    
    # Process cluster_sizes data (using maximum values)
    for metal in cluster_sizes_data:
        for anion in cluster_sizes_data[metal]:
            for value in cluster_sizes_data[metal][anion]:
                plot_data.append({
                    'metric': 'Max Cluster Size',
                    'value': value,
                    'metal': metal,
                    'anion': anion,
                    'system': f'{metal}-{anion}'
                })
    
    # Process min_avg_distance data
    for metal in min_avg_distance_data:
        for anion in min_avg_distance_data[metal]:
            for value in min_avg_distance_data[metal][anion]:
                plot_data.append({
                    'metric': 'Min Avg Distance (Å)',
                    'value': value,
                    'metal': metal,
                    'anion': anion,
                    'system': f'{metal}-{anion}'
                })
    
    if not plot_data:
        print("No data available for distribution plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Remove outliers using IQR method for cleaner distributions
    def remove_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    # Create figure with improved layout and more space
    fig = plt.figure(figsize=(26, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, top=0.90, bottom=0.20, 
                         left=0.08, right=0.95)
    
    # Enhanced color palette for publication
    colors = ['#E31A1C', '#1F78B4', '#33A02C']  # Red, Blue, Green
    metric_names = ['Number of Clusters', 'Max Cluster Size', 'Min Avg Distance (Å)']
    
    # 1. Main violin plots for each metric (top row)
    for i, metric in enumerate(metric_names):
        ax = fig.add_subplot(gs[0, i])
        metric_data = df[df['metric'] == metric]['value']
        
        if len(metric_data) > 0:
            # Remove outliers for cleaner visualization
            clean_data = remove_outliers(metric_data)
            
            if len(clean_data) > 10:  # Ensure enough data points
                # Create violin plot with custom styling
                violin_parts = ax.violinplot([clean_data], positions=[0], widths=0.7, 
                                           showmeans=False, showmedians=False, showextrema=False)
                
                # Style violin body
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.75)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.8)
                
                # Calculate statistics
                mean_val = np.mean(clean_data)
                median_val = np.median(clean_data)
                q25, q75 = np.percentile(clean_data, [25, 75])
                
                # Add statistical line segments within the violin with proper spacing
                violin_width = 0.35
                
                # Mean line (thick, solid) with text annotation
                ax.hlines(mean_val, -violin_width, violin_width, colors='white', 
                         linewidth=5, linestyle='-', alpha=0.95)
                ax.hlines(mean_val, -violin_width, violin_width, colors='black', 
                         linewidth=3, linestyle='-', alpha=1.0)
                
                # Add mean text directly on the line, positioned to the right
                ax.text(violin_width + 0.15, mean_val, f'μ = {mean_val:.2f}', 
                       verticalalignment='center', horizontalalignment='left',
                       fontsize=14, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor='black'))
                
                # Enhanced median line (thicker and more prominent) with text annotation
                ax.hlines(median_val, -violin_width*1.1, violin_width*1.1, colors='white', 
                         linewidth=6, linestyle='-', alpha=0.95)
                ax.hlines(median_val, -violin_width*1.1, violin_width*1.1, colors='red', 
                         linewidth=4, linestyle='-', alpha=1.0)
                
                # Add median text directly on the line, positioned to the left with enhanced styling
                ax.text(-violin_width - 0.15, median_val, f'M = {median_val:.2f}', 
                       verticalalignment='center', horizontalalignment='right',
                       fontsize=15, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                alpha=0.95, edgecolor='red', linewidth=2))
                
                # Add data range indicators
                data_min, data_max = np.min(clean_data), np.max(clean_data)
                ax.vlines(0, data_min, data_max, colors='black', linewidth=1.5, alpha=0.7)
                
                # Add summary statistics in a better position (top left corner)
                stats_text = (f'σ = {np.std(clean_data):.2f}\n'
                            f'n = {len(clean_data):,}\n'
                            f'Range: {data_min:.2f} - {data_max:.2f}')
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                                alpha=0.8, edgecolor='navy'),
                       fontsize=13, fontfamily='monospace', fontweight='bold')
                
                # Improved title and labels
                ax.set_title(f'{metric}\nDistribution Analysis', fontsize=18, 
                           fontweight='bold', pad=25)
                
                if metric == 'Min Avg Distance (Å)':
                    ax.set_ylabel('Distance (Å)', fontsize=16, fontweight='bold')
                elif 'Size' in metric:
                    ax.set_ylabel('Cluster Size', fontsize=16, fontweight='bold')
                else:
                    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
                
                # Expand x-axis limits to accommodate text annotations
                ax.set_xlim(-0.8, 0.8)
                ax.set_xticks([])
                ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.0)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
    
    # 2. Comparative analysis plots (bottom row) with improved spacing
    
    # Top metals comparison
    ax_metals = fig.add_subplot(gs[1, 0])
    
    # Get metals with most data points across all metrics
    metal_counts = df.groupby('metal').size().sort_values(ascending=False)
    top_metals = metal_counts.head(8).index.tolist()
    
    positions = []
    violin_data = []
    labels = []
    colors_extended = []
    
    pos = 0
    for i, metric in enumerate(metric_names):
        metric_df = df[df['metric'] == metric]
        for j, metal in enumerate(top_metals):
            metal_data = metric_df[metric_df['metal'] == metal]['value']
            if len(metal_data) >= 5:  # Minimum data requirement
                clean_metal_data = remove_outliers(metal_data)
                if len(clean_metal_data) >= 3:
                    violin_data.append(clean_metal_data)
                    positions.append(pos)
                    labels.append(f'{metal}\n({metric.split()[0]})')
                    colors_extended.append(colors[i])
                    pos += 1
    
    if violin_data:
        violin_parts = ax_metals.violinplot(violin_data, positions=positions, widths=0.9,
                                          showmeans=False, showmedians=False, showextrema=False)
        
        # Style each violin with appropriate color
        for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors_extended)):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)
            
            # Enhanced median lines with value annotations
            if i < len(violin_data):
                median_val = np.median(violin_data[i])
                # Thicker median line with white outline for better visibility
                ax_metals.hlines(median_val, positions[i]-0.45, positions[i]+0.45, 
                               colors='white', linewidth=5, linestyle='-')
                ax_metals.hlines(median_val, positions[i]-0.45, positions[i]+0.45, 
                               colors='red', linewidth=3.5, linestyle='-')
                
                # Add median value text above the line
                ax_metals.text(positions[i], median_val + 0.02 * (ax_metals.get_ylim()[1] - ax_metals.get_ylim()[0]), 
                             f'{median_val:.1f}', ha='center', va='bottom',
                             fontsize=12, fontweight='bold', color='red',
                             bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))
    
    ax_metals.set_title('Distribution by Top Metals', fontsize=18, fontweight='bold', pad=25)
    ax_metals.set_ylabel('Normalized Values', fontsize=16, fontweight='bold')
    ax_metals.set_xticks(positions[::2])  # Show every 2nd label to avoid crowding
    ax_metals.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], 
                             rotation=45, ha='right', fontsize=12)
    ax_metals.grid(True, alpha=0.4, linestyle=':', linewidth=1.0)
    ax_metals.spines['top'].set_visible(False)
    ax_metals.spines['right'].set_visible(False)
    
    # Anion comparison
    ax_anions = fig.add_subplot(gs[1, 1])
    
    anion_counts = df.groupby('anion').size().sort_values(ascending=False)
    good_anions = anion_counts[anion_counts >= 50].index.tolist()  # Anions with sufficient data
    
    anion_positions = []
    anion_violin_data = []
    anion_labels = []
    anion_colors = []
    
    pos = 0
    for i, metric in enumerate(metric_names):
        metric_df = df[df['metric'] == metric]
        for anion in good_anions:
            anion_data = metric_df[metric_df['anion'] == anion]['value']
            if len(anion_data) >= 10:
                clean_anion_data = remove_outliers(anion_data)
                if len(clean_anion_data) >= 5:
                    anion_violin_data.append(clean_anion_data)
                    anion_positions.append(pos)
                    anion_labels.append(f'{anion}\n({metric.split()[0]})')
                    anion_colors.append(colors[i])
                    pos += 1
    
    if anion_violin_data:
        violin_parts = ax_anions.violinplot(anion_violin_data, positions=anion_positions, 
                                          widths=0.9, showmeans=False, showmedians=False, 
                                          showextrema=False)
        
        for i, (pc, color) in enumerate(zip(violin_parts['bodies'], anion_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)
            
            # Enhanced median lines with value annotations
            if i < len(anion_violin_data):
                median_val = np.median(anion_violin_data[i])
                # Thicker median line with white outline for better visibility
                ax_anions.hlines(median_val, anion_positions[i]-0.45, anion_positions[i]+0.45, 
                               colors='white', linewidth=5, linestyle='-')
                ax_anions.hlines(median_val, anion_positions[i]-0.45, anion_positions[i]+0.45, 
                               colors='red', linewidth=3.5, linestyle='-')
                
                # Add median value text above the line
                ax_anions.text(anion_positions[i], median_val + 0.02 * (ax_anions.get_ylim()[1] - ax_anions.get_ylim()[0]), 
                             f'{median_val:.1f}', ha='center', va='bottom',
                             fontsize=12, fontweight='bold', color='red',
                             bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))
    
    ax_anions.set_title('Distribution by Anion Groups', fontsize=18, fontweight='bold', pad=25)
    ax_anions.set_ylabel('Normalized Values', fontsize=16, fontweight='bold')
    ax_anions.set_xticks(anion_positions[::2])  # Show every 2nd label
    ax_anions.set_xticklabels([anion_labels[i] for i in range(0, len(anion_labels), 2)], 
                             rotation=45, ha='right', fontsize=12)
    ax_anions.grid(True, alpha=0.4, linestyle=':', linewidth=1.0)
    ax_anions.spines['top'].set_visible(False)
    ax_anions.spines['right'].set_visible(False)
    
    # Combined density comparison
    ax_density = fig.add_subplot(gs[1, 2])
    
    for i, metric in enumerate(metric_names):
        metric_data = df[df['metric'] == metric]['value']
        if len(metric_data) > 0:
            clean_density_data = remove_outliers(metric_data)
            
            # Create density plot
            density = stats.gaussian_kde(clean_density_data)
            x_range = np.linspace(clean_density_data.min(), clean_density_data.max(), 200)
            density_values = density(x_range)
            
            ax_density.fill_between(x_range, density_values, alpha=0.7, 
                                  color=colors[i], label=metric)
            ax_density.plot(x_range, density_values, color=colors[i], 
                          linewidth=3, alpha=0.95)
    
    ax_density.set_title('Probability Density Comparison', fontsize=18, fontweight='bold', pad=25)
    ax_density.set_xlabel('Values', fontsize=16, fontweight='bold')
    ax_density.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax_density.legend(frameon=True, fancybox=True, shadow=True, fontsize=14, loc='upper right')
    ax_density.grid(True, alpha=0.4, linestyle=':', linewidth=1.0)
    ax_density.spines['top'].set_visible(False)
    ax_density.spines['right'].set_visible(False)
    
    # Overall title with enhanced styling
    fig.suptitle('Comprehensive Cluster Metrics Distribution Analysis\nPublicaton-Quality Violin Plots with Statistical Line Segments', 
                fontsize=22, fontweight='bold', y=0.95)
    
    # Add improved legend for statistical lines (without quartiles)
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=4, label='Mean (μ)'),
        plt.Line2D([0], [0], color='red', linewidth=4, label='Median (M)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.03), 
              ncol=2, fontsize=16, frameon=True, fancybox=True, shadow=True)
    
    # Save with high resolution
    plt.savefig(f"{output_dir}/comprehensive_distribution_analysis.png", 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f"{output_dir}/comprehensive_distribution_analysis.pdf", 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Saved comprehensive distribution analysis to {output_dir}")
    plt.close()
    
    # Reset matplotlib parameters to defaults
    plt.rcdefaults()
    
    return df

def analyze_point_groups(point_groups_data, materials_by_point_group):
    """
    Analyze the distribution of point groups across metals, anions, and cluster sizes.
    
    Args:
        point_groups_data: Nested dictionary with point groups by metal-anion
        materials_by_point_group: Dictionary of materials organized by point group and cluster size
        
    Returns:
        Tuple of point group statistics and analysis results
    """
    print("\nAnalyzing point group symmetry distributions...")
    
    # Count overall point group occurrences
    all_point_groups = []
    for metal in point_groups_data:
        for anion in point_groups_data[metal]:
            for pg_dict in point_groups_data[metal][anion]:
                all_point_groups.extend(pg_dict.values())
    
    point_group_counts = Counter(all_point_groups)
    
    # Count point groups by metal
    metal_point_groups = defaultdict(Counter)
    for metal in point_groups_data:
        for anion in point_groups_data[metal]:
            for pg_dict in point_groups_data[metal][anion]:
                for pg in pg_dict.values():
                    metal_point_groups[metal][pg] += 1
    
    # Count point groups by anion
    anion_point_groups = defaultdict(Counter)
    for metal in point_groups_data:
        for anion in point_groups_data[metal]:
            for pg_dict in point_groups_data[metal][anion]:
                for pg in pg_dict.values():
                    anion_point_groups[anion][pg] += 1
    
    # Count point groups by cluster size
    size_point_groups = defaultdict(Counter)
    for point_group in materials_by_point_group:
        for size in materials_by_point_group[point_group]:
            size_point_groups[size][point_group] += len(materials_by_point_group[point_group][size])
    
    # Find most common combinations
    metal_pg_combinations = defaultdict(Counter)
    anion_pg_combinations = defaultdict(Counter)
    
    for metal in point_groups_data:
        for anion in point_groups_data[metal]:
            for pg_dict in point_groups_data[metal][anion]:
                for pg in pg_dict.values():
                    metal_pg_combinations[pg][metal] += 1
                    anion_pg_combinations[pg][anion] += 1
    
    return (point_group_counts, metal_point_groups, anion_point_groups, 
            size_point_groups, metal_pg_combinations, anion_pg_combinations)

def find_exemplar_materials(materials_by_point_group, size_range=[2, 3, 4, 5, 6, 7, 8]):
    """
    Find exemplar materials for top point groups in each cluster size.
    
    Args:
        materials_by_point_group: Dictionary of materials by point group and size
        size_range: List of cluster sizes to analyze
        
    Returns:
        Dictionary of exemplar materials for each point group and size
    """
    exemplars = {}
    
    for point_group in materials_by_point_group:
        exemplars[point_group] = {}
        
        for size in size_range:
            if size in materials_by_point_group[point_group] and materials_by_point_group[point_group][size]:
                # Sort by material_id for consistency
                sorted_materials = sorted(materials_by_point_group[point_group][size], 
                                        key=lambda x: str(x.get('material_id', '')))
                # Take the first example
                exemplars[point_group][size] = sorted_materials[0]
    
    return exemplars

def create_point_group_visualizations(point_group_counts, metal_point_groups, anion_point_groups, 
                                    size_point_groups, metal_pg_combinations, anion_pg_combinations,
                                    exemplars, output_dir):
    """
    Create publication-quality visualizations for point group analysis.
    
    Args:
        point_group_counts: Counter of point group occurrences
        metal_point_groups: Counter of point groups by metal
        anion_point_groups: Counter of point groups by anion
        size_point_groups: Counter of point groups by size
        metal_pg_combinations: Counter of metals by point group
        anion_pg_combinations: Counter of anions by point group
        exemplars: Dictionary of exemplar materials
        output_dir: Output directory path
    """
    print("\nCreating point group visualizations...")
    
    # Set publication-quality parameters for all plots
    set_publication_style()
    
    # 1. Overall point group distribution
    plt.figure(figsize=(16, 12))
    top_groups = dict(point_group_counts.most_common(20))
    
    # Create a colorful bar chart with enhanced styling
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_groups)))
    bars = plt.bar(range(len(top_groups)), list(top_groups.values()), color=colors, width=0.7)
    
    # Add count labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*height,
                f'{height}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.title('Distribution of Point Group Symmetries in Clusters', fontweight='bold', pad=20)
    plt.xlabel('Point Group', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks(range(len(top_groups)), list(top_groups.keys()), rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Point group distribution by cluster size (heatmap)
    # Prepare data for heatmap - INCLUDE ALL POINT GROUPS
    
    # Find all point groups that appear in the data
    all_point_groups = list(point_group_counts.keys())
    
    # Filter for important high-symmetry groups + most common groups
    important_groups = ['Oh', 'Td', 'Th', 'D*h', 'C*v', 'D3h', 'C3v', 'C2v', 'C1', 'Cs', 'Ci', 'D2h', 'D2d', 'D4h', 'D3d', 'I', 'Ih']
    important_found = [pg for pg in important_groups if pg in all_point_groups]
    
    # Add most common groups that aren't already in important_groups
    most_common = [pg for pg, _ in point_group_counts.most_common(15) if pg not in important_found]
    
    # Combine and sort by frequency
    selected_point_groups = important_found + most_common
    selected_point_groups = sorted(selected_point_groups, 
                                  key=lambda pg: point_group_counts.get(pg, 0), 
                                  reverse=True)
    
    # Ensure there aren't too many groups for clarity
    if len(selected_point_groups) > 15:
        selected_point_groups = selected_point_groups[:15]
    
    # Get all cluster sizes from the data and sort
    size_range = sorted([size for size in size_point_groups.keys() if isinstance(size, (int, float))])
    
    # Filter size range to requested values if possible
    target_sizes = [2, 3, 4, 5, 6, 7, 8]
    filtered_sizes = [size for size in size_range if size in target_sizes]
    if filtered_sizes:
        size_range = filtered_sizes
    
    # Create matrix for heatmap
    pg_size_matrix = np.zeros((len(selected_point_groups), len(size_range)))
    
    for i, pg in enumerate(selected_point_groups):
        for j, size in enumerate(size_range):
            if size in size_point_groups and pg in size_point_groups[size]:
                pg_size_matrix[i, j] = size_point_groups[size][pg]
    
    # Create a custom annotation function for point group-cluster size plot
    def point_group_size_annotate(ax, matrix, row_labels, col_labels):
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                if matrix[i, j] > 0:
                    # Determine text color based on value intensity
                    normalized_value = (matrix[i, j] - np.min(matrix[matrix > 0])) / (np.max(matrix) - np.min(matrix[matrix > 0]))
                    text_color = 'white'
                    outline_color = 'black'
                    
                    # Add count with outline for better visibility
                    ax.text(j + 0.5, i + 0.5, f'{int(matrix[i, j])}', 
                           ha='center', va='center', color=text_color,
                           fontweight='bold', fontsize=18,
                           path_effects=[plt.matplotlib.patheffects.withStroke(
                               linewidth=3, foreground=outline_color)])
    
    # Set up figure with appropriate size
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use a colorful heatmap with improved aesthetics
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=np.max(pg_size_matrix) if np.max(pg_size_matrix) > 0 else 1)
    
    # Turn off default grid
    ax.grid(False)
    
    # Use pcolormesh for precise grid alignment
    x = np.arange(len(size_range) + 1)
    y = np.arange(len(selected_point_groups) + 1)
    mesh = ax.pcolormesh(x, y, pg_size_matrix, cmap=cmap, norm=norm)
    
    # Add explicit grid lines that align perfectly with cell boundaries
    for i in range(len(selected_point_groups) + 1):
        ax.axhline(y=i, color='white', linewidth=1.5, alpha=0.8)
    for j in range(len(size_range) + 1):
        ax.axvline(x=j, color='white', linewidth=1.5, alpha=0.8)
    
    # Add custom annotations
    point_group_size_annotate(ax, pg_size_matrix, selected_point_groups, size_range)
    
    # Set labels and title - removed "with Example Material IDs" from title
    ax.set_xticks(np.arange(len(size_range)) + 0.5)
    ax.set_xticklabels([f'Size {size}' for size in size_range], fontsize=14)
    ax.set_yticks(np.arange(len(selected_point_groups)) + 0.5)
    ax.set_yticklabels(selected_point_groups, fontsize=14)
    
    ax.set_title('Point Group Distribution by Cluster Size', 
               fontweight='bold', pad=20)
    ax.set_xlabel('Cluster Size', fontweight='bold')
    ax.set_ylabel('Point Group', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Number of Clusters', fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/point_group_by_size.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/point_group_by_size.pdf", bbox_inches='tight')
    plt.close()
    
    # 3. Top metals and anions for each major point group (horizontal bar charts)
    # Get top point groups including high symmetry groups
    important_pgs = [pg for pg in ['Oh', 'Td', 'D*h', 'C*v', 'C3v', 'C2v', 'Cs', 'D3h'] if pg in point_group_counts]
    other_top_pgs = [pg for pg, _ in point_group_counts.most_common() if pg not in important_pgs]
    top_pgs = important_pgs + other_top_pgs
    top_pgs = top_pgs[:5]  # Limit to 5 for visualization clarity
    
    # Prepare figure with subplots for each point group
    fig, axes = plt.subplots(len(top_pgs), 2, figsize=(18, 5*len(top_pgs)))
    
    for i, pg in enumerate(top_pgs):
        # Top metals for this point group
        ax_metal = axes[i, 0]
        if pg in metal_pg_combinations:
            # Reverse the order (higher to lower) as requested
            top_metals = dict(metal_pg_combinations[pg].most_common(8))
            # Reverse the dictionary order
            top_metals = {k: v for k, v in reversed(list(top_metals.items()))}
            
            # Create horizontal bar chart
            metal_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(top_metals)))
            metal_bars = ax_metal.barh(range(len(top_metals)), list(top_metals.values()), 
                                     color=metal_colors, height=0.7)
            
            # Add count labels
            for bar in metal_bars:
                width = bar.get_width()
                ax_metal.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{int(width)}', va='center', fontsize=14, fontweight='bold')
            
            ax_metal.set_yticks(range(len(top_metals)))
            ax_metal.set_yticklabels(list(top_metals.keys()), fontsize=14)
            ax_metal.set_title(f'Top Metals for {pg}', fontweight='bold', fontsize=18)
            ax_metal.set_xlabel('Count', fontweight='bold', fontsize=16)
            ax_metal.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Top anions for this point group
        ax_anion = axes[i, 1]
        if pg in anion_pg_combinations:
            # Reverse the order (higher to lower) as requested
            top_anions = dict(anion_pg_combinations[pg].most_common(8))
            # Reverse the dictionary order
            top_anions = {k: v for k, v in reversed(list(top_anions.items()))}
            
            # Create horizontal bar chart
            anion_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_anions)))
            anion_bars = ax_anion.barh(range(len(top_anions)), list(top_anions.values()), 
                                     color=anion_colors, height=0.7)
            
            # Add count labels
            for bar in anion_bars:
                width = bar.get_width()
                ax_anion.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{int(width)}', va='center', fontsize=14, fontweight='bold')
            
            ax_anion.set_yticks(range(len(top_anions)))
            ax_anion.set_yticklabels(list(top_anions.keys()), fontsize=14)
            ax_anion.set_title(f'Top Anions for {pg}', fontweight='bold', fontsize=18)
            ax_anion.set_xlabel('Count', fontweight='bold', fontsize=16)
            ax_anion.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.suptitle('Metal and Anion Distributions for Top Point Groups', 
                fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{output_dir}/point_group_metal_anion_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/point_group_metal_anion_distribution.pdf", bbox_inches='tight')
    plt.close()
    
    # 4. Correlation matrix between metals and point groups
    # Prepare correlation matrix with all point groups
    top_metals = [metal for metal, _ in Counter({metal: sum(counts.values()) 
                                               for metal, counts in metal_point_groups.items()}).most_common(10)]
    
    # Use the same selected point groups as the size heatmap
    metal_pg_matrix = np.zeros((len(top_metals), len(selected_point_groups)))
    
    for i, metal in enumerate(top_metals):
        for j, pg in enumerate(selected_point_groups):
            if metal in metal_point_groups and pg in metal_point_groups[metal]:
                metal_pg_matrix[i, j] = metal_point_groups[metal][pg]
    
    # Set up figure with appropriate size
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use a custom diverging colormap
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=np.max(metal_pg_matrix) if np.max(metal_pg_matrix) > 0 else 1)
    
    # Turn off default grid
    ax.grid(False)
    
    # Use pcolormesh for precise grid alignment
    x = np.arange(len(selected_point_groups) + 1)
    y = np.arange(len(top_metals) + 1)
    mesh = ax.pcolormesh(x, y, metal_pg_matrix, cmap=cmap, norm=norm)
    
    # Add explicit grid lines that align perfectly with cell boundaries
    for i in range(len(top_metals) + 1):
        ax.axhline(y=i, color='white', linewidth=1.5, alpha=0.8)
    for j in range(len(selected_point_groups) + 1):
        ax.axvline(x=j, color='white', linewidth=1.5, alpha=0.8)
    
    # Add value annotations
    for i in range(len(top_metals)):
        for j in range(len(selected_point_groups)):
            if metal_pg_matrix[i, j] > 0:
                text_color = 'white'
                outline_color = 'black'
                
                # Add count with outline for better visibility
                ax.text(j + 0.5, i + 0.5, f'{int(metal_pg_matrix[i, j])}', 
                       ha='center', va='center', color=text_color,
                       fontweight='bold', fontsize=20,
                       path_effects=[plt.matplotlib.patheffects.withStroke(
                           linewidth=3, foreground=outline_color)])
    
    # Set labels and title
    ax.set_xticks(np.arange(len(selected_point_groups)) + 0.5)
    ax.set_xticklabels(selected_point_groups, rotation=45, ha='right', fontsize=18)
    ax.set_yticks(np.arange(len(top_metals)) + 0.5)
    ax.set_yticklabels(top_metals, fontsize=18)
    
    ax.set_title('Correlation Between Transition Metals and Point Groups', 
               fontweight='bold', pad=20, fontsize=26)
    ax.set_xlabel('Point Group', fontweight='bold', fontsize=24)
    ax.set_ylabel('Transition Metal', fontweight='bold', fontsize=24)
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Number of Clusters', fontweight='bold', fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metal_point_group_correlation.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/metal_point_group_correlation.pdf", bbox_inches='tight')
    plt.close()
    
    # 5. Distribution of top point groups across transition metals
    # Take important high symmetry groups and common groups
    top_important_groups = []
    for pg in ['Oh', 'Td', 'D*h', 'C*v', 'C3v', 'C2v', 'Cs', 'D3h']:
        if pg in point_group_counts:
            top_important_groups.append(pg)
    
    # If we don't have enough groups, add more from most common
    if len(top_important_groups) < 5:
        for pg, _ in point_group_counts.most_common():
            if pg not in top_important_groups:
                top_important_groups.append(pg)
                if len(top_important_groups) >= 5:
                    break
    
    # Select top 5 for clarity
    top_pg_groups = top_important_groups[:5]
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Prepare data for grouped bar chart
    x_positions = np.arange(len(top_metals))
    width = 0.15  # Width of each bar
    offsets = np.linspace(-0.3, 0.3, len(top_pg_groups))
    
    for i, pg in enumerate(top_pg_groups):
        counts = []
        for metal in top_metals:
            if metal in metal_point_groups and pg in metal_point_groups[metal]:
                counts.append(metal_point_groups[metal][pg])
            else:
                counts.append(0)
        
        bars = ax.bar(x_positions + offsets[i], counts, width, 
                    label=f'{pg}', alpha=0.8, 
                    color=plt.cm.viridis(i/len(top_pg_groups)))
        
        # Add count labels for non-zero values
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                      f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Set labels and title
    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_metals, fontsize=14)
    ax.set_title('Distribution of Top Point Groups Across Transition Metals', 
               fontweight='bold', pad=20, fontsize=22)
    ax.set_xlabel('Transition Metal', fontweight='bold', fontsize=18)
    ax.set_ylabel('Number of Clusters', fontweight='bold', fontsize=18)
    ax.legend(title='Point Group', frameon=True, fancybox=True, shadow=True, 
             fontsize=14, title_fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add note about point group interpretation
    plt.figtext(0.5, 0.01, 
               "Point group interpretations: D*h (linear arrangements), Oh (octahedral), Td (tetrahedral),\n"
               "C3v (trigonal pyramidal), C2v (angular), Cs (mirror plane), C1 (no symmetry)",
               fontsize=14, ha='center', 
               bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(f"{output_dir}/top_point_groups_by_metal.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/top_point_groups_by_metal.pdf", bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib parameters
    plt.rcdefaults()

def generate_point_group_report(point_group_counts, metal_point_groups, anion_point_groups, 
                              size_point_groups, metal_pg_combinations, anion_pg_combinations,
                              exemplars, output_dir):
    """
    Generate a comprehensive report on point group analysis with examples.
    
    Args:
               point_group_counts: Counter of point group occurrences
        metal_point_groups: Counter of point groups by metal
        anion_point_groups: Counter of point groups by anion
        size_point_groups: Counter of point groups by size
        exemplars: Dictionary of exemplar materials
        output_dir: Output directory path
    """
    report_path = f"{output_dir}/point_group_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Point Group Symmetry Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("This analysis examines the distribution of point groups (symmetry groups) in\n")
        f.write("transition metal clusters. Point groups describe the symmetry of molecular\n")
        f.write("arrangements, revealing information about structural preferences.\n\n")
        
        # Overall statistics
        total_clusters = sum(point_group_counts.values())
        
        f.write(f"OVERALL STATISTICS:\n")
        f.write(f"Total clusters with point group data: {total_clusters:,}\n")
        f.write(f"Number of unique point groups: {len(point_group_counts):,}\n\n")
        
        # Top point groups
        f.write("TOP POINT GROUPS BY FREQUENCY:\n")
        f.write("=" * 60 + "\n")
        f.write("Rank  Point Group  Count  Percentage\n")
        f.write("-" * 60 + "\n")
        for i, (pg, count) in enumerate(point_group_counts.most_common(15), 1):
            percentage = (count / total_clusters) * 100
            f.write(f"{i:2d}.   {pg:10s}    {count:5d}   {percentage:6.2f}%\n")
        
        # Point groups by cluster size
        f.write("\nPOINT GROUP DISTRIBUTION BY CLUSTER SIZE:\n")
        f.write("=" * 70 + "\n")
        f.write("Size   Top Point Groups (count)\n")
        f.write("-" * 70 + "\n")
        
        # Sort sizes numerically, including only numeric sizes
        numeric_sizes = [size for size in size_point_groups.keys() if isinstance(size, (int, float))]
        target_sizes = [2, 3, 4, 5, 6, 7, 8]
        filtered_sizes = [size for size in sorted(numeric_sizes) if size in target_sizes]
        
        for size in filtered_sizes:
            top_pgs = size_point_groups[size].most_common(10)
            f.write(f"{int(size):4d}   {', '.join([f'{pg} ({count})' for pg, count in top_pgs])}\n")
        
        # Top metals for each major point group
        f.write("\nTOP METALS FOR MAJOR POINT GROUPS:\n")
        f.write("=" * 70 + "\n")
        
        for pg, count in point_group_counts.most_common(8):
            f.write(f"\nPoint Group: {pg} (Total: {count})\n")
            f.write("-" * 40 + "\n")
            f.write("Rank  Metal  Count  Percentage\n")
            f.write("-" * 40 + "\n")
            
            if pg in metal_pg_combinations:
                for i, (metal, metal_count) in enumerate(metal_pg_combinations[pg].most_common(8), 1):
                    percentage = (metal_count / count) * 100
                    f.write(f"{i:2d}.   {metal:4s}  {metal_count:5d}   {percentage:6.2f}%\n")
        
        # Top anions for each major point group
        f.write("\nTOP ANIONS FOR MAJOR POINT GROUPS:\n")
        f.write("=" * 70 + "\n")
        
        for pg, count in point_group_counts.most_common(8):
            f.write(f"\nPoint Group: {pg} (Total: {count})\n")
            f.write("-" * 40 + "\n")
            f.write("Rank  Anion  Count  Percentage\n")
            f.write("-" * 40 + "\n")
            
            if pg in anion_pg_combinations:
                for i, (anion, anion_count) in enumerate(anion_pg_combinations[pg].most_common(8), 1):
                    percentage = (anion_count / count) * 100
                    f.write(f"{i:2d}.   {anion:5s}  {anion_count:5d}   {percentage:6.2f}%\n")
        
        # Example materials for each point group and size
        f.write("\nEXAMPLE MATERIALS BY POINT GROUP AND CLUSTER SIZE:\n")
        f.write("=" * 80 + "\n")
        
        for pg, count in point_group_counts.most_common(8):
            f.write(f"\nPoint Group: {pg}\n")
            f.write("-" * 80 + "\n")
            f.write("Size  Material ID                  System        Metal-Anion\n")
            f.write("-" * 80 + "\n")
            
            if pg in exemplars:
                for size in sorted(exemplars[pg].keys()):
                    example = exemplars[pg][size]
                    material_id = example.get('material_id', 'Unknown')
                    system = example.get('compound_system', 'Unknown')
                    metal = example.get('metal', 'Unknown')
                    anion = example.get('anion', 'Unknown')
                    
                    f.write(f"{int(size):4d}  {material_id:30s}  {system:12s}  {metal}-{anion}\n")
        
        f.write("\nPOINT GROUP INTERPRETATIONS:\n")
        f.write("=" * 80 + "\n")
        f.write("Point Group   Description\n")
        f.write("-" * 80 + "\n")
        f.write("D*h           High symmetry linear arrangements (e.g., linear metal chains)\n")
        f.write("C*v           Cylindrical symmetry with infinite rotation axis and vertical mirror planes\n")
        f.write("C3v           3-fold rotational symmetry with vertical mirror planes (e.g., trigonal pyramids)\n")
        f.write("C2v           2-fold rotational symmetry with vertical mirror planes (e.g., angular molecules)\n")
        f.write("Cs            Only a single mirror plane symmetry element\n")
        f.write("C1            No symmetry elements (asymmetric structures)\n")
        f.write("Ci            Only a center of inversion\n")
        f.write("D2h           2-fold rotational symmetry with horizontal mirror plane (e.g., planar rectangles)\n")
        f.write("D3h           3-fold rotational symmetry with horizontal mirror plane (e.g., trigonal planar)\n")
        f.write("D2d           2-fold rotational symmetry with perpendicular 2-fold axes (e.g., tetrahedra)\n")
        
        f.write("\nCONCLUSIONS AND INSIGHTS:\n")
        f.write("=" * 80 + "\n")
        f.write("1. The most common point group symmetries in transition metal clusters are D*h and C*v,\n")
        f.write("   suggesting a preference for linear and cylindrical arrangements.\n\n")
        
        f.write("2. Different metals show distinct preferences for specific point groups:\n")
        for metal, counts in sorted(metal_point_groups.items(), 
                                 key=lambda x: sum(x[1].values()), reverse=True)[:5]:
            top_pg = counts.most_common(1)[0][0]
            percentage = (counts[top_pg] / sum(counts.values())) * 100
            f.write(f"   - {metal}: predominantly forms {top_pg} clusters ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("3. Cluster size correlates with point group distribution:\n")
        for size in [2, 3, 4, 5, 6]:
            if size in size_point_groups and size_point_groups[size]:
                top_pg = size_point_groups[size].most_common(1)[0][0]
                percentage = (size_point_groups[size][top_pg] / sum(size_point_groups[size].values())) * 100
                f.write(f"   - Size {size}: predominantly {top_pg} symmetry ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("4. Specific metal-anion combinations show strong preferences for particular symmetries:\n")
        
        # Top metal-anion combinations for each major point group
        top_pgs = [pg for pg, _ in point_group_counts.most_common(3)]
        for pg in top_pgs:
            if pg in metal_pg_combinations and pg in anion_pg_combinations:
                top_metal = metal_pg_combinations[pg].most_common(1)[0][0]
                top_anion = anion_pg_combinations[pg].most_common(1)[0][0]
                f.write(f"   - {top_metal}-{top_anion} commonly forms {pg} clusters\n")
        
        f.write("\nThe above analysis provides valuable insights into structure-composition relationships\n")
        f.write("for transition metal clusters, which can guide future materials design and discovery.\n")
    
    print(f"Comprehensive point group report saved to {report_path}")
    
def set_publication_style():
    """Set standardized publication-quality parameters for all plots."""
    plt.rcParams.update({
        'font.size': 18,                # Larger base font size
        'axes.titlesize': 22,           # Larger title
        'axes.labelsize': 20,           # Larger axis labels
        'xtick.labelsize': 16,          # Larger tick labels
        'ytick.labelsize': 16,
        'legend.fontsize': 16,          # Larger legend
        'figure.titlesize': 24,         # Larger figure title
        'font.family': 'serif',         # Serif fonts for academic publishing
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',     # Professional math fonts
        'axes.linewidth': 1.8,          # Thicker axis lines
        'grid.linewidth': 1.2,          # Thicker grid lines
        'lines.linewidth': 2.8,         # Thicker plot lines
        'patch.linewidth': 2.0,         # Thicker outlines
        'axes.grid': True,              # Show grid by default
        'grid.alpha': 0.3,              # Subtle grid
        'savefig.dpi': 300,             # High resolution
        'savefig.bbox': 'tight',        # Tight layout when saving
        'savefig.pad_inches': 0.2,      # Small padding
        'figure.figsize': (12, 9),      # Larger default figure size
        'figure.autolayout': True       # Automatic layout
    })

def main():
    """Main function to run the normalized periodic table analysis."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    cluster_csv_path = script_dir.parent / "merged_results_all_prop_filtered_v3.csv"
    summary_csv_path = script_dir.parent / "summary.csv"
    output_dir = script_dir / "periodic_plots"
    
    print("Normalized Periodic Table Cluster Analysis")
    print("=" * 50)
    print(f"Cluster data: {cluster_csv_path}")
    print(f"Summary data: {summary_csv_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    compound_counts, metal_compound_totals, anion_compound_totals = load_summary_data(summary_csv_path)
    
    # Update to handle the new return values for point group analysis
    cluster_data, metal_cluster_totals, anion_cluster_totals, num_clusters_data, cluster_sizes_data, min_avg_distance_data, point_groups_data, materials_by_point_group = load_and_process_cluster_data(cluster_csv_path)
    
    if not cluster_data or not compound_counts:
        print("Failed to load required data. Exiting.")
        return
    
    # Calculate normalized data
    print("\nCalculating cluster formation efficiency...")
    normalized_data, metal_efficiency, anion_efficiency = calculate_normalized_data(cluster_data, compound_counts)
    
    print(f"Calculated efficiency for {len(metal_efficiency)} metals and {len(anion_efficiency)} anions")
    
    # Create CSV files for periodic_trends
    metal_csv = output_dir / "metal_efficiency.csv"
    create_csv_for_periodic_trends(metal_efficiency, metal_csv, "metal efficiency")
    
    # Create periodic table heatmaps using periodic_trends
    print("\nCreating periodic table heatmaps...")
    
    try:
        # Metal efficiency heatmap
        plot_periodic_table_heatmap(
            str(metal_csv),
            str(output_dir / "metal_efficiency_periodic_table.html"),
            "Transition Metal Cluster Formation Efficiency (Clusters per Compound)",
            cmap="plasma"
        )
        print("Created metal efficiency periodic table heatmap")
        
        # Create anion-specific efficiency CSVs and plots
        top_anions = sorted(anion_efficiency.items(), key=lambda x: x[1], reverse=True)[:5]
        for anion, _ in top_anions:
            anion_metal_efficiency = {}
            for metal in normalized_data:
                if anion in normalized_data[metal]:
                    anion_metal_efficiency[metal] = normalized_data[metal][anion]
            
            if anion_metal_efficiency:
                anion_csv = output_dir / f"metal_{anion}_efficiency.csv"
                create_csv_for_periodic_trends(anion_metal_efficiency, anion_csv, f"metal-{anion} efficiency")
                
                plot_periodic_table_heatmap(
                    str(anion_csv),
                    str(output_dir / f"metal_{anion}_efficiency_periodic_table.html"),
                    f"Metal-{anion} Cluster Formation Efficiency",
                    cmap="viridis"
                )
                print(f"Created metal-{anion} efficiency periodic table")
        
    except Exception as e:
        print(f"Error creating periodic table plots: {e}")
    
    # Create additional matplotlib visualizations
    print("\nCreating additional visualizations...")
    create_matplotlib_heatmaps(normalized_data, metal_efficiency, anion_efficiency, output_dir)
    
    # Create visualizations for additional metrics
    create_all_metric_visualizations(num_clusters_data, cluster_sizes_data, min_avg_distance_data,
                                   compound_counts, output_dir)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    generate_comprehensive_report(cluster_data, compound_counts, normalized_data, 
                                metal_efficiency, anion_efficiency, output_dir)
    
    # Create comprehensive distribution plot
    create_comprehensive_distribution_plot(num_clusters_data, cluster_sizes_data, min_avg_distance_data, output_dir)
    
    # Analyze point group distributions if data is available
    if point_groups_data and materials_by_point_group:
        print("\nAnalyzing point group distributions...")
        point_group_counts, metal_point_groups, anion_point_groups, size_point_groups, metal_pg_combinations, anion_pg_combinations = analyze_point_groups(point_groups_data, materials_by_point_group)
        
        # Find exemplars for top point groups
        exemplars = find_exemplar_materials(materials_by_point_group)
        
        # Create visualizations for point group analysis
        create_point_group_visualizations(point_group_counts, metal_point_groups, anion_point_groups, 
                                        size_point_groups, metal_pg_combinations, anion_pg_combinations,
                                        exemplars, output_dir)
        
        # Generate point group report
        generate_point_group_report(point_group_counts, metal_point_groups, anion_point_groups, 
                                  size_point_groups, metal_pg_combinations, anion_pg_combinations,
                                  exemplars, output_dir)
    
    print(f"\nAnalysis complete! Results saved in {output_dir}")
    print("\nGenerated files:")
    print("- metal_efficiency_periodic_table.html (interactive)")
    print("- metal_*_efficiency_periodic_table.html (anion-specific)")
    print("- metal_number_of_clusters_periodic_table.html")
    print("- metal_max_cluster_size_periodic_table.html")
    print("- metal_min_avg_distance_periodic_table.html")
    print("- efficiency_interaction_matrix.png/pdf")
    print("- number_of_clusters_interaction_matrix.png/pdf")
    print("- max_cluster_size_interaction_matrix.png/pdf")
    print("- min_avg_distance_interaction_matrix.png/pdf")
    print("- comprehensive_distribution_lava_lamp.png/pdf")
    print("- normalized_cluster_analysis_report.txt")
    if point_groups_data:
        print("- point_group_distribution.png/pdf")
        print("- point_group_by_size.png/pdf")
        print("- point_group_metal_anion_distribution.png/pdf")
        print("- metal_point_group_correlation.png/pdf")
        print("- top_point_groups_by_metal.png/pdf")
        print("- point_group_analysis_report.txt")
    print("- *.csv files for periodic_trends data")

if __name__ == "__main__":
    main()