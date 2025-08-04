#!/usr/bin/env python
"""
Magnetization-Cluster Visualization Script

This script creates publication-quality visualizations exploring the relationship between
total_magnetization and cluster properties including:
1. Total magnetization vs cluster number
2. Total magnetization vs cluster size  
3. Total magnetization vs point group
4. Magnetization distribution analysis

Uses the same formatting style as property_visualizations.py for consistency.
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import plasma, inferno, magma, viridis, cividis, turbo
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
from collections import Counter, defaultdict
from pathlib import Path
import re

# Set publication-quality parameters
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

def load_data(csv_path):
    """
    Load and preprocess the data from the CSV file.
    
    Args:
        csv_path: Path to the CSV file containing the data
        
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    print(f"Loading data from {csv_path}...")
    
    try:
        # Use pandas optimized CSV reader with appropriate data types
        df = pd.read_csv(csv_path, engine='c', 
                         dtype={'material_id': str, 'formula': str, 
                                'space_group': str, 'predicted_dimentionality': str})
        
        print(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_magnetization_data(df):
    """
    Preprocess the data specifically for magnetization analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pandas.DataFrame: Processed DataFrame ready for magnetization visualization
    """
    if df is None or len(df) == 0:
        print("No data to process")
        return None
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean and convert total_magnetization to numeric
    processed_df['total_magnetization_clean'] = pd.to_numeric(processed_df['total_magnetization'], errors='coerce')
    
    # Clean num_clusters
    processed_df['num_clusters_clean'] = pd.to_numeric(processed_df['num_clusters'], errors='coerce')
    
    # Extract max cluster size from cluster_sizes
    def extract_max_cluster_size(size_str):
        if pd.isna(size_str):
            return 0
        try:
            if isinstance(size_str, str) and '[' in size_str:
                sizes = ast.literal_eval(size_str)
                if isinstance(sizes, list) and sizes:
                    return max(sizes)
                return 0
            elif isinstance(size_str, list):
                return max(size_str) if size_str else 0
            else:
                return float(size_str)
        except:
            return 0
    
    processed_df['max_cluster_size'] = processed_df['cluster_sizes'].apply(extract_max_cluster_size)
    
    # Extract main point group
    def extract_main_point_group(pg_str):
        if pd.isna(pg_str) or not isinstance(pg_str, str):
            return "Unknown"
        try:
            if '{' in pg_str:
                pg_dict = ast.literal_eval(pg_str.replace("'", '"'))
                if isinstance(pg_dict, dict) and pg_dict:
                    return next(iter(pg_dict.values()))
            return pg_str
        except:
            return "Unknown"
    
    if 'point_groups' in processed_df.columns:
        processed_df['main_point_group'] = processed_df['point_groups'].apply(extract_main_point_group)
    else:
        processed_df['main_point_group'] = "Unknown"
    
    # Create magnetization categories
    def categorize_magnetization(mag):
        if pd.isna(mag):
            return "Unknown"
        if abs(mag) < 0.1:
            return "Non-magnetic (|μ| < 0.1)"
        elif abs(mag) < 1.0:
            return "Weakly magnetic (0.1 ≤ |μ| < 1.0)"
        elif abs(mag) < 5.0:
            return "Moderately magnetic (1.0 ≤ |μ| < 5.0)"
        else:
            return "Strongly magnetic (|μ| ≥ 5.0)"
    
    processed_df['magnetization_category'] = processed_df['total_magnetization_clean'].apply(categorize_magnetization)
    
    # Create cluster size categories
    unique_sizes = sorted([s for s in processed_df['max_cluster_size'].unique() if s > 0 and s <= 20])
    size_categories = [str(int(s)) for s in unique_sizes[:10]]
    processed_df['size_category'] = processed_df['max_cluster_size'].apply(
        lambda x: str(int(x)) if x in unique_sizes[:10] else ("Other" if x > 0 else "0")
    )
    if len(unique_sizes) > 10:
        size_categories.append("Other")
    
    # Create cluster number categories (1 to 10)
    processed_df['num_clusters_limited'] = processed_df['num_clusters_clean'].apply(
        lambda x: min(int(x), 10) if x > 0 else 0
    )
    processed_df['cluster_num_category'] = processed_df['num_clusters_limited'].apply(
        lambda x: str(int(x)) if 1 <= x <= 10 else "0"
    )
    
    print(f"Preprocessing complete: {len(processed_df)} records")
    print(f"Magnetization data available for {processed_df['total_magnetization_clean'].notna().sum()} materials")
    return processed_df

def plot_magnetization_distribution(df, output_dir, cmap="viridis"):
    """
    Create publication-quality pie chart of magnetization distribution.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
        cmap: Colormap to use
    """
    print("Creating magnetization distribution pie chart...")
    
    # Filter out materials with missing magnetization data
    mag_df = df.dropna(subset=['total_magnetization_clean'])
    
    if len(mag_df) == 0:
        print("Warning: No materials with magnetization data found")
        return
    
    # Count magnetization categories
    mag_counts = mag_df['magnetization_category'].value_counts()
    
    # Create a figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Choose visually pleasing colors
    colormap = plt.cm.get_cmap(cmap)
    colors = [colormap(i/len(mag_counts)) for i in range(len(mag_counts))]
    
    # Create wedge properties
    wedgeprops = {'linewidth': 2, 'edgecolor': 'white'}
    
    # Determine if we should show percentage labels based on slice size
    total_count = sum(mag_counts.values)
    
    def autopct_format(pct):
        # Only show percentage if slice is larger than 5%
        return f'{pct:.1f}%' if pct > 5 else ''
    
    # Create the pie chart with enhancements
    wedges, texts, autotexts = ax.pie(
        mag_counts.values,
        labels=None,
        autopct=autopct_format,
        startangle=90,
        wedgeprops=wedgeprops,
        colors=colors,
        textprops={'fontsize': 16, 'fontweight': 'bold'},
        pctdistance=0.85
    )
    
    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add title with enhanced styling
    plt.title('Distribution of Total Magnetization', fontweight='bold', fontsize=24, pad=20)
    
    # Create legend with counts and percentages
    legend_labels = []
    for category, count in mag_counts.items():
        percentage = (count / total_count) * 100
        legend_labels.append(f"{category}: {count} ({percentage:.1f}%)")
    
    plt.legend(wedges, legend_labels, title="Magnetization Categories", 
              loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1),
              fontsize=14, title_fontsize=16)
    
    # Add annotation for total count
    plt.annotate(f'Total: {total_count}',
                xy=(-0.12, -0.12), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="0.9", ec="0.5", alpha=0.8),
                fontsize=14, ha='left', va='center')
    
    # Save figures in multiple formats
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_distribution.pdf", bbox_inches='tight')
    print(f"Saved magnetization distribution plot to {output_dir}")
    plt.close()

def plot_magnetization_vs_cluster_number(df, output_dir):
    """
    Create scatter plot of magnetization vs number of clusters.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization vs cluster number scatter plot...")
    
    # Filter valid data
    valid_df = df.dropna(subset=['total_magnetization_clean', 'num_clusters_clean'])
    valid_df = valid_df[valid_df['num_clusters_clean'] > 0]
    
    if len(valid_df) == 0:
        print("Warning: No valid data for magnetization vs cluster number plot")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with color coding by magnetization category
    mag_colors = {
        "Non-magnetic (|μ| < 0.1)": '#1f77b4',        # Blue
        "Weakly magnetic (0.1 ≤ |μ| < 1.0)": '#ff7f0e',  # Orange
        "Moderately magnetic (1.0 ≤ |μ| < 5.0)": '#2ca02c',  # Green
        "Strongly magnetic (|μ| ≥ 5.0)": '#d62728'     # Red
    }
    
    for mag_cat in mag_colors.keys():
        mask = valid_df['magnetization_category'] == mag_cat
        if mask.any():
            plt.scatter(valid_df[mask]['num_clusters_clean'], 
                       valid_df[mask]['total_magnetization_clean'],
                       c=mag_colors[mag_cat], label=mag_cat, 
                       alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
    
    # Add trend line
    if len(valid_df) > 5:
        z = np.polyfit(valid_df['num_clusters_clean'], valid_df['total_magnetization_clean'], 1)
        p = np.poly1d(z)
        plt.plot(valid_df['num_clusters_clean'], p(valid_df['num_clusters_clean']), 
                "r--", alpha=0.8, linewidth=2, label='Trend line')
        
        # Calculate and display correlation coefficient
        correlation = np.corrcoef(valid_df['num_clusters_clean'], valid_df['total_magnetization_clean'])[0, 1]
        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=14, fontweight='bold')
    
    plt.xlabel('Number of Clusters', fontweight='bold', fontsize=20)
    plt.ylabel('Total Magnetization (μB)', fontweight='bold', fontsize=20)
    plt.title('Total Magnetization vs Number of Clusters', fontweight='bold', fontsize=24, pad=20)
    plt.legend(fontsize=14, title_fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_number.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_number.pdf", bbox_inches='tight')
    print(f"Saved magnetization vs cluster number plot to {output_dir}")
    plt.close()

def plot_magnetization_vs_cluster_size(df, output_dir):
    """
    Create scatter plot of magnetization vs maximum cluster size.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization vs cluster size scatter plot...")
    
    # Filter valid data
    valid_df = df.dropna(subset=['total_magnetization_clean', 'max_cluster_size'])
    valid_df = valid_df[valid_df['max_cluster_size'] > 0]
    
    if len(valid_df) == 0:
        print("Warning: No valid data for magnetization vs cluster size plot")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with color coding by magnetization category
    mag_colors = {
        "Non-magnetic (|μ| < 0.1)": '#1f77b4',        # Blue
        "Weakly magnetic (0.1 ≤ |μ| < 1.0)": '#ff7f0e',  # Orange
        "Moderately magnetic (1.0 ≤ |μ| < 5.0)": '#2ca02c',  # Green
        "Strongly magnetic (|μ| ≥ 5.0)": '#d62728'     # Red
    }
    
    for mag_cat in mag_colors.keys():
        mask = valid_df['magnetization_category'] == mag_cat
        if mask.any():
            plt.scatter(valid_df[mask]['max_cluster_size'], 
                       valid_df[mask]['total_magnetization_clean'],
                       c=mag_colors[mag_cat], label=mag_cat, 
                       alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
    
    # Add trend line
    if len(valid_df) > 5:
        z = np.polyfit(valid_df['max_cluster_size'], valid_df['total_magnetization_clean'], 1)
        p = np.poly1d(z)
        plt.plot(valid_df['max_cluster_size'], p(valid_df['max_cluster_size']), 
                "r--", alpha=0.8, linewidth=2, label='Trend line')
        
        # Calculate and display correlation coefficient
        correlation = np.corrcoef(valid_df['max_cluster_size'], valid_df['total_magnetization_clean'])[0, 1]
        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=14, fontweight='bold')
    
    plt.xlabel('Maximum Cluster Size', fontweight='bold', fontsize=20)
    plt.ylabel('Total Magnetization (μB)', fontweight='bold', fontsize=20)
    plt.title('Total Magnetization vs Maximum Cluster Size', fontweight='bold', fontsize=24, pad=20)
    plt.legend(fontsize=14, title_fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_size.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_size.pdf", bbox_inches='tight')
    print(f"Saved magnetization vs cluster size plot to {output_dir}")
    plt.close()

def plot_magnetization_by_point_group(df, output_dir):
    """
    Create violin plot of magnetization distribution by point group, including Oh and Td.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization by point group violin plot...")
    
    # Filter valid data
    valid_df = df.dropna(subset=['total_magnetization_clean', 'main_point_group'])
    valid_df = valid_df[valid_df['main_point_group'] != "Unknown"]
    
    if len(valid_df) == 0:
        print("Warning: No valid data for magnetization by point group plot")
        return
    
    # Get top point groups by frequency, ensuring Oh and Td are included
    all_pg_counts = valid_df['main_point_group'].value_counts()
    
    # Ensure we include Oh and Td if they exist
    important_pgs = ['Oh', 'Td']
    top_n = 6  # Top N most frequent point groups
    
    # First add the specified important point groups if they exist
    point_groups = []
    for pg in important_pgs:
        if pg in all_pg_counts.index:
            point_groups.append(pg)
            print(f"Including specified point group: {pg} (count: {all_pg_counts[pg]})")
    
    # Then add top N most frequent ones, excluding those already added
    for pg in all_pg_counts.index:
        if pg not in point_groups and len(point_groups) < top_n + len([p for p in important_pgs if p in all_pg_counts.index]):
            point_groups.append(pg)
    
    print(f"Selected point groups: {point_groups}")
    
    # Filter to selected point groups
    filtered_df = valid_df[valid_df['main_point_group'].isin(point_groups)]
    
    if len(filtered_df) == 0:
        print("Warning: No data for selected point groups")
        return
    
    plt.figure(figsize=(16, 10))
    
    # Prepare data for violin plot
    magnetization_by_pg = []
    valid_pgs = []
    positions = []
    
    for i, pg in enumerate(point_groups):
        pg_data = filtered_df[filtered_df['main_point_group'] == pg]['total_magnetization_clean'].values
        if len(pg_data) > 0:
            magnetization_by_pg.append(pg_data)
            valid_pgs.append(pg)
            positions.append(i+1)
            print(f"Point group {pg}: {len(pg_data)} data points")
    
    # Create violin plot
    violin_parts = plt.violinplot(
        magnetization_by_pg, 
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.8
    )
    
    # Customize violin appearance using consistent colormap
    for i, pc in enumerate(violin_parts['bodies']):
        # Use magma colormap for consistency with other violin plots
        pc.set_facecolor(plt.cm.magma(i/len(magnetization_by_pg)))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmedians'].set_color('black')
    
    # Add scatter points for individual data points
    for i, pg in enumerate(valid_pgs):
        pg_data = filtered_df[filtered_df['main_point_group'] == pg]['total_magnetization_clean']
        # Use jitter for better visibility
        x = np.random.normal(positions[i], 0.05, size=len(pg_data))
        plt.scatter(x, pg_data, alpha=0.3, s=10, color='black', edgecolor=None)
    
    plt.xlabel('Point Group', fontweight='bold', fontsize=20)
    plt.ylabel('Total Magnetization (μB)', fontweight='bold', fontsize=20)
    plt.title('Total Magnetization Distribution by Point Group', fontweight='bold', fontsize=24, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Set x-ticks to point group names
    plt.xticks(positions, valid_pgs, rotation=45, ha='right', fontsize=16)
    
    # Add statistics for each point group
    stats_text = "Statistics by Point Group:\n"
    for pg in valid_pgs:
        pg_data = filtered_df[filtered_df['main_point_group'] == pg]['total_magnetization_clean']
        stats_text += f"{pg}: mean={pg_data.mean():.2f}, median={pg_data.median():.2f}, n={len(pg_data)}\n"
    
    # Add text box with statistics
    plt.text(1.02, 0.5, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=12, va='center')
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_by_point_group_violin.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_by_point_group_violin.pdf", bbox_inches='tight')
    print(f"Saved magnetization by point group violin plot to {output_dir}")
    plt.close()

def create_magnetization_correlation_heatmap(df, output_dir):
    """
    Create correlation heatmap between magnetization and cluster properties.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization correlation heatmap...")
    
    # Select numerical columns for correlation
    numerical_cols = ['total_magnetization_clean', 'num_clusters_clean', 'max_cluster_size']
    
    # Add additional numerical columns if available
    if 'min_avg_distance' in df.columns:
        df['min_avg_distance_clean'] = pd.to_numeric(df['min_avg_distance'], errors='coerce')
        numerical_cols.append('min_avg_distance_clean')
    
    if 'energy_above_hull' in df.columns:
        df['energy_above_hull_clean'] = pd.to_numeric(df['energy_above_hull'], errors='coerce')
        numerical_cols.append('energy_above_hull_clean')
    
    # Clean data
    corr_df = df[numerical_cols].dropna()
    
    if len(corr_df) < 2:
        print("Warning: Insufficient data for correlation heatmap")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create column labels that are more readable
    label_mapping = {
        'total_magnetization_clean': 'Total Magnetization',
        'num_clusters_clean': 'Num Clusters',
        'max_cluster_size': 'Max Cluster Size',
        'min_avg_distance_clean': 'Min Avg Distance',
        'energy_above_hull_clean': 'Energy Above Hull'
    }
    
    labels = [label_mapping.get(col, col) for col in corr_matrix.columns]
    
    # Set up colormap with divergent colors for correlation
    cmap = plt.cm.RdBu_r
    
    # Create the heatmap
    im = plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=16, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=14)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            plt.text(j, i, f"{value:.3f}", ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=14)
    
    # Set ticks and labels
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=14)
    plt.yticks(range(len(labels)), labels, fontsize=14)
    
    plt.title('Correlation Matrix: Magnetization and Cluster Properties', 
             fontweight='bold', fontsize=20, pad=20)
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_correlation_heatmap.pdf", bbox_inches='tight')
    print(f"Saved magnetization correlation heatmap to {output_dir}")
    plt.close()

def create_magnetization_histogram(df, output_dir):
    """
    Create histogram of magnetization distribution, focusing on the range 0 to 5.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization histogram (0 to 5 range)...")
    
    # Filter valid magnetization data
    valid_df = df.dropna(subset=['total_magnetization_clean'])
    
    if len(valid_df) == 0:
        print("Warning: No valid magnetization data for histogram")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Extract absolute magnetization values and filter to range 0-5
    magnetization_values = valid_df['total_magnetization_clean'].abs()
    filtered_values = magnetization_values[magnetization_values <= 5]
    
    print(f"Total magnetization values: {len(magnetization_values)}")
    print(f"Values in range 0-5: {len(filtered_values)} ({len(filtered_values)/len(magnetization_values)*100:.1f}%)")
    
    # Create histogram
    n, bins, patches = plt.hist(filtered_values, bins=50, alpha=0.7, 
                               color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Color bars based on magnetization categories
    for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
        if bin_edge < 0.1:
            patch.set_facecolor('#1f77b4')  # Non-magnetic
        elif bin_edge < 1.0:
            patch.set_facecolor('#ff7f0e')  # Weakly magnetic
        elif bin_edge < 5.0:
            patch.set_facecolor('#2ca02c')  # Moderately magnetic
    
    plt.xlabel('|Total Magnetization| (μB)', fontweight='bold', fontsize=20)
    plt.ylabel('Number of Materials', fontweight='bold', fontsize=20)
    plt.title('Distribution of Total Magnetization (0-5 μB Range)', fontweight='bold', fontsize=24, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for category boundaries
    plt.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Weak magnetic threshold')
    plt.axvline(x=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Moderate magnetic threshold')
    
    # Add statistics for filtered data
    mean_mag = filtered_values.mean()
    median_mag = filtered_values.median()
    std_mag = filtered_values.std()
    
    plt.text(0.02, 0.98, f'Mean: {mean_mag:.3f} μB\nMedian: {median_mag:.3f} μB\nStd: {std_mag:.3f} μB',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=14, fontweight='bold', va='top')
    
    plt.legend(fontsize=12)
    
    # Set x-axis limits to focus on the 0-5 range
    plt.xlim(0, 5)
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_histogram_0_to_5.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_histogram_0_to_5.pdf", bbox_inches='tight')
    print(f"Saved magnetization histogram (0-5 range) to {output_dir}")
    plt.close()

def plot_magnetization_vs_cluster_number_violin(df, output_dir):
    """
    Create violin plot of magnetization vs cluster number.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization vs cluster number violin plot...")
    
    # Filter valid data
    valid_df = df.dropna(subset=['total_magnetization_clean', 'num_clusters_clean'])
    valid_df = valid_df[valid_df['num_clusters_clean'] > 0]
    
    if len(valid_df) == 0:
        print("Warning: No valid data for magnetization vs cluster number plot")
        return
    
    # Only keep cluster numbers with significant data (more than 30 points)
    cluster_counts = valid_df['num_clusters_limited'].value_counts()
    significant_clusters = cluster_counts[cluster_counts >= 30].index.tolist()
    significant_clusters = sorted(significant_clusters)
    
    # Filter to significant clusters
    significant_df = valid_df[valid_df['num_clusters_limited'].isin(significant_clusters)]
    
    if len(significant_df) == 0:
        print("Warning: No clusters with significant data points (>=30)")
        return
    
    print(f"Using clusters with ≥30 data points: {significant_clusters}")
    print(f"Total data points for violin plot: {len(significant_df)}")
    
    plt.figure(figsize=(14, 10))
    
    # Prepare data for violin plot
    data = []
    positions = []
    
    for i, cluster_num in enumerate(significant_clusters):
        cluster_data = significant_df[significant_df['num_clusters_limited'] == cluster_num]['total_magnetization_clean'].values
        if len(cluster_data) > 0:
            data.append(cluster_data)
            positions.append(i+1)
    
    # Create violin plot
    violin_parts = plt.violinplot(
        data, 
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.8
    )
    
    # Customize violin appearance
    for i, pc in enumerate(violin_parts['bodies']):
        # Use viridis colormap
        pc.set_facecolor(plt.cm.viridis(i/len(data)))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmedians'].set_color('black')
    
    # Add scatter points for individual data points
    for i, cluster_num in enumerate(significant_clusters):
        cluster_data = significant_df[significant_df['num_clusters_limited'] == cluster_num]['total_magnetization_clean'].values
        # Jitter the x-positions for better visibility
        x = np.random.normal(i+1, 0.05, size=len(cluster_data))
        plt.scatter(x, cluster_data, alpha=0.3, s=10, color='black', edgecolor=None)
    
    # Add labels and title
    plt.xlabel('Number of Clusters', fontweight='bold', fontsize=20)
    plt.ylabel('Total Magnetization (μB)', fontweight='bold', fontsize=20)
    plt.title('Total Magnetization vs Number of Clusters', fontweight='bold', fontsize=24, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Set x-ticks to cluster numbers
    plt.xticks(positions, [str(c) for c in significant_clusters], fontsize=16)
    
    # Add statistics annotation
    stats_text = "Statistics by Cluster Number:\n"
    for i, cluster_num in enumerate(significant_clusters):
        cluster_data = significant_df[significant_df['num_clusters_limited'] == cluster_num]['total_magnetization_clean']
        stats_text += f"Cluster {cluster_num}: mean={cluster_data.mean():.2f}, median={cluster_data.median():.2f}, n={len(cluster_data)}\n"
    
    # Add text box with statistics
    plt.text(1.02, 0.5, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=12, va='center')
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_number_violin.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_number_violin.pdf", bbox_inches='tight')
    print(f"Saved magnetization vs cluster number violin plot to {output_dir}")
    plt.close()

def plot_magnetization_vs_cluster_size_violin(df, output_dir):
    """
    Create violin plot of magnetization vs maximum cluster size.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating magnetization vs cluster size violin plot...")
    
    # Filter valid data
    valid_df = df.dropna(subset=['total_magnetization_clean', 'max_cluster_size'])
    valid_df = valid_df[valid_df['max_cluster_size'] > 0]
    
    if len(valid_df) == 0:
        print("Warning: No valid data for magnetization vs cluster size plot")
        return
    
    # Find significant cluster sizes (those with at least 30 data points)
    size_counts = valid_df['size_category'].value_counts()
    significant_sizes = size_counts[size_counts >= 30].index.tolist()
    # Convert to integers where possible for proper sorting
    significant_sizes_int = []
    for size in significant_sizes:
        try:
            if size != "Other":
                significant_sizes_int.append(int(size))
        except ValueError:
            pass
    
    significant_sizes_int = sorted(significant_sizes_int)
    # Convert back to strings
    significant_sizes = [str(s) for s in significant_sizes_int]
    if "Other" in significant_sizes:
        significant_sizes.append("Other")
    
    # Filter to significant sizes
    significant_df = valid_df[valid_df['size_category'].isin(significant_sizes)]
    
    if len(significant_df) == 0:
        print("Warning: No cluster sizes with significant data points (>=30)")
        return
    
    print(f"Using cluster sizes with ≥30 data points: {significant_sizes}")
    print(f"Total data points for violin plot: {len(significant_df)}")
    
    plt.figure(figsize=(14, 10))
    
    # Prepare data for violin plot
    data = []
    positions = []
    
    for i, size in enumerate(significant_sizes):
        size_data = significant_df[significant_df['size_category'] == size]['total_magnetization_clean'].values
        if len(size_data) > 0:
            data.append(size_data)
            positions.append(i+1)
    
    # Create violin plot
    violin_parts = plt.violinplot(
        data, 
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.8
    )
    
    # Customize violin appearance
    for i, pc in enumerate(violin_parts['bodies']):
        # Use plasma colormap for variety
        pc.set_facecolor(plt.cm.plasma(i/len(data)))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmedians'].set_color('black')
    
    # Add scatter points for individual data points
    for i, size in enumerate(significant_sizes):
        size_data = significant_df[significant_df['size_category'] == size]['total_magnetization_clean'].values
        # Jitter the x-positions for better visibility
        x = np.random.normal(i+1, 0.05, size=len(size_data))
        plt.scatter(x, size_data, alpha=0.3, s=10, color='black', edgecolor=None)
    
    # Add labels and title
    plt.xlabel('Maximum Cluster Size', fontweight='bold', fontsize=20)
    plt.ylabel('Total Magnetization (μB)', fontweight='bold', fontsize=20)
    plt.title('Total Magnetization vs Maximum Cluster Size', fontweight='bold', fontsize=24, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Set x-ticks to cluster sizes
    plt.xticks(positions, significant_sizes, fontsize=16)
    
    # Add statistics annotation
    stats_text = "Statistics by Cluster Size:\n"
    for i, size in enumerate(significant_sizes):
        size_data = significant_df[significant_df['size_category'] == size]['total_magnetization_clean']
        stats_text += f"Size {size}: mean={size_data.mean():.2f}, median={size_data.median():.2f}, n={len(size_data)}\n"
    
    # Add text box with statistics
    plt.text(1.02, 0.5, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=12, va='center')
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_size_violin.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/magnetization_vs_cluster_size_violin.pdf", bbox_inches='tight')
    print(f"Saved magnetization vs cluster size violin plot to {output_dir}")
    plt.close()

def main():
    """Main function to run the magnetization-cluster visualizations."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_csv_path = script_dir.parent / "merged_results_all_prop_filtered_v3.csv"
    output_dir = script_dir / "magnetization_visualizations"
    
    print("Magnetization-Cluster Visualization Script")
    print("=" * 50)
    print(f"Data source: {data_csv_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    set_publication_style()
    
    # Load and preprocess data
    df = load_data(data_csv_path)
    if df is None:
        print("Error: Failed to load data. Exiting.")
        return
    
    processed_df = preprocess_magnetization_data(df)
    if processed_df is None:
        print("Error: Failed to preprocess data. Exiting.")
        return
    
    # Create visualizations
    plot_magnetization_distribution(processed_df, output_dir, cmap="viridis")
    plot_magnetization_vs_cluster_number(processed_df, output_dir)
    plot_magnetization_vs_cluster_size(processed_df, output_dir)
    plot_magnetization_by_point_group(processed_df, output_dir)
    create_magnetization_correlation_heatmap(processed_df, output_dir)
    create_magnetization_histogram(processed_df, output_dir)
    plot_magnetization_vs_cluster_number_violin(processed_df, output_dir)
    plot_magnetization_vs_cluster_size_violin(processed_df, output_dir)
    
    print("\nAll magnetization visualizations complete!")
    print("\nGenerated files:")
    print("- magnetization_distribution.png/pdf")
    print("- magnetization_vs_cluster_number.png/pdf")
    print("- magnetization_vs_cluster_size.png/pdf")
    print("- magnetization_by_point_group.png/pdf")
    print("- magnetization_correlation_heatmap.png/pdf")
    print("- magnetization_histogram.png/pdf")
    print("- magnetization_vs_cluster_number_violin.png/pdf")
    print("- magnetization_vs_cluster_size_violin.png/pdf")

if __name__ == "__main__":
    main()