#!/usr/bin/env python
"""
Property Visualizations Script

This script creates publication-quality visualizations of:
1. Space group distribution
2. Predicted dimensionality distribution (0D, 1D, 2D, 3D)
3. Combined property pie chart (Enantiomorphic, Piezoelectric, Polar)

Similar to periodic_table_heatmap.py, this script maintains high visual quality standards.
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

def clean_space_group(space_group):
    """Extract clean space group symbol from the space_group column."""
    if pd.isna(space_group) or not isinstance(space_group, str):
        return "Unknown"
    
    # Try to extract the symbol from strings like "symbol='P6_3/mmc'"
    match = re.search(r"symbol=['\"]([^'\"]+)['\"]", space_group)
    if match:
        return match.group(1)
    
    # If no match found, just return the original string if it's short enough
    if len(space_group) < 15:  # Arbitrary threshold to avoid long strings
        return space_group
    
    return "Unknown"

def preprocess_data(df):
    """
    Preprocess the data for visualization.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pandas.DataFrame: Processed DataFrame ready for visualization
    """
    if df is None or len(df) == 0:
        print("No data to process")
        return None
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in ['Enantiomorphic', 'Piezoelectric', 'Polar']:
        processed_df[col] = processed_df[col].fillna('N')
        
    # Clean up space group information
    processed_df['clean_space_group'] = processed_df['space_group'].apply(clean_space_group)
    
    # Clean up dimensionality - ensure it's in standard format (0D, 1D, 2D, 3D)
    def clean_dimensionality(dim):
        if pd.isna(dim):
            return "Unknown"
        dim_str = str(dim).strip().upper()
        if dim_str in ["0D", "1D", "2D", "3D"]:
            return dim_str
        return "Unknown"
    
    processed_df['clean_dimensionality'] = processed_df['predicted_dimentionality'].apply(clean_dimensionality)
    
    # Create combined property column
    processed_df['combined_properties'] = processed_df.apply(
        lambda row: f"E:{row['Enantiomorphic']},P:{row['Piezoelectric']},Pol:{row['Polar']}", 
        axis=1
    )
    
    print(f"Preprocessing complete: {len(processed_df)} records")
    return processed_df

def plot_space_group_distribution(df, output_dir, top_n=20, cmap="viridis"):
    """
    Create publication-quality visualization of space group distribution.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
        top_n: Number of top space groups to display
        cmap: Colormap to use
    """
    print(f"Creating space group distribution plot (top {top_n})...")
    
    # Count space groups
    space_groups = df['clean_space_group'].value_counts()
    
    # Get top N space groups
    top_space_groups = space_groups.head(top_n)
    
    # Set up figure
    plt.figure(figsize=(16, 12))
    
    # Create colorful bar chart with enhanced styling
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(top_space_groups)))
    bars = plt.bar(range(len(top_space_groups)), top_space_groups.values, color=colors, width=0.7)
    
    # Add count labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*height,
                f'{height}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add space group labels with rotation for readability
    plt.xticks(range(len(top_space_groups)), top_space_groups.index, rotation=45, ha='right', fontsize=16)
    
    # Add descriptive labels and title
    plt.title('Distribution of Space Groups in Clusters', fontweight='bold', pad=20, fontsize=24)
    plt.xlabel('Space Group', fontweight='bold', fontsize=20)
    plt.ylabel('Frequency', fontweight='bold', fontsize=20)
    
    # Enhance grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figures in multiple formats
    plt.tight_layout()
    plt.savefig(f"{output_dir}/space_group_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/space_group_distribution.pdf", bbox_inches='tight')
    print(f"Saved space group distribution plot to {output_dir}")
    plt.close()

def plot_dimensionality_pie(df, output_dir, cmap="plasma"):
    """
    Create publication-quality pie chart of dimensionality distribution.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
        cmap: Colormap to use
    """
    print("Creating dimensionality pie chart...")
    
    # Count dimensionality values
    dim_counts = df['clean_dimensionality'].value_counts()
    
    # Make sure we have all dimension types represented (0D, 1D, 2D, 3D)
    all_dims = ["0D", "1D", "2D", "3D", "Unknown"]
    dim_data = {dim: dim_counts.get(dim, 0) for dim in all_dims}
    
    # Remove "Unknown" if it's 0
    if dim_data["Unknown"] == 0:
        dim_data.pop("Unknown")
    
    # Create a figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Choose visually pleasing colors for dimensionality
    colormap = plt.cm.get_cmap(cmap)
    colors = [colormap(i/4) for i in range(len(dim_data))]
    
    # Create wedge properties
    wedgeprops = {'linewidth': 2, 'edgecolor': 'white'}
    
    # Determine if we should show percentage labels based on slice size
    total_count = sum(dim_data.values())
    
    def autopct_format(pct):
        # Only show percentage if slice is larger than 5%
        return f'{pct:.1f}%' if pct > 5 else ''
    
    # Create the pie chart with enhancements
    wedges, texts, autotexts = ax.pie(
        list(dim_data.values()),
        labels=None,
        autopct=autopct_format,
        startangle=90,
        wedgeprops=wedgeprops,
        colors=colors,
        textprops={'fontsize': 24, 'fontweight': 'bold'},
        pctdistance=0.85
    )
    
    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(24)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add title with enhanced styling
    plt.title('Distribution of Predicted Dimensionality', fontweight='bold', fontsize=28, pad=20)
    
    # Create legend with counts and percentages - improve spacing and positioning
    legend_labels = []
    for dim, count in dim_data.items():
        percentage = (count / total_count) * 100
        legend_labels.append(f"{dim}: {count} ({percentage:.1f}%)")
    
    plt.legend(wedges, legend_labels, title="Dimensionality", 
              loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1),
              fontsize=24, title_fontsize=28)
    
    # Add annotation for total count
    plt.annotate(f'Total: {total_count}',
                xy=(-0.12, -0.12), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="0.9", ec="0.5", alpha=0.8),
                fontsize=14, ha='left', va='center')
    
    # Save figures in multiple formats with extra space for legend
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dimensionality_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/dimensionality_distribution.pdf", bbox_inches='tight')
    print(f"Saved dimensionality distribution plot to {output_dir}")
    plt.close()

def plot_combined_properties_pie(df, output_dir, cmap="viridis"):
    """
    Create publication-quality pie chart of combined property distribution
    (Enantiomorphic, Piezoelectric, Polar).
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
        cmap: Colormap to use
    """
    print("Creating combined properties pie chart...")
    
    # Create property combinations and count
    property_combinations = defaultdict(int)
    
    for _, row in df.iterrows():
        enantio = 'E' if row['Enantiomorphic'] == 'Y' else ''
        piezo = 'P' if row['Piezoelectric'] == 'Y' else ''
        polar = 'Pol' if row['Polar'] == 'Y' else ''
        
        properties = []
        if enantio: properties.append(enantio)
        if piezo: properties.append(piezo)
        if polar: properties.append(polar)
        
        if not properties:
            key = "None"
        else:
            key = "+".join(properties)
        
        property_combinations[key] += 1
    
    # Set up the figure with more room for legend
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Prepare data for pie chart
    labels = list(property_combinations.keys())
    sizes = list(property_combinations.values())
    total_count = sum(sizes)
    
    # Sorting to ensure consistent ordering (larger slices first)
    sorted_data = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*sorted_data) if sorted_data else ([], [])
    
    # Choose colors from colormap
    colormap = plt.cm.get_cmap(cmap)
    colors = [colormap(i/len(labels)) for i in range(len(labels))]
    
    # Create explode array to emphasize important combinations
    explode = [0.05 if 'P' in label and 'Pol' in label else 0.0 for label in labels]
    
    # Create wedge properties
    wedgeprops = {'linewidth': 2, 'edgecolor': 'white'}
    
    # Function to conditionally show percentages only for larger slices
    def autopct_format(pct):
        # Only show percentage if slice is larger than 5%
        return f'{pct:.1f}%' if pct > 5 else ''
    
    # Create the pie chart with enhancements
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=autopct_format,
        startangle=90,
        wedgeprops=wedgeprops,
        colors=colors,
        textprops={'fontsize': 18, 'fontweight': 'bold'},
        pctdistance=0.85
    )
    
    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(20)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add title with enhanced styling
    plt.title('Distribution of Material Properties\n(Enantiomorphic, Piezoelectric, Polar)', 
             fontweight='bold', fontsize=20, pad=20)
    
    # Create legend with counts and descriptions - fix spacing issue
    legend_labels = []
    for label, size in zip(labels, sizes):
        percentage = (size / total_count) * 100
        if label == "None":
            legend_labels.append(f"None: {size} ({percentage:.1f}%)")
        else:
            # Create more descriptive label
            desc_parts = []
            if 'E' in label: desc_parts.append("Enantiomorphic")
            if 'P' in label: desc_parts.append("Piezoelectric") 
            if 'Pol' in label: desc_parts.append("Polar")
            desc = " + ".join(desc_parts)
            legend_labels.append(f"{desc}: {size} ({percentage:.1f}%)")
    
    # Move legend to the right with improved spacing
    plt.legend(wedges, legend_labels, title="Material Properties", 
              loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1),
              fontsize=18, title_fontsize=20)
    
    # Add annotation for total count
    plt.annotate(f'Total: {total_count}',
                xy=(-0.12, -0.12), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="0.9", ec="0.5", alpha=0.8),
                fontsize=18, ha='left', va='center')
    
    # Add a note explaining the properties - positioned below the chart
    note = ("E = Enantiomorphic (non-superimposable mirror images)\n"
           "P = Piezoelectric (generates electric charge under stress)\n"
           "Pol = Polar (has permanent electric dipole moment)")
    plt.annotate(note, xy=(0.5, -0.18), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.5", alpha=0.9),
                fontsize=18, ha='center', va='top')
    
    # Save figures in multiple formats with more space for annotations
    plt.tight_layout()
    plt.savefig(f"{output_dir}/property_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/property_distribution.pdf", bbox_inches='tight')
    print(f"Saved property distribution plot to {output_dir}")
    plt.close()

def create_cluster_size_point_group_property_correlation(df, output_dir):
    """
    Create a visualization showing correlation between cluster size, point groups,
    and material properties.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating cluster size-point group-property correlation plot...")
    
    # Convert Y/N to 1/0 for properties
    for col in ['Enantiomorphic', 'Piezoelectric', 'Polar']:
        df[f'{col}_binary'] = df[col].map({'Y': 1, 'N': 0})
    
    # Extract point group information if available
    has_point_group_data = 'point_groups' in df.columns
    if has_point_group_data:
        # Clean up and extract main point group
        def extract_main_point_group(pg_str):
            if pd.isna(pg_str) or not isinstance(pg_str, str):
                return "Unknown"
            
            # Try to extract point groups from format like {'X0': 'C*v', 'X1': 'D*h'}
            try:
                if '{' in pg_str:
                    pg_dict = ast.literal_eval(pg_str.replace("'", '"'))
                    if isinstance(pg_dict, dict) and pg_dict:
                        # Return the first point group found
                        return next(iter(pg_dict.values()))
                return pg_str
            except:
                return "Unknown"
        
        df['main_point_group'] = df['point_groups'].apply(extract_main_point_group)
        
        # Get top point groups
        top_point_groups = df['main_point_group'].value_counts().head(5).index.tolist()
    else:
        # If no point group data, create a dummy column
        df['main_point_group'] = "Unknown"
        top_point_groups = ["Unknown"]
    
    # Extract cluster size information if available
    has_cluster_size_data = 'cluster_sizes' in df.columns
    if has_cluster_size_data:
        # Process cluster sizes (usually stored as list)
        def extract_max_cluster_size(size_str):
            if pd.isna(size_str):
                return 0
            try:
                if isinstance(size_str, str) and '[' in size_str:
                    # Parse as list
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
        
        df['max_cluster_size'] = df['cluster_sizes'].apply(extract_max_cluster_size)
        
        # Create explicit size categories for all observed sizes
        unique_sizes = sorted(df['max_cluster_size'].unique())
        # Remove 0 if present and limit to reasonable range
        unique_sizes = [s for s in unique_sizes if s > 0 and s <= 20]
        size_categories = [str(int(s)) for s in unique_sizes[:10]]  # Limit to top 10 sizes
        
        df['size_category'] = df['max_cluster_size'].apply(lambda x: str(int(x)) if x in unique_sizes[:10] else "Other")
        
        # Add "Other" category if there are sizes beyond the top 10
        if len(unique_sizes) > 10:
            size_categories.append("Other")
    else:
        # If no cluster size data, create dummy columns
        df['max_cluster_size'] = 0
        df['size_category'] = "Unknown"
        size_categories = ["Unknown"]
    
    # Limit cluster numbers to 10 and create explicit categories
    df['num_clusters_clean'] = pd.to_numeric(df['num_clusters'], errors='coerce').fillna(0)
    df['num_clusters_limited'] = df['num_clusters_clean'].apply(lambda x: min(int(x), 10) if x > 0 else 0)
    
    # Create explicit cluster number categories (1 to 10)
    cluster_num_categories = [str(i) for i in range(1, 11)]
    df['cluster_num_category'] = df['num_clusters_limited'].apply(lambda x: str(int(x)) if 1 <= x <= 10 else "0")
    
    # Filter out materials with 0 clusters for meaningful analysis
    df_filtered = df[df['num_clusters_limited'] > 0].copy()
    
    # Create matrices for heatmaps
    property_categories = ["Enantiomorphic", "Piezoelectric", "Polar"]
    
    # Matrix for size vs property (normalized by row)
    size_property_matrix = np.zeros((len(size_categories), len(property_categories)))
    size_property_counts = np.zeros((len(size_categories), len(property_categories)))
    
    # Matrix for point group vs property (normalized by row)
    pg_property_matrix = np.zeros((len(top_point_groups), len(property_categories)))
    pg_property_counts = np.zeros((len(top_point_groups), len(property_categories)))
    
    # Matrix for cluster number vs property (normalized by row)
    cluster_num_property_matrix = np.zeros((len(cluster_num_categories), len(property_categories)))
    cluster_num_property_counts = np.zeros((len(cluster_num_categories), len(property_categories)))
    
    # Matrix for size vs point group (normalized by row)
    size_pg_matrix = np.zeros((len(size_categories), len(top_point_groups)))
    
    # Fill size vs property matrix with normalization
    for i, size_cat in enumerate(size_categories):
        size_df = df_filtered[df_filtered['size_category'] == size_cat]
        if len(size_df) == 0:
            continue
            
        for j, prop in enumerate(property_categories):
            # Count structures with this property and size
            count = size_df[f'{prop}_binary'].sum()
            total = len(size_df)
            
            # Calculate percentage (normalized by category)
            if total > 0:
                size_property_matrix[i, j] = count / total * 100
                size_property_counts[i, j] = count
    
    # Fill cluster number vs property matrix with normalization
    for i, cluster_num in enumerate(cluster_num_categories):
        cluster_df = df_filtered[df_filtered['cluster_num_category'] == cluster_num]
        if len(cluster_df) == 0:
            continue
            
        for j, prop in enumerate(property_categories):
            # Count structures with this property and cluster number
            count = cluster_df[f'{prop}_binary'].sum()
            total = len(cluster_df)
            
            # Calculate percentage (normalized by category)
            if total > 0:
                cluster_num_property_matrix[i, j] = count / total * 100
                cluster_num_property_counts[i, j] = count
    
    # Fill size vs point group matrix with normalization
    for i, size_cat in enumerate(size_categories):
        size_df = df_filtered[df_filtered['size_category'] == size_cat]
        if len(size_df) == 0:
            continue
            
        for k, pg in enumerate(top_point_groups):
            pg_count = len(size_df[size_df['main_point_group'] == pg])
            total = len(size_df)
            if total > 0:
                size_pg_matrix[i, k] = pg_count / total * 100
    
    # Fill point group vs property matrix with normalization
    for i, pg in enumerate(top_point_groups):
        pg_df = df_filtered[df_filtered['main_point_group'] == pg]
        if len(pg_df) == 0:
            continue
            
        for j, prop in enumerate(property_categories):
            # Count structures with this property and point group
            count = pg_df[f'{prop}_binary'].sum()
            total = len(pg_df)
            
            # Calculate percentage (normalized by category)
            if total > 0:
                pg_property_matrix[i, j] = count / total * 100
                pg_property_counts[i, j] = count
    
    # Create the visualization - 2x2 grid of heatmaps
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Cluster Size vs Property Matrix (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    create_heatmap(ax1, size_property_matrix, size_categories, property_categories,
                 'Max Cluster Size vs Material Properties\n(Normalized by Size Category)', 
                 'Material Properties', 'Max Cluster Size')
    
    # 2. Point Group vs Property Matrix (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    create_heatmap(ax2, pg_property_matrix, top_point_groups, property_categories,
                 'Point Group vs Material Properties\n(Normalized by Point Group)', 
                 'Material Properties', 'Point Group')
    
    # 3. Cluster Number vs Property Matrix (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    create_heatmap(ax3, cluster_num_property_matrix, cluster_num_categories, property_categories,
                 'Number of Clusters vs Material Properties\n(Normalized by Cluster Count)', 
                 'Material Properties', 'Number of Clusters')
    
    # 4. Size vs Point Group Matrix (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    create_heatmap(ax4, size_pg_matrix, size_categories, top_point_groups,
                 'Max Cluster Size vs Point Group\n(Normalized by Size Category)', 
                 'Point Group', 'Max Cluster Size')
    
    # Add overall title
    fig.suptitle('Normalized Correlation Analysis: Cluster Properties and Material Properties',
               fontsize=24, fontweight='bold', y=0.98)
    
    # Add explanatory note
    note = ("Note: All values are normalized by row (category). Each percentage represents the fraction of materials\n"
           "within a specific category that exhibit the corresponding property or characteristic.")
    fig.text(0.5, 0.01, note, ha='center', fontsize=14, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save with high resolution
    plt.savefig(f"{output_dir}/structure_property_correlation.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/structure_property_correlation.pdf", bbox_inches='tight')
    print(f"Saved structure-property correlation matrix to {output_dir}")
    plt.close()

def create_heatmap(ax, data_matrix, row_labels, col_labels, title, xlabel, ylabel, count_matrix=None):
    """Helper function to create consistent heatmaps with professional formatting"""
    # Use viridis colormap for consistency with periodic_table_heatmap.py
    cmap = plt.cm.viridis
    
    # Set up normalization - handle empty matrix
    if np.max(data_matrix) > 0:
        vmin = 0
        vmax = min(100, np.max(data_matrix))  # Cap at 100% for percentage data
    else:
        vmin = 0
        vmax = 100
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create the heatmap with white grid lines for professional appearance
    im = ax.imshow(data_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add colorbar with professional styling
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Percentage (%)', fontsize=18, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    
    # Set ticks and labels with professional formatting
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=16, fontweight='bold')
    ax.set_yticklabels(row_labels, fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Remove subgrid
    ax.grid(False)
    
    # Add text annotations with contrast-aware colors and professional formatting
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            percentage = data_matrix[i, j]
            
            # Skip text for very small values
            if percentage < 1.0:
                continue
                
            # Choose text color based on background intensity for better contrast
            text_color = 'white' 
            
            # Add just percentage with professional formatting (no counts)
            text = f"{percentage:.1f}%"
            
            # Use path effects to create an outline for better visibility
            text_obj = ax.text(j, i, text, 
                   ha="center", va="center", 
                   color=text_color,
                   fontweight="bold", 
                   fontsize=20)
                   
            # Add outline effect for better visibility
            text_obj.set_path_effects([
                path_effects.withStroke(linewidth=3, foreground='black')
            ])
    
    # Add title and labels with professional styling
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold', labelpad=10)
    
    # Remove outer spines for cleaner appearance
    for spine in ax.spines.values():
        spine.set_visible(False)

def create_stability_analysis(df, output_dir):
    """
    Create comprehensive visualizations showing the relationship between material stability
    (energy_above_hull) and various cluster properties.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save the output
    """
    print("Creating material stability analysis...")
    
    # Clean and prepare stability data
    df['energy_above_hull_clean'] = pd.to_numeric(df['energy_above_hull'], errors='coerce')
    
    # Remove materials with missing stability data
    stable_df = df.dropna(subset=['energy_above_hull_clean']).copy()
    
    if len(stable_df) == 0:
        print("Warning: No materials with energy_above_hull data found")
        return
    
    # Define stability categories
    def categorize_stability(energy):
        if energy <= 0.025:  # Very stable
            return "Very Stable (≤0.025 eV/atom)"
        elif energy <= 0.1:   # Stable
            return "Stable (0.025-0.1 eV/atom)"
        elif energy <= 0.3:   # Metastable
            return "Metastable (0.1-0.3 eV/atom)"
        else:                 # Unstable
            return "Unstable (>0.3 eV/atom)"
    
    stable_df['stability_category'] = stable_df['energy_above_hull_clean'].apply(categorize_stability)
    
    # Extract cluster properties for analysis
    stable_df['num_clusters_clean'] = pd.to_numeric(stable_df['num_clusters'], errors='coerce')
    
    # Extract max cluster size
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
    
    stable_df['max_cluster_size'] = stable_df['cluster_sizes'].apply(extract_max_cluster_size)
    
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
    
    if 'point_groups' in stable_df.columns:
        stable_df['main_point_group'] = stable_df['point_groups'].apply(extract_main_point_group)
    else:
        stable_df['main_point_group'] = "Unknown"
    
    # Clean min_avg_distance
    stable_df['min_avg_distance_clean'] = pd.to_numeric(stable_df['min_avg_distance'], errors='coerce')
    
    # Limit cluster numbers to 10 and create explicit categories
    stable_df['num_clusters_limited'] = stable_df['num_clusters_clean'].apply(lambda x: min(int(x), 10) if x > 0 else 0)
    stable_df['cluster_num_category'] = stable_df['num_clusters_limited'].apply(lambda x: str(int(x)) if 1 <= x <= 10 else "0")
    
    # Create explicit size categories for all observed sizes
    unique_sizes = sorted(stable_df['max_cluster_size'].unique())
    unique_sizes = [s for s in unique_sizes if s > 0 and s <= 20]
    size_categories = [str(int(s)) for s in unique_sizes[:10]]  # Limit to top 10 sizes
    stable_df['size_category'] = stable_df['max_cluster_size'].apply(lambda x: str(int(x)) if x in unique_sizes[:10] else "Other")
    if len(unique_sizes) > 10:
        size_categories.append("Other")
    
    # Filter out materials with 0 clusters for meaningful analysis
    stable_df_filtered = stable_df[stable_df['num_clusters_limited'] > 0].copy()
    
    # Create separate figures for each visualization
    
    # 1. Energy vs Number of Clusters
    plt.figure(figsize=(12, 10))
    create_stability_scatter(plt.gca(), stable_df_filtered, 'num_clusters_clean', 'energy_above_hull_clean',
                         'Number of Clusters vs Energy Above Hull',
                         'Number of Clusters', 'Energy Above Hull (eV/atom)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_vs_cluster_number.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_vs_cluster_number.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. Energy vs Max Cluster Size
    plt.figure(figsize=(12, 10))
    create_stability_scatter(plt.gca(), stable_df_filtered, 'max_cluster_size', 'energy_above_hull_clean',
                         'Max Cluster Size vs Energy Above Hull',
                         'Max Cluster Size', 'Energy Above Hull (eV/atom)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_vs_cluster_size.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_vs_cluster_size.pdf", bbox_inches='tight')
    plt.close()
    
    # 3. Energy vs Min Average Distance
    plt.figure(figsize=(12, 10))
    create_stability_scatter(plt.gca(), stable_df_filtered, 'min_avg_distance_clean', 'energy_above_hull_clean',
                         'Min Average Distance vs Energy Above Hull',
                         'Min Average Distance (Å)', 'Energy Above Hull (eV/atom)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_vs_min_distance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_vs_min_distance.pdf", bbox_inches='tight')
    plt.close()
    
    # 4. Normalized stability distribution by number of clusters
    plt.figure(figsize=(14, 10))
    cluster_num_categories = [str(i) for i in range(1, 11)]
    create_normalized_stability_distribution(plt.gca(), stable_df_filtered, 'cluster_num_category', cluster_num_categories,
                              'Stability Distribution by Number of Clusters\n(Normalized by Cluster Count)',
                              'Number of Clusters')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_distribution_by_cluster_number.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_distribution_by_cluster_number.pdf", bbox_inches='tight')
    plt.close()
    
    # 5. Normalized stability distribution by cluster size
    plt.figure(figsize=(14, 10))
    create_normalized_stability_distribution(plt.gca(), stable_df_filtered, 'size_category', size_categories,
                              'Stability Distribution by Cluster Size\n(Normalized by Size Category)',
                              'Cluster Size')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_distribution_by_cluster_size.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_distribution_by_cluster_size.pdf", bbox_inches='tight')
    plt.close()
    
    # 6. Normalized stability distribution by point group
    plt.figure(figsize=(14, 10))
    # Get top 5 point groups for cleaner visualization
    top_point_groups = stable_df_filtered['main_point_group'].value_counts().head(5).index.tolist()
    stable_df_top_pg = stable_df_filtered[stable_df_filtered['main_point_group'].isin(top_point_groups)]
    create_normalized_stability_distribution(plt.gca(), stable_df_top_pg, 'main_point_group', top_point_groups,
                              'Stability Distribution by Point Group\n(Normalized by Point Group)',
                              'Point Group')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_distribution_by_point_group.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_distribution_by_point_group.pdf", bbox_inches='tight')
    plt.close()
    
    # 7. Correlation heatmap
    plt.figure(figsize=(12, 10))
    create_stability_correlation_heatmap(plt.gca(), stable_df_filtered)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_correlation_matrix.pdf", bbox_inches='tight')
    plt.close()
    
    # 8. Energy distribution histogram
    plt.figure(figsize=(12, 10))
    create_energy_distribution_histogram(plt.gca(), stable_df_filtered)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_distribution_histogram.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/energy_distribution_histogram.pdf", bbox_inches='tight')
    plt.close()
    
    # 9. Normalized stability vs Dimensionality
    plt.figure(figsize=(14, 10))
    dim_categories = ["0D", "1D", "2D", "3D"]
    create_normalized_stability_distribution(plt.gca(), stable_df_filtered, 'clean_dimensionality', dim_categories,
                              'Stability Distribution by Dimensionality\n(Normalized by Dimensionality)',
                              'Predicted Dimensionality')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_distribution_by_dimensionality.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/stability_distribution_by_dimensionality.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual stability analysis plots to {output_dir}")
    plt.close()

def create_stability_scatter(ax, df, x_col, y_col, title, xlabel, ylabel):
    """Helper function to create scatter plots for stability analysis"""
    # Remove NaN values
    clean_df = df.dropna(subset=[x_col, y_col])
    
    if len(clean_df) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return
    
    x_data = clean_df[x_col]
    y_data = clean_df[y_col]
    
    # Create scatter plot with color coding by stability
    stability_colors = {
        "Very Stable (≤0.025 eV/atom)": '#2E8B57',      # Sea green
        "Stable (0.025-0.1 eV/atom)": '#32CD32',        # Lime green  
        "Metastable (0.1-0.3 eV/atom)": '#FFD700',      # Gold
        "Unstable (>0.3 eV/atom)": '#DC143C'            # Crimson
    }
    
    for stability_cat in stability_colors.keys():
        mask = clean_df['stability_category'] == stability_cat
        if mask.any():
            ax.scatter(x_data[mask], y_data[mask], 
                      c=stability_colors[stability_cat], 
                      label=stability_cat, alpha=0.7, s=50)
    
    # Add trend line
    if len(clean_df) > 5:  # Only if we have enough points
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
        
        # Calculate and display correlation coefficient
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add stability threshold line
    ax.axhline(y=0.025, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Stability threshold')

def create_normalized_stability_distribution(ax, df, group_col, categories, title, xlabel):
    """Helper function to create normalized stability distribution plots"""
    # Remove NaN values
    clean_df = df.dropna(subset=[group_col, 'energy_above_hull_clean'])
    
    if len(clean_df) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title(title, fontsize=18, fontweight='bold')
        return
    
    # Filter to only include categories that exist in the data
    existing_categories = [cat for cat in categories if cat in clean_df[group_col].values]
    
    if not existing_categories:
        ax.text(0.5, 0.5, 'No matching categories', ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title(title, fontsize=18, fontweight='bold')
        return
    
    stability_categories = ["Very Stable (≤0.025 eV/atom)", "Stable (0.025-0.1 eV/atom)", 
                          "Metastable (0.1-0.3 eV/atom)", "Unstable (>0.3 eV/atom)"]
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#DC143C']
    
    # Calculate normalized percentages for each group
    percentages = np.zeros((len(existing_categories), len(stability_categories)))
    counts = np.zeros((len(existing_categories), len(stability_categories)))
    
    for i, category in enumerate(existing_categories):
        cat_df = clean_df[clean_df[group_col] == category]
        total = len(cat_df)
        
        if total == 0:
            continue
            
        for j, stability_cat in enumerate(stability_categories):
            count = len(cat_df[cat_df['stability_category'] == stability_cat])
            percentages[i, j] = (count / total * 100)  # Normalized by category
            counts[i, j] = count
    
    # Create stacked bar chart
    bottom = np.zeros(len(existing_categories))
    bars = []
    
    for j, (stability_cat, color) in enumerate(zip(stability_categories, colors)):
        bars.append(ax.bar(range(len(existing_categories)), percentages[:, j], bottom=bottom, 
                          color=color, label=stability_cat, alpha=0.9, edgecolor='black', linewidth=0.5))
        bottom += percentages[:, j]
    
    # Add count annotations
    for i, category in enumerate(existing_categories):
        y_pos = 0
        total_count = int(counts[i, :].sum())
        
        # Add total count at the top of each bar
        ax.text(i, 102, f'n={total_count}', ha='center', va='bottom',
               fontweight='bold', fontsize=14, color='black')
        
        for j in range(len(stability_categories)):
            count = int(counts[i, j])
            if count > 0 and percentages[i, j] > 5:  # Only show label if slice is large enough
                y_center = y_pos + percentages[i, j] / 2
                text_obj = ax.text(i, y_center, str(count), ha='center', va='center',
                       fontweight='bold', fontsize=14, color='white')
                
                # Add outline effect for better visibility
                text_obj.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground='black')
                ])
                
            y_pos += percentages[i, j]
    
    # Set x-axis labels
    ax.set_xticks(range(len(existing_categories)))
    ax.set_xticklabels(existing_categories, fontsize=14, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylim(0, 110)  # Extra space for count labels
    
    # Rotate x-axis labels if needed
    if len(existing_categories) > 0 and len(str(existing_categories[0])) > 3:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Always add a legend with consistent positioning
    ax.legend(fontsize=14, title_fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')

def create_stability_distribution(ax, df, group_col, title, xlabel):
    """Helper function to create stability distribution plots"""
    # Remove NaN values
    clean_df = df.dropna(subset=[group_col, 'energy_above_hull_clean'])
    
    if len(clean_df) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return
    
    # Get unique groups and their stability distributions
    groups = clean_df[group_col].unique()
    groups = sorted([g for g in groups if pd.notna(g)])
    
    stability_categories = ["Very Stable (≤0.025 eV/atom)", "Stable (0.025-0.1 eV/atom)", 
                          "Metastable (0.1-0.3 eV/atom)", "Unstable (>0.3 eV/atom)"]
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#DC143C']
    
    # Calculate percentages for each group
    percentages = np.zeros((len(groups), len(stability_categories)))
    counts = np.zeros((len(groups), len(stability_categories)))
    
    for i, group in enumerate(groups):
        group_df = clean_df[clean_df[group_col] == group]
        total = len(group_df)
        
        for j, stability_cat in enumerate(stability_categories):
            count = len(group_df[group_df['stability_category'] == stability_cat])
            percentages[i, j] = (count / total * 100) if total > 0 else 0
            counts[i, j] = count
    
    # Create stacked bar chart
    bottom = np.zeros(len(groups))
    bars = []
    
    for j, (stability_cat, color) in enumerate(zip(stability_categories, colors)):
        bars.append(ax.bar(groups, percentages[:, j], bottom=bottom, 
                          color=color, label=stability_cat, alpha=0.8))
        bottom += percentages[:, j]
    
    # Add count annotations
    for i, group in enumerate(groups):
        y_pos = 0
        for j in range(len(stability_categories)):
            count = int(counts[i, j])
            if count > 0:
                y_center = y_pos + percentages[i, j] / 2
                ax.text(i, y_center, str(count), ha='center', va='center',
                       fontweight='bold', fontsize=10)
            y_pos += percentages[i, j]
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Rotate x-axis labels if needed
    if len(str(groups[0])) > 3:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def create_stability_correlation_heatmap(ax, df):
    """Create correlation heatmap between stability and numerical properties with professional formatting"""
    # Select numerical columns for correlation
    numerical_cols = ['energy_above_hull_clean', 'num_clusters_clean', 'max_cluster_size', 
                     'min_avg_distance_clean']
    
    # Clean data
    corr_df = df[numerical_cols].dropna()
    
    if len(corr_df) < 2:
        ax.text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        return
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create column labels that are more readable
    labels = ['Energy Above Hull', 'Num Clusters', 'Max Cluster Size', 'Min Avg Distance']
    
    # Set up colormap with divergent colors for correlation (RdBu_r)
    cmap = plt.cm.RdBu_r
    
    # Set up normalization from -1 to 1 for correlation values
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    # Turn off default grid
    ax.grid(False)
    
    # Use pcolormesh for precise grid alignment
    data_matrix = corr_matrix.values
    x = np.arange(len(labels) + 1)
    y = np.arange(len(labels) + 1)
    mesh = ax.pcolormesh(x, y, data_matrix, cmap=cmap, norm=norm)
    
    # Add explicit grid lines that align perfectly with cell boundaries
    for i in range(len(labels) + 1):
        ax.axhline(y=i, color='white', linewidth=1.5, alpha=0.8)
    for j in range(len(labels) + 1):
        ax.axvline(x=j, color='white', linewidth=1.5, alpha=0.8)
    
    # Add annotations with improved contrast
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = data_matrix[i, j]
            
            # Format the text according to the provided format string
            text = f"{value:.3f}"
            
            # Use white text for dark backgrounds, black for light backgrounds
            if abs(value) > 0.5:
                text_color = 'white'
                outline_color = 'black'
            else:
                text_color = 'black'
                outline_color = 'white'
            
            # Add text with outline for better visibility - centered in cells
            ax.text(j + 0.5, i + 0.5, text, 
                   ha='center', va='center', 
                   color=text_color, 
                   fontweight='bold', 
                   fontsize=14,
                   path_effects=[plt.matplotlib.patheffects.withStroke(
                       linewidth=3, foreground=outline_color)])
    
    # Customize plot appearance
    ax.set_title('Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
    
    # Center the x-tick labels in the cells
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    
    # Center the y-tick labels in the cells
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=12)
    
    # Add colorbar with consistent styling
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=14, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Remove spines for cleaner appearance
    for spine in ax.spines.values():
        spine.set_visible(False)

def create_energy_distribution_histogram(ax, df):
    """Create histogram of energy above hull distribution with improved text positioning"""
    energies = df['energy_above_hull_clean'].dropna()
    
    if len(energies) == 0:
        ax.text(0.5, 0.5, 'No energy data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Energy Above Hull Distribution', fontsize=14, fontweight='bold')
        return
    
    # Create histogram with log scale for better visualization
    bins = np.logspace(np.log10(max(energies.min(), 0.001)), np.log10(energies.max()), 30)
    
    n, bins, patches = ax.hist(energies, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color bars based on stability
    for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
        if bin_edge <= 0.025:
            patch.set_facecolor('#2E8B57')  # Very stable
        elif bin_edge <= 0.1:
            patch.set_facecolor('#32CD32')  # Stable
        elif bin_edge <= 0.3:
            patch.set_facecolor('#FFD700')  # Metastable
        else:
            patch.set_facecolor('#DC143C')  # Unstable
    
    ax.set_xscale('log')
    ax.set_xlabel('Energy Above Hull (eV/atom)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Materials', fontsize=12, fontweight='bold')
    ax.set_title('Energy Above Hull Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add stability threshold lines
    ax.axvline(x=0.025, color='green', linestyle='--', linewidth=2, label='Very stable threshold')
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Stable threshold')
    ax.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Metastable threshold')
    
    # Add statistics with improved positioning to avoid covering objects
    mean_energy = energies.mean()
    median_energy = energies.median()
    
    # Position text box in upper left corner to avoid covering histogram bars
    ax.text(0.02, 0.98, f'Mean: {mean_energy:.3f} eV/atom\nMedian: {median_energy:.3f} eV/atom',
            transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=11, fontweight='bold', verticalalignment='top', horizontalalignment='left')

def create_separate_correlation_plots(df, output_dir):
    """
    Create individual correlation plots for various material properties.
    
    Args:
        df: Processed DataFrame with property data
        output_dir: Directory to save the output plots
    """
    print("Creating detailed property correlation plots...")
    
    # Extract and clean relevant properties
    properties = ['Enantiomorphic', 'Piezoelectric', 'Polar']
    
    # Convert Y/N to 1/0 for binary properties
    for prop in properties:
        df[f'{prop}_binary'] = df[prop].map({'Y': 1, 'N': 0})
    
    # Ensure numerical values for key metrics
    df['num_clusters_clean'] = pd.to_numeric(df['num_clusters'], errors='coerce')
    df['min_avg_distance_clean'] = pd.to_numeric(df['min_avg_distance'], errors='coerce')
    df['energy_above_hull_clean'] = pd.to_numeric(df['energy_above_hull'], errors='coerce')
    
    # Extract max cluster size from cluster_sizes
    def extract_max_cluster_size(size_str):
        if pd.isna(size_str):
            return 0
        try:
            if isinstance(size_str, str) and '[' in size_str:
                sizes = ast.literal_eval(size_str)
                return max(sizes) if sizes else 0
            elif isinstance(size_str, list):
                return max(size_str) if size_str else 0
            else:
                return float(size_str)
        except:
            return 0
    
    df['max_cluster_size'] = df['cluster_sizes'].apply(extract_max_cluster_size)
    
    # Clean data by removing rows with missing values in critical columns
    plot_df = df.dropna(subset=['num_clusters_clean', 'max_cluster_size']).copy()
    
    # Features to correlate with properties
    features = {
        'num_clusters_clean': 'Number of Clusters',
        'max_cluster_size': 'Maximum Cluster Size',
        'min_avg_distance_clean': 'Minimum Average Distance',
        'energy_above_hull_clean': 'Energy Above Hull (eV/atom)'
    }
    
    # Create individual correlation plots
    for prop in properties:
        binary_col = f'{prop}_binary'
        
        # Create a figure with subplots for each feature
        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        axes = axes.flatten()
        
        # Plot each feature correlation
        for i, (feature_col, feature_name) in enumerate(features.items()):
            ax = axes[i]
            
            # Filter valid data
            valid_df = plot_df.dropna(subset=[feature_col, binary_col])
            
            if len(valid_df) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                continue
            
            # Create violin plots for each category
            violin_parts = ax.violinplot(
                [valid_df[valid_df[binary_col] == 0][feature_col], 
                 valid_df[valid_df[binary_col] == 1][feature_col]],
                showmeans=True,
                showmedians=False
            )
            
            # Color the violin plots
            for pc in violin_parts['bodies']:
                pc.set_facecolor('#3498db')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            # Customize appearance
            violin_parts['cmeans'].set_color('red')
            
            # Add scatter points for better data visibility
            for j, binary_val in enumerate([0, 1]):
                subset = valid_df[valid_df[binary_col] == binary_val]
                ax.scatter(
                    [j + 1] * len(subset), 
                    subset[feature_col],
                    alpha=0.3, 
                    s=30, 
                    color='#2c3e50',
                    edgecolor='white'
                )
            
            # Calculate statistics
            non_property_values = valid_df[valid_df[binary_col] == 0][feature_col]
            property_values = valid_df[valid_df[binary_col] == 1][feature_col]
            
            non_property_mean = non_property_values.mean()
            property_mean = property_values.mean()
            
            # Add mean values and difference annotation
            mean_diff = property_mean - non_property_mean
            percent_diff = (mean_diff / non_property_mean) * 100 if non_property_mean != 0 else float('inf')
            
            ax.text(0.5, 0.95, 
                    f"Mean (No {prop}): {non_property_mean:.3f}\n"
                    f"Mean (With {prop}): {property_mean:.3f}\n"
                    f"Difference: {mean_diff:.3f} ({percent_diff:.1f}%)",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=12, ha='center', va='top')
            
            # Set plot labels
            ax.set_xlabel(f'Material {prop} Property', fontsize=16, fontweight='bold')
            ax.set_ylabel(feature_name, fontsize=16, fontweight='bold')
            ax.set_title(f'Distribution of {feature_name} by {prop} Property', fontsize=18, fontweight='bold')
            
            # Set x-tick labels
            ax.set_xticks([1, 2])
            ax.set_xticklabels([f'No {prop}', f'With {prop}'], fontsize=14)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set overall title
        fig.suptitle(f'Correlation Between {prop} Property and Cluster Features',
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Add explanatory note
        note = (f"Note: This visualization shows the distribution of various cluster features\n"
               f"for materials with and without the {prop} property.")
        fig.text(0.5, 0.01, note, ha='center', fontsize=14, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Save with high resolution
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{output_dir}/{prop.lower()}_correlation_plots.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/{prop.lower()}_correlation_plots.pdf", bbox_inches='tight')
        plt.close()
    
    # Create pairwise correlation matrix for numerical features
    numerical_cols = ['num_clusters_clean', 'max_cluster_size', 'min_avg_distance_clean', 
                     'energy_above_hull_clean', 'Enantiomorphic_binary', 'Piezoelectric_binary', 'Polar_binary']
    
    # Clean data for correlation matrix
    corr_df = df[numerical_cols].dropna()
    
    if len(corr_df) > 10:
        plt.figure(figsize=(16, 14))
        corr_matrix = corr_df.corr()
        
        # Define friendly column names for display
        friendly_names = {
            'num_clusters_clean': 'Number of Clusters',
            'max_cluster_size': 'Max Cluster Size',
            'min_avg_distance_clean': 'Min Avg Distance',
            'energy_above_hull_clean': 'Energy Above Hull',
            'Enantiomorphic_binary': 'Enantiomorphic',
            'Piezoelectric_binary': 'Piezoelectric',
            'Polar_binary': 'Polar'
        }
        
        # Rename for display
        display_matrix = corr_matrix.copy()
        display_matrix.index = [friendly_names.get(col, col) for col in display_matrix.index]
        display_matrix.columns = [friendly_names.get(col, col) for col in display_matrix.columns]
        
        # Plot heatmap
        cmap = plt.cm.RdBu_r
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        im = plt.imshow(display_matrix, cmap=cmap, vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=16, fontweight='bold')
        
        # Add text annotations
        for i in range(len(display_matrix.index)):
            for j in range(len(display_matrix.columns)):
                value = display_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color, fontweight='bold', fontsize=14)
        
        # Configure axes
        plt.xticks(range(len(display_matrix.columns)), display_matrix.columns, rotation=45, ha='right', fontsize=14)
        plt.yticks(range(len(display_matrix.index)), display_matrix.index, fontsize=14)
        
        # Set title
        plt.title('Correlation Matrix of Material Properties and Cluster Features', fontsize=20, fontweight='bold', pad=20)
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/property_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/property_correlation_matrix.pdf", bbox_inches='tight')
        plt.close()
    
    # Create binary property correlation heatmap
    binary_df = df[['Enantiomorphic_binary', 'Piezoelectric_binary', 'Polar_binary']].dropna()
    
    if len(binary_df) > 10:
        plt.figure(figsize=(12, 10))
        binary_corr = binary_df.corr()
        
        # Create heatmap
        im = plt.imshow(binary_corr, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=16, fontweight='bold')
        
        # Add text annotations
        for i in range(len(binary_corr.index)):
            for j in range(len(binary_corr.columns)):
                value = binary_corr.iloc[i, j]
                color = 'white' if value > 0.5 else 'black'
                plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color, fontweight='bold', fontsize=14)
        
        # Configure axes
        plt.xticks(range(len(binary_corr.columns)), binary_corr.columns, fontsize=14)
        plt.yticks(range(len(binary_corr.index)), binary_corr.index, fontsize=14)
        
        # Set title
        plt.title('Correlation Between Binary Material Properties', fontsize=20, fontweight='bold', pad=20)
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/binary_property_correlation.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/binary_property_correlation.pdf", bbox_inches='tight')
        plt.close()
    
    print(f"Saved correlation plots to {output_dir}")

def main():
    """Main function to run the property visualizations."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_csv_path = script_dir.parent / "merged_results_all_prop_filtered_v3.csv"
    output_dir = script_dir / "property_visualizations"
    
    print("Property Visualizations Script")
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
    
    processed_df = preprocess_data(df)
    if processed_df is None:
        print("Error: Failed to preprocess data. Exiting.")
        return
    
    # Create visualizations
    plot_space_group_distribution(processed_df, output_dir, top_n=20, cmap="plasma")
    plot_dimensionality_pie(processed_df, output_dir, cmap="viridis")
    plot_combined_properties_pie(processed_df, output_dir, cmap="magma")
    create_cluster_size_point_group_property_correlation(processed_df, output_dir)
    create_stability_analysis(processed_df, output_dir)
    create_separate_correlation_plots(processed_df, output_dir)
    
    print("\nAll visualizations complete!")
    print("\nGenerated files:")
    print("- space_group_distribution.png/pdf")
    print("- dimensionality_distribution.png/pdf")
    print("- property_distribution.png/pdf")
    print("- structure_property_correlation.png/pdf")
    print("- stability_analysis plots (multiple files)")
    print("- separate correlation plots (multiple files)")

if __name__ == "__main__":
    main()