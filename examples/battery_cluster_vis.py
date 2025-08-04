#!/usr/bin/env python3
"""
Battery-Cluster Visualization Script

This script creates publication-quality visualizations showing correlations and trends
between cluster statistics and battery properties using data from battery compounds.

Author: Cluster Finder Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })

def load_data(data_path):
    """Load and preprocess the battery compounds data."""
    print(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the CSV data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total compounds")
    
    # Filter out compounds without battery data
    battery_cols = ['battery_type', 'stability_charge', 'stability_discharge', 'average_voltage', 'capacity_grav']
    df_battery = df.dropna(subset=battery_cols, how='all')
    print(f"Found {len(df_battery)} compounds with battery data")
    
    # Parse cluster sizes from string representation
    df_battery = df_battery.copy()
    df_battery['max_cluster_size'] = df_battery['cluster_sizes'].apply(
        lambda x: max(eval(x)) if pd.notna(x) and x != '[]' else 0
    )
    df_battery['avg_cluster_size'] = df_battery['cluster_sizes'].apply(
        lambda x: np.mean(eval(x)) if pd.notna(x) and x != '[]' else 0
    )
    
    return df_battery

def plot_capacity_vs_cluster_properties(df, output_dir):
    """Plot battery capacity vs cluster size relationships."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Battery Capacity vs Cluster Properties', fontsize=18, fontweight='bold')
    
    # Filter data with valid capacity values
    df_valid = df.dropna(subset=['capacity_grav'])
    
    # Plot 1: Capacity vs Number of Clusters
    ax1 = axes[0, 0]
    valid_mask1 = df_valid['num_clusters'].notna() & df_valid['capacity_grav'].notna()
    if valid_mask1.sum() > 1:
        scatter = ax1.scatter(df_valid[valid_mask1]['num_clusters'], df_valid[valid_mask1]['capacity_grav'], 
                             alpha=0.7, s=60, c=df_valid[valid_mask1]['average_voltage'], 
                             cmap='viridis', edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Gravimetric Capacity (mAh/g)')
        ax1.set_title('Capacity vs Number of Clusters')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        try:
            r, p = pearsonr(df_valid[valid_mask1]['num_clusters'], df_valid[valid_mask1]['capacity_grav'])
            ax1.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                     transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
        except:
            ax1.text(0.05, 0.95, 'Correlation\nnot available', 
                     transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
    
    # Plot 2: Capacity vs Max Cluster Size
    ax2 = axes[0, 1]
    valid_mask2 = df_valid['max_cluster_size'].notna() & df_valid['capacity_grav'].notna()
    if valid_mask2.sum() > 1:
        ax2.scatter(df_valid[valid_mask2]['max_cluster_size'], df_valid[valid_mask2]['capacity_grav'], 
                    alpha=0.7, s=60, c=df_valid[valid_mask2]['average_voltage'], 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Maximum Cluster Size')
        ax2.set_ylabel('Gravimetric Capacity (mAh/g)')
        ax2.set_title('Capacity vs Maximum Cluster Size')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        try:
            r, p = pearsonr(df_valid[valid_mask2]['max_cluster_size'], df_valid[valid_mask2]['capacity_grav'])
            ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                     transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
        except:
            ax2.text(0.05, 0.95, 'Correlation\nnot available', 
                     transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
    
    # Plot 3: Capacity vs Min Average Distance
    ax3 = axes[1, 0]
    valid_mask3 = df_valid['min_avg_distance'].notna() & df_valid['capacity_grav'].notna()
    if valid_mask3.sum() > 1:
        ax3.scatter(df_valid[valid_mask3]['min_avg_distance'], df_valid[valid_mask3]['capacity_grav'], 
                    alpha=0.7, s=60, c=df_valid[valid_mask3]['average_voltage'], 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Min Average Distance (Å)')
        ax3.set_ylabel('Gravimetric Capacity (mAh/g)')
        ax3.set_title('Capacity vs Min Average Distance')
        ax3.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        try:
            r, p = pearsonr(df_valid[valid_mask3]['min_avg_distance'], df_valid[valid_mask3]['capacity_grav'])
            ax3.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                     transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
        except:
            ax3.text(0.05, 0.95, 'Correlation\nnot available', 
                     transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
    
    # Plot 4: Capacity vs Formation Energy
    ax4 = axes[1, 1]
    valid_mask4 = df_valid['formation_energy_per_atom'].notna() & df_valid['capacity_grav'].notna()
    if valid_mask4.sum() > 1:
        ax4.scatter(df_valid[valid_mask4]['formation_energy_per_atom'], df_valid[valid_mask4]['capacity_grav'], 
                    alpha=0.7, s=60, c=df_valid[valid_mask4]['average_voltage'], 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Formation Energy per Atom (eV)')
        ax4.set_ylabel('Gravimetric Capacity (mAh/g)')
        ax4.set_title('Capacity vs Formation Energy')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        try:
            r, p = pearsonr(df_valid[valid_mask4]['formation_energy_per_atom'], df_valid[valid_mask4]['capacity_grav'])
            ax4.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                     transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
        except:
            ax4.text(0.05, 0.95, 'Correlation\nnot available', 
                     transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')
    
    # Add colorbar positioned to not overlap with plots
    if valid_mask1.sum() > 1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Average Voltage (V)', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(f"{output_dir}/battery_capacity_vs_cluster_properties.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_capacity_vs_cluster_properties.pdf", bbox_inches='tight')
    print(f"Saved capacity vs cluster properties plot to {output_dir}")
    plt.close()

def plot_voltage_vs_cluster_properties(df, output_dir):
    """Plot average voltage vs cluster properties."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Battery Voltage vs Cluster Properties', fontsize=18, fontweight='bold')
    
    # Filter data with valid voltage values
    df_valid = df.dropna(subset=['average_voltage'])
    
    # Plot 1: Voltage vs Number of Clusters
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_valid['num_clusters'], df_valid['average_voltage'], 
                         alpha=0.7, s=60, c=df_valid['capacity_grav'], 
                         cmap='plasma', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Average Voltage (V)')
    ax1.set_title('Voltage vs Number of Clusters')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['num_clusters'], df_valid['average_voltage'])
        ax1.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax1.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Plot 2: Voltage vs Average Cluster Size
    ax2 = axes[0, 1]
    ax2.scatter(df_valid['avg_cluster_size'], df_valid['average_voltage'], 
                alpha=0.7, s=60, c=df_valid['capacity_grav'], 
                cmap='plasma', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Average Cluster Size')
    ax2.set_ylabel('Average Voltage (V)')
    ax2.set_title('Voltage vs Average Cluster Size')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['avg_cluster_size'], df_valid['average_voltage'])
        ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax2.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Plot 3: Voltage vs Min Average Distance
    ax3 = axes[1, 0]
    ax3.scatter(df_valid['min_avg_distance'], df_valid['average_voltage'], 
                alpha=0.7, s=60, c=df_valid['capacity_grav'], 
                cmap='plasma', edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Min Average Distance (Å)')
    ax3.set_ylabel('Average Voltage (V)')
    ax3.set_title('Voltage vs Min Average Distance')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['min_avg_distance'], df_valid['average_voltage'])
        ax3.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax3.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Plot 4: Voltage vs Band Gap
    ax4 = axes[1, 1]
    ax4.scatter(df_valid['band_gap'], df_valid['average_voltage'], 
                alpha=0.7, s=60, c=df_valid['capacity_grav'], 
                cmap='plasma', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Band Gap (eV)')
    ax4.set_ylabel('Average Voltage (V)')
    ax4.set_title('Voltage vs Band Gap')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['band_gap'], df_valid['average_voltage'])
        ax4.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax4.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Add colorbar positioned to not overlap with plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Gravimetric Capacity (mAh/g)', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(f"{output_dir}/battery_voltage_vs_cluster_properties.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_voltage_vs_cluster_properties.pdf", bbox_inches='tight')
    print(f"Saved voltage vs cluster properties plot to {output_dir}")
    plt.close()

def plot_stability_analysis(df, output_dir):
    """Plot battery stability vs cluster properties."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Battery Stability vs Cluster Properties', fontsize=18, fontweight='bold')
    
    # Filter data with valid stability values
    df_valid = df.dropna(subset=['stability_charge', 'stability_discharge'])
    
    # Plot 1: Charge Stability vs Energy Above Hull
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_valid['energy_above_hull'], df_valid['stability_charge'], 
                         alpha=0.7, s=60, c=df_valid['num_clusters'], 
                         cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Energy Above Hull (eV/atom)')
    ax1.set_ylabel('Charge Stability')
    ax1.set_title('Charge Stability vs Energy Above Hull')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['energy_above_hull'], df_valid['stability_charge'])
        ax1.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax1.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Plot 2: Discharge Stability vs Formation Energy
    ax2 = axes[0, 1]
    ax2.scatter(df_valid['formation_energy_per_atom'], df_valid['stability_discharge'], 
                alpha=0.7, s=60, c=df_valid['num_clusters'], 
                cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Formation Energy per Atom (eV)')
    ax2.set_ylabel('Discharge Stability')
    ax2.set_title('Discharge Stability vs Formation Energy')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['formation_energy_per_atom'], df_valid['stability_discharge'])
        ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax2.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Plot 3: Stability Difference vs Cluster Count
    df_valid['stability_diff'] = df_valid['stability_charge'] - df_valid['stability_discharge']
    ax3 = axes[1, 0]
    ax3.scatter(df_valid['num_clusters'], df_valid['stability_diff'], 
                alpha=0.7, s=60, c=df_valid['average_voltage'], 
                cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Stability Difference (Charge - Discharge)')
    ax3.set_title('Stability Difference vs Number of Clusters')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['num_clusters'], df_valid['stability_diff'])
        ax3.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax3.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Plot 4: Average Stability vs Min Average Distance
    df_valid['avg_stability'] = (df_valid['stability_charge'] + df_valid['stability_discharge']) / 2
    ax4 = axes[1, 1]
    ax4.scatter(df_valid['min_avg_distance'], df_valid['avg_stability'], 
                alpha=0.7, s=60, c=df_valid['average_voltage'], 
                cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Min Average Distance (Å)')
    ax4.set_ylabel('Average Stability')
    ax4.set_title('Average Stability vs Min Average Distance')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    try:
        r, p = pearsonr(df_valid['min_avg_distance'], df_valid['avg_stability'])
        ax4.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                 transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    except:
        ax4.text(0.05, 0.95, 'Correlation\nnot available', 
                 transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    
    # Add colorbar positioned to not overlap with plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Number of Clusters', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(f"{output_dir}/battery_stability_vs_cluster_properties.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_stability_vs_cluster_properties.pdf", bbox_inches='tight')
    print(f"Saved stability vs cluster properties plot to {output_dir}")
    plt.close()

def plot_correlation_matrix(df, output_dir):
    """Plot correlation matrix between battery and cluster properties."""
    # Select relevant columns for correlation analysis - focus on cluster properties
    battery_cols = ['capacity_grav', 'average_voltage', 'stability_charge', 'stability_discharge']
    cluster_cols = ['num_clusters', 'max_cluster_size', 'avg_cluster_size', 
                   'min_avg_distance', 'total_magnetization', 'formation_energy_per_atom', 
                   'energy_above_hull', 'band_gap']
    
    # Create a copy of the dataframe and convert data types
    df_clean = df.copy()
    
    # Ensure all columns are numeric
    all_cols = battery_cols + cluster_cols
    available_cols = [col for col in all_cols if col in df_clean.columns]
    
    # Convert to numeric, coercing errors to NaN
    for col in available_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Filter data and compute correlations
    df_corr = df_clean[available_cols].dropna()
    
    if len(df_corr) < 2:
        print("Insufficient data for correlation analysis")
        return
    
    # Remove columns with no variance (all identical values)
    df_corr = df_corr.loc[:, df_corr.var() != 0]
    
    if len(df_corr.columns) < 2:
        print("Insufficient columns with variance for correlation analysis")
        return
    
    correlation_matrix = df_corr.corr()
    
    # Create the plot with larger figure size to accommodate text
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": 0.6}, ax=ax)
    
    ax.set_title('Correlation Matrix: Battery vs Cluster Properties', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add statistics text positioned outside the plot area
    n_samples = len(df_corr)
    stats_text = f'n = {n_samples} compounds\nwith complete data'
    
    # Position text box outside the plot area
    plt.figtext(0.85, 0.15, stats_text,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=12, ha='left', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/battery_cluster_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_cluster_correlation_matrix.pdf", bbox_inches='tight')
    print(f"Saved correlation matrix to {output_dir}")
    plt.close()

def plot_compound_system_analysis(df, output_dir):
    """Plot battery performance by compound system."""
    # Filter data with battery properties
    df_valid = df.dropna(subset=['capacity_grav', 'average_voltage'])
    
    # Get top compound systems by count
    system_counts = df_valid['compound_system'].value_counts()
    top_systems = system_counts.head(10).index
    df_top = df_valid[df_valid['compound_system'].isin(top_systems)]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Battery Performance by Compound System', fontsize=18, fontweight='bold')
    
    # Plot 1: Capacity by compound system
    ax1 = axes[0, 0]
    box_data1 = [df_top[df_top['compound_system'] == system]['capacity_grav'].values 
                 for system in top_systems]
    bp1 = ax1.boxplot(box_data1, labels=top_systems, patch_artist=True)
    ax1.set_xlabel('Compound System')
    ax1.set_ylabel('Gravimetric Capacity (mAh/g)')
    ax1.set_title('Capacity Distribution by Compound System')
    ax1.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp1['boxes'])))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 2: Voltage by compound system
    ax2 = axes[0, 1]
    box_data2 = [df_top[df_top['compound_system'] == system]['average_voltage'].values 
                 for system in top_systems]
    bp2 = ax2.boxplot(box_data2, labels=top_systems, patch_artist=True)
    ax2.set_xlabel('Compound System')
    ax2.set_ylabel('Average Voltage (V)')
    ax2.set_title('Voltage Distribution by Compound System')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 3: Number of clusters by compound system
    ax3 = axes[1, 0]
    box_data3 = [df_top[df_top['compound_system'] == system]['num_clusters'].values 
                 for system in top_systems]
    bp3 = ax3.boxplot(box_data3, labels=top_systems, patch_artist=True)
    ax3.set_xlabel('Compound System')
    ax3.set_ylabel('Number of Clusters')
    ax3.set_title('Cluster Count Distribution by Compound System')
    ax3.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 4: Scatter plot of capacity vs voltage colored by system
    ax4 = axes[1, 1]
    for i, system in enumerate(top_systems):
        system_data = df_top[df_top['compound_system'] == system]
        ax4.scatter(system_data['average_voltage'], system_data['capacity_grav'], 
                   label=system, alpha=0.7, s=60, color=colors[i])
    
    ax4.set_xlabel('Average Voltage (V)')
    ax4.set_ylabel('Gravimetric Capacity (mAh/g)')
    ax4.set_title('Capacity vs Voltage by Compound System')
    # Position legend outside plot area at the bottom
    ax4.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=5, fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend at bottom
    plt.savefig(f"{output_dir}/battery_performance_by_compound_system.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_performance_by_compound_system.pdf", bbox_inches='tight')
    print(f"Saved compound system analysis to {output_dir}")
    plt.close()

def plot_dimensionality_analysis(df, output_dir):
    """Plot battery performance by predicted dimensionality."""
    # Filter data with battery properties and dimensionality
    df_valid = df.dropna(subset=['capacity_grav', 'average_voltage', 'predicted_dimentionality'])
    
    if len(df_valid) == 0:
        print("No data available for dimensionality analysis")
        return
    
    # Get dimensionality counts
    dim_counts = df_valid['predicted_dimentionality'].value_counts()
    print(f"Dimensionality distribution: {dim_counts}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Battery Performance by Predicted Dimensionality', fontsize=18, fontweight='bold')
    
    # Plot 1: Capacity by dimensionality
    ax1 = axes[0, 0]
    dimensions = df_valid['predicted_dimentionality'].unique()
    box_data1 = [df_valid[df_valid['predicted_dimentionality'] == dim]['capacity_grav'].values 
                 for dim in dimensions]
    bp1 = ax1.boxplot(box_data1, labels=dimensions, patch_artist=True)
    ax1.set_xlabel('Predicted Dimensionality')
    ax1.set_ylabel('Gravimetric Capacity (mAh/g)')
    ax1.set_title('Capacity by Dimensionality')
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp1['boxes'])))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 2: Voltage by dimensionality
    ax2 = axes[0, 1]
    box_data2 = [df_valid[df_valid['predicted_dimentionality'] == dim]['average_voltage'].values 
                 for dim in dimensions]
    bp2 = ax2.boxplot(box_data2, labels=dimensions, patch_artist=True)
    ax2.set_xlabel('Predicted Dimensionality')
    ax2.set_ylabel('Average Voltage (V)')
    ax2.set_title('Voltage by Dimensionality')
    
    # Color the boxes
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 3: Number of clusters by dimensionality
    ax3 = axes[1, 0]
    box_data3 = [df_valid[df_valid['predicted_dimentionality'] == dim]['num_clusters'].values 
                 for dim in dimensions]
    bp3 = ax3.boxplot(box_data3, labels=dimensions, patch_artist=True)
    ax3.set_xlabel('Predicted Dimensionality')
    ax3.set_ylabel('Number of Clusters')
    ax3.set_title('Cluster Count by Dimensionality')
    
    # Color the boxes
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 4: Min Average Distance by dimensionality
    ax4 = axes[1, 1]
    box_data4 = [df_valid[df_valid['predicted_dimentionality'] == dim]['min_avg_distance'].values 
                 for dim in dimensions]
    bp4 = ax4.boxplot(box_data4, labels=dimensions, patch_artist=True)
    ax4.set_xlabel('Predicted Dimensionality')
    ax4.set_ylabel('Min Average Distance (Å)')
    ax4.set_title('Min Avg Distance by Dimensionality')
    
    # Color the boxes
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/battery_performance_by_dimensionality.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_performance_by_dimensionality.pdf", bbox_inches='tight')
    print(f"Saved dimensionality analysis to {output_dir}")
    plt.close()

def plot_point_group_analysis(df, output_dir):
    """Plot battery performance by cluster point groups."""
    # Filter data with battery properties and point groups
    df_valid = df.dropna(subset=['capacity_grav', 'average_voltage'])
    
    # Extract main point group from the point_groups column
    def extract_main_point_group(pg_str):
        if pd.isna(pg_str) or not isinstance(pg_str, str):
            return "Unknown"
        try:
            if '{' in pg_str:
                # Parse dictionary-like string
                pg_dict = eval(pg_str.replace("'", '"'))
                if isinstance(pg_dict, dict) and pg_dict:
                    return list(pg_dict.values())[0]  # Get first point group
            return pg_str
        except:
            return "Unknown"
    
    df_valid['main_point_group'] = df_valid['point_groups'].apply(extract_main_point_group)
    df_valid = df_valid[df_valid['main_point_group'] != "Unknown"]
    
    if len(df_valid) == 0:
        print("No data available for point group analysis")
        return
    
    # Get top point groups by frequency
    pg_counts = df_valid['main_point_group'].value_counts()
    top_pgs = pg_counts.head(8).index  # Top 8 most common point groups
    df_top = df_valid[df_valid['main_point_group'].isin(top_pgs)]
    
    print(f"Top point groups: {list(top_pgs)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Battery Performance by Cluster Point Groups', fontsize=18, fontweight='bold')
    
    # Plot 1: Capacity by point group
    ax1 = axes[0, 0]
    box_data1 = [df_top[df_top['main_point_group'] == pg]['capacity_grav'].values 
                 for pg in top_pgs]
    bp1 = ax1.boxplot(box_data1, labels=top_pgs, patch_artist=True)
    ax1.set_xlabel('Point Group')
    ax1.set_ylabel('Gravimetric Capacity (mAh/g)')
    ax1.set_title('Capacity by Point Group')
    ax1.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(bp1['boxes'])))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 2: Voltage by point group
    ax2 = axes[0, 1]
    box_data2 = [df_top[df_top['main_point_group'] == pg]['average_voltage'].values 
                 for pg in top_pgs]
    bp2 = ax2.boxplot(box_data2, labels=top_pgs, patch_artist=True)
    ax2.set_xlabel('Point Group')
    ax2.set_ylabel('Average Voltage (V)')
    ax2.set_title('Voltage by Point Group')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 3: Cluster size distribution by point group
    ax3 = axes[1, 0]
    box_data3 = [df_top[df_top['main_point_group'] == pg]['max_cluster_size'].values 
                 for pg in top_pgs]
    bp3 = ax3.boxplot(box_data3, labels=top_pgs, patch_artist=True)
    ax3.set_xlabel('Point Group')
    ax3.set_ylabel('Max Cluster Size')
    ax3.set_title('Max Cluster Size by Point Group')
    ax3.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 4: Min average distance by point group
    ax4 = axes[1, 1]
    box_data4 = [df_top[df_top['main_point_group'] == pg]['min_avg_distance'].values 
                 for pg in top_pgs]
    bp4 = ax4.boxplot(box_data4, labels=top_pgs, patch_artist=True)
    ax4.set_xlabel('Point Group')
    ax4.set_ylabel('Min Average Distance (Å)')
    ax4.set_title('Min Avg Distance by Point Group')
    ax4.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/battery_performance_by_point_group.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/battery_performance_by_point_group.pdf", bbox_inches='tight')
    print(f"Saved point group analysis to {output_dir}")
    plt.close()

def main():
    """Main function to run the battery-cluster visualizations."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_csv_path = script_dir.parent / "battery_compounds_with_electrode_data_v3.csv"
    output_dir = script_dir / "battery_visualizations"
    
    print("Battery-Cluster Visualization Script")
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
    
    if len(df) == 0:
        print("No battery data found. Exiting.")
        return
    
    print(f"Analyzing {len(df)} compounds with battery data...")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_capacity_vs_cluster_properties(df, output_dir)
    plot_voltage_vs_cluster_properties(df, output_dir)
    plot_stability_analysis(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_compound_system_analysis(df, output_dir)
    plot_dimensionality_analysis(df, output_dir)
    plot_point_group_analysis(df, output_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 30)
    
    battery_stats = df[['capacity_grav', 'average_voltage', 'stability_charge', 'stability_discharge']].describe()
    cluster_stats = df[['num_clusters', 'max_cluster_size', 'min_avg_distance', 'total_magnetization']].describe()
    
    print("\nBattery Properties:")
    print(battery_stats)
    print("\nCluster Properties:")
    print(cluster_stats)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()