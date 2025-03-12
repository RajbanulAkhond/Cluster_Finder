#!/usr/bin/env python
"""
Example script to demonstrate the use of the rank_clusters function.

This script:
1. Loads the Ir-O_ranked_clusters.csv dataset
2. Ranks the clusters using default criteria and additional properties
3. Displays the top 10 ranked materials with their properties
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import the cluster_finder module
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from cluster_finder.analysis.postprocess import rank_clusters

def main():
    # Path to the test data CSV file
    data_file = os.path.join(parent_dir, "cluster_finder", "tests", "data", "Ir-O_ranked_clusters.csv")
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    print(f"Loading cluster data from {data_file}")
    
    # Define custom properties to retrieve from Materials Project
    # These will be added to the default ranking criteria
    custom_props = ["formation_energy_per_atom"]
    
    # Define custom weights for properties (positive values favor higher values, negative favor lower)
    prop_weights = {
        "formation_energy_per_atom": -1.0  # Lower formation energy is favored
    }
    
    print("Ranking clusters with default criteria plus formation energy...")
    
    # Get your Materials Project API key from environment variable
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("Warning: MP_API_KEY environment variable not set.")
        print("Continuing without retrieving additional properties from Materials Project.")
        custom_props = None
        prop_weights = None
    
    # Rank the clusters
    ranked_df = rank_clusters(
        data_source=data_file,
        api_key=api_key,
        custom_props=custom_props,
        prop_weights=prop_weights,
        include_default_ranking=True
    )
    
    # Display top 10 ranked materials with their properties
    print("\nTop 10 Ranked Iridium Oxide Materials:")
    print("======================================")
    
    # Select important columns to display - adjust based on available columns
    display_columns = [
        "material_id", "formula", "rank_score",
        "energy_above_hull", "highest_point_group", "max_point_group_order",
        "space_group", "space_group_order", "min_avg_distance"
    ]
    
    # Add custom properties if they were retrieved
    if custom_props:
        for prop in custom_props:
            if prop in ranked_df.columns:
                display_columns.append(prop)
    
    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in display_columns if col in ranked_df.columns]
    
    # Display the top 10 results with selected columns
    top_10 = ranked_df.head(10)[existing_columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(top_10)
    
    # Save the ranked results to a CSV file
    output_file = os.path.join(parent_dir, "examples", "Ir-O_ranked_results.csv")
    ranked_df.to_csv(output_file, index=False)
    print(f"\nFull ranked results saved to {output_file}")

if __name__ == "__main__":
    main()