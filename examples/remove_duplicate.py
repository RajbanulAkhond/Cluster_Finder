#!/usr/bin/env python3
"""
Script to remove duplicate materials from CSV based on material_id.

Rules for handling duplicates:
1. If duplicates are in the same compound system: remove one of them
2. If duplicates are in different compound systems: keep the one where the 
   transition metal appears more times in cluster_elements, remove others
3. If transition metals appear same number of times: keep one, remove others
"""

import pandas as pd
import ast
import argparse
import sys
from collections import defaultdict


def get_transition_metals():
    """Return a set of transition metal symbols."""
    return {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'
    }


def parse_compound_system(compound_system):
    """
    Parse compound system to extract transition metal and anion.
    
    Args:
        compound_system: String like "Ta-B", "Mo-B", "Fe-Nb-B"
    
    Returns:
        tuple: (transition_metal, anion) where transition_metal is the first
               transition metal found, and anion is the last element
    """
    elements = compound_system.split('-')
    transition_metals = get_transition_metals()
    
    # Find the first transition metal
    transition_metal = None
    for element in elements:
        if element in transition_metals:
            transition_metal = element
            break
    
    # The anion is typically the last element
    anion = elements[-1] if elements else ""
    
    return (transition_metal or "", anion)


def create_sort_key(compound_system):
    """
    Create a sort key for compound systems: first by transition metal, then by anion.
    
    Args:
        compound_system: String like "Ta-B", "Mo-B"
    
    Returns:
        tuple: Sort key (transition_metal, anion)
    """
    transition_metal, anion = parse_compound_system(compound_system)
    return (transition_metal, anion)


def count_transition_metal_in_clusters(cluster_elements_str, compound_system):
    """
    Count occurrences of the transition metal from compound_system in cluster_elements.
    
    Args:
        cluster_elements_str: String representation of cluster elements
        compound_system: String like "Ta-B" or "Mo-B"
    
    Returns:
        int: Count of transition metal occurrences
    """
    try:
        # Parse the cluster elements string
        cluster_elements = ast.literal_eval(cluster_elements_str)
        
        # Get the transition metal from compound system (first element before hyphen)
        transition_metal = compound_system.split('-')[0]
        
        # Count occurrences across all clusters
        count = 0
        for cluster in cluster_elements:
            count += cluster.count(transition_metal)
        
        return count
    except (ValueError, SyntaxError, AttributeError):
        return 0


def remove_duplicates(df):
    """
    Remove duplicate materials based on material_id with specified logic.
    
    Args:
        df: pandas DataFrame with the materials data
    
    Returns:
        pandas DataFrame: Cleaned DataFrame without duplicates
    """
    transition_metals = get_transition_metals()
    
    # Group by material_id to find duplicates
    grouped = df.groupby('material_id')
    
    rows_to_keep = []
    duplicates_removed = 0
    
    for material_id, group in grouped:
        if len(group) == 1:
            # No duplicates, keep the row
            rows_to_keep.append(group.index[0])
        else:
            # Handle duplicates
            duplicates_removed += len(group) - 1
            
            # Check if all duplicates are in the same compound system
            compound_systems = group['compound_system'].unique()
            
            if len(compound_systems) == 1:
                # Same compound system - keep the first one
                rows_to_keep.append(group.index[0])
                print(f"  Removed {len(group)-1} duplicate(s) of {material_id} in same compound system {compound_systems[0]}")
            else:
                # Different compound systems - apply transition metal count logic
                best_row_idx = None
                max_tm_count = -1
                
                for idx, row in group.iterrows():
                    tm_count = count_transition_metal_in_clusters(
                        row['cluster_elements'], 
                        row['compound_system']
                    )
                    
                    if tm_count > max_tm_count:
                        max_tm_count = tm_count
                        best_row_idx = idx
                
                if best_row_idx is not None:
                    rows_to_keep.append(best_row_idx)
                else:
                    # Fallback - keep the first one
                    rows_to_keep.append(group.index[0])
                
                systems_str = ", ".join(compound_systems)
                print(f"  Removed {len(group)-1} duplicate(s) of {material_id} across systems [{systems_str}], kept system with most transition metal occurrences")
    
    return df.loc[rows_to_keep].copy(), duplicates_removed


def main():
    parser = argparse.ArgumentParser(description='Remove duplicate materials from CSV based on material_id')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: input_file with _no_duplicates suffix)')
    
    args = parser.parse_args()
    
    # Read the CSV file
    try:
        print(f"Reading CSV file: {args.input_file}")
        df = pd.read_csv(args.input_file)
        print(f"Total materials loaded: {len(df)}")
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = ['material_id', 'compound_system', 'cluster_elements']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        sys.exit(1)
    
    # Find and display duplicate statistics
    duplicate_count = df['material_id'].duplicated().sum()
    unique_materials = df['material_id'].nunique()
    
    print(f"Unique materials: {unique_materials}")
    print(f"Duplicate entries found: {duplicate_count}")
    
    if duplicate_count == 0:
        print("No duplicates found. Nothing to remove.")
        return
    
    print("\nRemoving duplicates...")
    
    # Remove duplicates
    cleaned_df, duplicates_removed = remove_duplicates(df)
    
    # Sort the cleaned DataFrame by compound system (transition metal first, then anion)
    print("Organizing compound systems by transition metal and anion...")
    cleaned_df['_sort_key'] = cleaned_df['compound_system'].apply(create_sort_key)
    cleaned_df = cleaned_df.sort_values('_sort_key').drop('_sort_key', axis=1)
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        if args.input_file.endswith('.csv'):
            output_file = args.input_file[:-4] + '_no_duplicates.csv'
        else:
            output_file = args.input_file + '_no_duplicates.csv'
    
    # Save the cleaned CSV
    try:
        cleaned_df.to_csv(output_file, index=False)
        print(f"\nCleaned CSV saved to: {output_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        sys.exit(1)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Original total materials: {len(df)}")
    print(f"  Final total materials: {len(cleaned_df)}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Reduction: {duplicates_removed/len(df)*100:.1f}%")
    
    # Show organization summary
    compound_systems = cleaned_df['compound_system'].unique()
    transition_metals = set()
    for cs in compound_systems:
        tm, _ = parse_compound_system(cs)
        if tm:
            transition_metals.add(tm)
    
    print(f"  Organized by {len(transition_metals)} transition metals: {sorted(transition_metals)}")


if __name__ == "__main__":
    main()