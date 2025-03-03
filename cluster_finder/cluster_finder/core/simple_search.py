"""
Simple search algorithms for cluster_finder package.

This module contains simplified functions for finding specific cluster types.
"""

from pymatgen.core.structure import Structure
from .structure import find_non_equivalent_positions
from ..utils.helpers import calculate_metal_distances, get_transition_metals, search_transition_metal_compounds

def find_trimers(structure, unique_sites, distances, transition_metals):
    """
    Search for trimer clusters in a structure.
    
    A trimer is defined as a central transition metal atom with at least 
    two transition metal neighbors within a specified cutoff distance.
    
    Parameters:
        structure (Structure): A pymatgen Structure object
        unique_sites (list): List of non-equivalent sites in the structure
        distances (dict): Dictionary mapping element symbols to cutoff distances
        transition_metals (list): List of transition metal element symbols
        
    Returns:
        list: List of tuples (central_site, neighbors) where neighbors are transition metals
    """
    return [
        (site, [n for n in structure.get_neighbors(site, distances.get(site.specie.symbol)) if n.specie.symbol in transition_metals][:2])
        for site in unique_sites
        if distances.get(site.specie.symbol) and len([n for n in structure.get_neighbors(site, distances[site.specie.symbol]) if n.specie.symbol in transition_metals]) >= 2
    ]


def analyze_compound_trimers(entries=None, transition_metals=None, api_key=None, criteria=None):
    """
    Analyze a list of entries to identify compounds with trimer clusters.
    
    Parameters:
        entries (list, optional): List of entries with structure information.
                                 If None, entries will be fetched using search_transition_metal_compounds.
        transition_metals (list, optional): List of transition metal element symbols.
                                          If None, uses the default list from get_transition_metals().
        api_key (str, optional): Materials Project API key.
        criteria (dict, optional): Additional search criteria for search_transition_metal_compounds.
        
    Returns:
        list: List of dictionaries containing material_id, formula, and trimer information
    """
    # Get default transition metals list if not provided
    if transition_metals is None:
        transition_metals = get_transition_metals()
    
    # If entries are not provided, search for compounds
    if entries is None:
        if criteria is None:
            criteria = {}
        entries = search_transition_metal_compounds(transition_metals, api_key, **criteria)
    
    # Calculate metal distances using the helper function
    metal_distances = calculate_metal_distances(transition_metals, api_key)
    
    compounds_with_trimers = []

    for result in entries:
        structure = result.structure
        material_id = result.material_id
        formula = result.formula_pretty

        unique_sites = find_non_equivalent_positions(structure, transition_metals)
        trimers = find_trimers(structure, unique_sites, metal_distances, transition_metals)

        if trimers:
            compounds_with_trimers.append({
                "material_id": material_id,
                "formula": formula, 
                "trimers": trimers
            })
    
    return compounds_with_trimers


def print_trimer_results(compounds_with_trimers):
    """
    Print a summary of compounds with trimer clusters.
    
    Parameters:
        compounds_with_trimers (list): List of dictionaries with trimer information
    """
    print(f"Found {len(compounds_with_trimers)} compounds with trimer clusters.")
    for compound in compounds_with_trimers:
        print(f"Material ID: {compound['material_id']}, Formula: {compound['formula']}")


def find_trimers_in_materials(api_key, transition_metals=None, criteria=None):
    """
    Convenience function to search for materials with trimer clusters in one step.
    
    Parameters:
        api_key (str): Materials Project API key.
        transition_metals (list, optional): List of transition metal element symbols.
                                          If None, uses the default list from get_transition_metals().
        criteria (dict, optional): Additional search criteria for search_transition_metal_compounds.
        
    Returns:
        list: List of dictionaries containing material_id, formula, and trimer information
    """
    compounds_with_trimers = analyze_compound_trimers(
        transition_metals=transition_metals,
        api_key=api_key,
        criteria=criteria
    )
    
    print_trimer_results(compounds_with_trimers)
    
    return compounds_with_trimers 