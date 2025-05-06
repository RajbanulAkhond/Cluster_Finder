#!/usr/bin/env python
"""
Example script demonstrating the use of the refactored cluster_finder
for analyzing multiple transition metal-anion systems in parallel.
"""
import os
import time
from pathlib import Path

from cluster_finder.analysis.analysis import run_analysis
from cluster_finder.analysis.batch import run_batch_analysis
from cluster_finder.utils.config_utils import load_config, get_element_combinations

def example_single_system():
    """Example of analyzing a single TM-anion system."""
    # Define your Materials Project API key
    # Ideally, this should be stored in an environment variable
    API_KEY = os.environ.get("MP_API_KEY", "your_api_key_here")
    
    # Create an output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Run analysis for a single system
    result = run_analysis(
        primary_tm="Nb",  # Primary transition metal
        anion="Cl",       # Anion
        api_key=API_KEY,
        output_dir=output_dir,
        n_jobs=2,         # Use 2 parallel jobs for CPU-intensive tasks
        save_pdf=True,
        save_csv=True
    )
    
    print(f"Analysis completed for Nb-Cl system")
    print(f"Found {result['compounds_count']} compounds")
    print(f"Found {result.get('compounds_with_clusters_count', 0)} compounds with clusters")
    print(f"Time taken: {result.get('time_taken', 0):.2f} seconds")
    
    if result.get('outputs', {}).get('pdf'):
        print(f"PDF report saved to: {result['outputs']['pdf']}")
    if result.get('outputs', {}).get('summary_csv'):
        print(f"CSV data saved to: {result['outputs']['summary_csv']}")
    
    return result

def example_batch_systems():
    """Example of analyzing multiple TM-anion systems in parallel."""
    # Define your Materials Project API key
    API_KEY = os.environ.get("MP_API_KEY", "your_api_key_here")
    
    # Create an output directory
    output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load the default configuration
    config = load_config()
    
    # You can override specific elements if needed
    specific_tms = ["Nb", "V"]
    specific_anions = ["Cl", "O"]
    
    # Get all combinations (or you could define them manually)
    systems = [(tm, anion) for tm in specific_tms for anion in specific_anions]
    print(f"Analyzing {len(systems)} systems: {systems}")
    
    # Set the number of parallel workers
    max_workers = min(4, len(systems))  # Use at most 4 workers or number of systems
    
    # Run batch analysis
    start_time = time.time()
    
    result = run_batch_analysis(
        api_key=API_KEY,
        output_dir=output_dir,
        specific_tms=specific_tms,
        specific_anions=specific_anions,
        max_workers=max_workers,    # Number of parallel system analyses
        n_jobs_per_analysis=2,      # Number of parallel jobs within each analysis
        save_pdf=True,
        save_csv=True
    )
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"Batch analysis completed in {elapsed_time:.2f} seconds")
    print(f"Completed systems: {result.get('completed_systems', 0)}")
    print(f"Failed systems: {result.get('failed_systems', 0)}")
    
    # Print results for each system
    for system_name, system_result in result.get("results", {}).items():
        status = system_result.get("status", "unknown")
        print(f"System {system_name}: {status}")
        if status == "completed":
            print(f"  - Compounds: {system_result.get('compounds_count', 0)}")
            print(f"  - With clusters: {system_result.get('compounds_with_clusters_count', 0)}")
            print(f"  - Time taken: {system_result.get('time_taken', 0):.2f} seconds")
    
    print(f"Detailed results saved to: {output_dir / 'batch_summary.json'}")
    
    return result

def example_cli_usage():
    """Example showing equivalent CLI commands."""
    print("Instead of running this script, you can use the CLI commands:")
    print("\n# For a single system:")
    print("cluster-finder single Nb Cl --api-key YOUR_API_KEY --output-dir results --n-jobs 2")
    
    print("\n# For batch analysis:")
    print("cluster-finder batch --api-key YOUR_API_KEY --output-dir batch_results " +
          "--tms Nb V --anions Cl O --max-workers 4 --n-jobs 2")
    
    print("\n# To show systems without running analysis:")
    print("cluster-finder batch --show-systems --tms Nb V --anions Cl O")

if __name__ == "__main__":
    # Uncomment one of these to run the examples
    # example_single_system()
    # example_batch_systems()
    example_cli_usage()
    
    print("\nRemember to set your Materials Project API key in the environment variable MP_API_KEY")
    print("For example: export MP_API_KEY=your_api_key_here")