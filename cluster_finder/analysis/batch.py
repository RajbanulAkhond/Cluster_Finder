#!/usr/bin/env python
"""
Batch processing module for running multiple TM-anion system analyses in parallel.
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console

from cluster_finder.utils.config_utils import load_config, get_element_combinations
from cluster_finder.utils.logger import get_logger
from cluster_finder.analysis.analysis import run_analysis
from cluster_finder.utils.exceptions import APIRateLimitError, InvalidInputError

# Get the logger without configuring it (will inherit from root logger)
logger = get_logger('cluster_finder.batch')
console = Console()

def run_batch_analysis(
    api_key: str,
    output_dir: Path,
    config_path: Optional[str] = None,
    specific_tms: Optional[List[str]] = None,
    specific_anions: Optional[List[str]] = None,
    max_workers: int = 4,
    save_pdf: bool = True,
    save_csv: bool = True,
    n_jobs_per_analysis: int = 2
) -> Dict[str, Any]:
    """
    Run analysis in parallel for multiple TM-anion systems.
    
    Args:
        api_key: Materials Project API key
        output_dir: Base directory to save all outputs
        config_path: Path to configuration file (optional)
        specific_tms: List of specific transition metals to analyze (overrides config)
        specific_anions: List of specific anions to analyze (overrides config)
        max_workers: Maximum number of parallel system analyses
        save_pdf: Whether to save PDF reports
        save_csv: Whether to save CSV data
        n_jobs_per_analysis: Number of parallel jobs for each analysis
        
    Returns:
        Dictionary containing summary results for all analyses
    """
    # Validate API key
    if not api_key:
        raise InvalidInputError("Materials Project API key is required")
    
    # Load configuration
    config = load_config(config_path)
    
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If specific elements are provided, use them instead of those in config
    if specific_tms or specific_anions:
        if config is None:
            config = {}
        if specific_tms:
            config['transition_metals'] = specific_tms
        if specific_anions:
            config['anions'] = specific_anions
    
    # Get all combinations of transition metals and anions
    systems = get_element_combinations(config)
    
    if not systems:
        raise InvalidInputError("No valid systems to analyze. Check your transition metals and anions selection.")
    
    logger.info(f"Preparing to analyze {len(systems)} TM-anion systems with {max_workers} workers")
    
    # Prepare summary object
    summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "transition_metals": config.get('transition_metals', []),
            "anions": config.get('anions', []),
            "max_workers": max_workers,
            "n_jobs_per_analysis": n_jobs_per_analysis
        },
        "results": {}
    }
    
    # Create a semaphore to limit MP API requests
    start_time = time.time()
    
    # Run analyses in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all analysis jobs
        futures = {}
        for system in systems:
            primary_tm, anion = system
            logger.info(f"Submitting analysis for {primary_tm}-{anion}")
            
            # Submit the job to the executor
            future = executor.submit(
                run_analysis,
                primary_tm=primary_tm,
                anion=anion,
                api_key=api_key,
                output_dir=output_dir,
                config=config,
                n_jobs=n_jobs_per_analysis,
                save_pdf=save_pdf,
                save_csv=save_csv
            )
            
            # Store the future with its system information
            futures[future] = (primary_tm, anion)
            
            # Add a small delay between submissions to avoid API rate limits
            time.sleep(1.0)  # Increased delay to better handle rate limits
        
        # Process results as they complete
        for future in as_completed(futures):
            primary_tm, anion = futures[future]
            system_name = f"{primary_tm}-{anion}"
            
            try:
                # Get the result
                result = future.result()
                logger.info(f"Completed analysis for {system_name}")
                
                # Add to summary
                summary["results"][system_name] = result
                
                # Write/update the summary file after each completion
                with open(output_dir / "batch_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Analysis failed for {system_name}: {e}")
                # Record the error in the summary
                summary["results"][system_name] = {
                    "status": "error",
                    "error": str(e),
                    "system": system_name
                }
                
                # Handle rate limit errors by adding longer delay
                if isinstance(e, APIRateLimitError):
                    logger.warning("Hit API rate limit, adding delay...")
                    time.sleep(5.0)
    
    # Calculate the total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Add final stats to summary
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["total_time_seconds"] = elapsed_time
    summary["completed_systems"] = len([r for r in summary["results"].values() 
                                       if r.get("status") == "completed"])
    summary["failed_systems"] = len([r for r in summary["results"].values() 
                                    if r.get("status") == "error"])
    
    # Save the final summary
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Batch analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")
    
    return summary

if __name__ == "__main__":
    # This script can be run directly for testing purposes
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch analysis of TM-anion systems")
    parser.add_argument("--api-key", required=True, help="Materials Project API key")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--tms", nargs="+", help="Specific transition metals to analyze")
    parser.add_argument("--anions", nargs="+", help="Specific anions to analyze")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel system analyses")
    parser.add_argument("--n-jobs", type=int, default=2, help="Number of parallel jobs for each analysis")
    parser.add_argument("--no-pdf", action="store_true", help="Do not save PDF reports")
    parser.add_argument("--no-csv", action="store_true", help="Do not save CSV data")
    
    args = parser.parse_args()
    
    run_batch_analysis(
        api_key=args.api_key,
        output_dir=Path(args.output_dir),
        config_path=args.config,
        specific_tms=args.tms,
        specific_anions=args.anions,
        max_workers=args.max_workers,
        save_pdf=not args.no_pdf,
        save_csv=not args.no_csv,
        n_jobs_per_analysis=args.n_jobs
    )