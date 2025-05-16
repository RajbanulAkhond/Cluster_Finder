#!/usr/bin/env python
"""
Batch processing module for running multiple TM-anion system analyses in parallel.
"""
import os
import time
import json
import logging
import logging.handlers
import sys
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table

from cluster_finder.utils.config_utils import load_config, get_element_combinations
from cluster_finder.utils.logger import get_logger, setup_logging
from cluster_finder.analysis.analysis import run_analysis
from cluster_finder.utils.exceptions import APIRateLimitError, InvalidInputError

# Get the logger without configuring it (will inherit from root logger)
logger = get_logger('cluster_finder.batch')
console = Console()

# Global variable for worker verbosity
_worker_verbose = False
_worker_n_jobs = 1

def worker_initializer(verbose=False, n_jobs=1):
    """Initialize worker processes with proper logging configuration."""
    # Use the existing rich logger configuration from the logger module
    from cluster_finder.utils.logger import setup_logging
    global _worker_verbose, _worker_n_jobs
    _worker_verbose = verbose
    _worker_n_jobs = n_jobs
    
    # Configure logging with the correct verbosity
    setup_logging(verbose=verbose)
    
    # Get the worker logger and configure it
    worker_logger = logging.getLogger('cluster_finder')
    # Always show errors, but only show info/debug in verbose mode
    worker_logger.setLevel(logging.INFO if verbose else logging.ERROR)
    
    if verbose:
        worker_logger.info(f"Worker process {multiprocessing.current_process().name} initialized")

# Simple function for multiprocessing to avoid pickling issues
def run_analysis_wrapper(primary_tm, anion, api_key, output_dir, config, n_jobs, save_pdf, save_csv):
    """Wrapper function for run_analysis to avoid pickling issues with multiprocessing."""
    try:
        # Get a properly configured logger for this worker process
        process_logger = get_logger(f"cluster_finder.worker.{primary_tm}-{anion}")
        
        # Convert Path object to string for serialization
        if isinstance(output_dir, Path):
            output_dir = str(output_dir)
            
        # Call the actual analysis function with proper verbosity
        result = run_analysis(
            primary_tm=primary_tm,
            anion=anion,
            api_key=api_key,
            output_dir=Path(output_dir),
            config=config,
            n_jobs=_worker_n_jobs,
            save_pdf=save_pdf,
            save_csv=save_csv,
            verbose=_worker_verbose  # Pass through verbosity setting
        )
        
        if _worker_verbose:
            process_logger.info(f"Completed analysis for {primary_tm}-{anion} system with {result.get('compounds_with_clusters_count', 0)} compounds with clusters")
        return result
        
    except Exception as e:
        process_logger = get_logger(f"cluster_finder.worker.{primary_tm}-{anion}")
        process_logger.error(f"Error in analysis for {primary_tm}-{anion}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "system": f"{primary_tm}-{anion}"
        }

def cleanup_multiprocessing_resources(logger):
    """Clean up multiprocessing resources and temporary files."""
    import tempfile
    import shutil
    import glob
    import gc
    import sys
    import multiprocessing
    import os
    import threading
    import queue
    import warnings
    import time
    import signal

    # Use a queue to track completion
    cleanup_done = queue.Queue()

    def cleanup_worker():
        try:
            # Temporarily suppress resource tracker warnings
            warnings.filterwarnings("ignore", category=UserWarning, 
                                 module="joblib.externals.loky.backend.resource_tracker")
            
            # First clean up loky backend resources
            try:
                from joblib.externals.loky import get_reusable_executor
                executor = get_reusable_executor(timeout=1)
                executor.shutdown(wait=True, kill_workers=True)
            except Exception:
                pass

            # Clean up temp folders with retry mechanism
            patterns = [
                os.path.join(tempfile.gettempdir(), "joblib_memmapping_folder_*"),
                os.path.join(tempfile.gettempdir(), "loky-*"),
                os.path.join("/tmp", "joblib_memmapping_folder_*"),
                os.path.join("/tmp", "loky-*")
            ]
            
            for pattern in patterns:
                for folder in glob.glob(pattern):
                    if os.path.exists(folder):
                        for attempt in range(3):
                            try:
                                shutil.rmtree(folder, ignore_errors=True)
                                if not os.path.exists(folder):
                                    break
                            except Exception:
                                time.sleep(0.5 * (2 ** attempt))

            # Explicitly clean up joblib resources
            try:
                from joblib.parallel import Parallel, parallel_backend
                Parallel()._managed_pool = None
                with parallel_backend('threading', n_jobs=1):
                    pass
            except Exception:
                pass
                
            # Force terminate any remaining child processes
            active_children = multiprocessing.active_children()
            for child in active_children:
                try:
                    child.terminate()
                    child.join(timeout=0.5)
                except:
                    # If join times out, force kill
                    if sys.platform != 'win32':  # POSIX systems
                        try:
                            os.kill(child.pid, signal.SIGKILL)
                        except:
                            pass

            # Clean up multiprocessing resources
            try:
                multiprocessing.util._cleanup()
            except:
                pass

            # Handle resource tracker
            if hasattr(multiprocessing, 'resource_tracker'):
                tracker_mod = multiprocessing.resource_tracker
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
                    except Exception:
                        pass

            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Signal completion
            cleanup_done.put(True)
            
        except Exception as cleanup_err:
            logger.debug(f"Non-critical cleanup error: {cleanup_err}")
            cleanup_done.put(False)

    # Start cleanup in a separate thread with a timeout
    cleanup_thread = threading.Thread(target=cleanup_worker)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    try:
        # Wait up to 5 seconds for cleanup
        cleanup_done.get(timeout=5.0)
    except queue.Empty:
        logger.debug("Cleanup timed out, but continuing execution")

def run_batch_analysis(
    api_key: str,
    output_dir: Path,
    config_path: Optional[str] = None,
    specific_tms: Optional[List[str]] = None,
    specific_anions: Optional[List[str]] = None,
    max_workers: int = 4,
    save_pdf: bool = True,
    save_csv: bool = True,
    n_jobs_per_analysis: int = 2,
    verbose: bool = False,
    use_mpi: bool = False,
    mpi_cores: Optional[int] = None
) -> Dict[str, Any]:
    """Run batch analysis on multiple TM-anion systems."""
    from ..utils.helpers import kill_resource_tracker
    
    # Clean up any existing resources before starting
    kill_resource_tracker()
    
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
    
    # Configure multiprocessing if requested
    if use_mpi:
        # Determine number of cores to use
        available_cores = multiprocessing.cpu_count()
        if mpi_cores is None:
            actual_cores = available_cores
        else:
            actual_cores = min(mpi_cores, available_cores)
        
        # Adjust workers based on number of cores and systems
        max_workers = min(actual_cores, len(systems))
        console.print(f"Using multiprocessing with {max_workers} cores")
    else:
        console.print(f"Using threading with {max_workers} workers")
    
    # Always show the number of systems to be analyzed
    console.print(f"Analyzing {len(systems)} TM-anion systems")
    
    # Prepare summary object
    summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "transition_metals": config.get('transition_metals', []),
            "anions": config.get('anions', []),
            "max_workers": max_workers,
            "n_jobs_per_analysis": n_jobs_per_analysis,
            "use_multiprocessing": use_mpi
        },
        "results": {}
    }
    
    start_time = time.time()
    
    # Set worker verbosity
    global _worker_verbose
    _worker_verbose = verbose
    
    # Choose the executor type based on the mpi flag
    ExecutorClass = ProcessPoolExecutor if use_mpi else ThreadPoolExecutor
    
    # Run analyses in parallel
    executor = None
    try:
        # Create the executor with worker initialization for multiprocessing
        if use_mpi:
            executor = ExecutorClass(
                max_workers=max_workers, 
                initializer=worker_initializer,
                initargs=(verbose, n_jobs_per_analysis)
            )
        else:
            executor = ExecutorClass(max_workers=max_workers)
            # For threading, configure logging in the main process
            setup_logging(verbose=verbose)
            
        # Submit all analysis jobs
        futures = {}
        for system in systems:
            primary_tm, anion = system
            if verbose:
                logger.info(f"Submitting analysis for {primary_tm}-{anion}")
            
            if use_mpi:
                future = executor.submit(
                    run_analysis_wrapper,
                    primary_tm,
                    anion,
                    api_key,
                    str(output_dir),  # Convert Path to string for serialization
                    config,
                    n_jobs_per_analysis,
                    save_pdf,
                    save_csv
                )
            else:
                future = executor.submit(
                    run_analysis_wrapper,
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
            time.sleep(1.0)
        
        # Process all results
        try:
            # In verbose mode, don't use the progress bar to avoid conflicts with logger output
            if verbose:
                # Process results without a progress bar in verbose mode
                for future in as_completed(futures):
                    primary_tm, anion = futures[future]
                    system_name = f"{primary_tm}-{anion}"
                    
                    try:
                        # Get the result
                        result = future.result()
                        # Show completion message
                        console.print(f"[green]✓ Completed: {system_name} - Found {result.get('compounds_with_clusters_count', 0)} compounds with clusters[/green]")
                        
                        # Add to summary
                        summary["results"][system_name] = result
                        
                        # Write/update the summary file after each completion
                        with open(output_dir / "batch_summary.json", "w") as f:
                            json.dump(summary, f, indent=2)
                            
                    except Exception as e:
                        error_msg = str(e)
                        # Show error message
                        console.print(f"[red]✗ Failed: {system_name} - {error_msg}[/red]")
                        
                        # Record the error in the summary
                        summary["results"][system_name] = {
                            "status": "error",
                            "error": error_msg,
                            "system": system_name
                        }
                        
                        # Handle rate limit errors by adding longer delay
                        if isinstance(e, APIRateLimitError):
                            console.print("[yellow]Hit API rate limit, adding delay...[/yellow]")
                            time.sleep(5.0)
            else:
                # Use the progress bar in non-verbose mode
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                    refresh_per_second=10
                ) as progress:
                    # Add a task for overall progress
                    task = progress.add_task(f"Processing {len(systems)} systems", total=len(systems))
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        primary_tm, anion = futures[future]
                        system_name = f"{primary_tm}-{anion}"
                        
                        try:
                            # Get the result
                            result = future.result()
                            
                            # Show completion message
                            console.print(f"[green]✓ Completed: {system_name} - Found {result.get('compounds_with_clusters_count', 0)} compounds with clusters[/green]")
                            
                            # Add to summary
                            summary["results"][system_name] = result
                            
                            # Write/update the summary file after each completion
                            with open(output_dir / "batch_summary.json", "w") as f:
                                json.dump(summary, f, indent=2)
                                
                        except Exception as e:
                            error_msg = str(e)
                            # Show error message
                            console.print(f"[red]✗ Failed: {system_name} - {error_msg}[/red]")
                            
                            # Record the error in the summary
                            summary["results"][system_name] = {
                                "status": "error",
                                "error": error_msg,
                                "system": system_name
                            }
                            
                            # Handle rate limit errors by adding longer delay
                            if isinstance(e, APIRateLimitError):
                                console.print("[yellow]Hit API rate limit, adding delay...[/yellow]")
                                time.sleep(5.0)
                        
                        # Update progress
                        progress.update(task, advance=1)
        # Make sure to catch keyboard interrupts to allow proper cleanup
        except KeyboardInterrupt:
            console.print("\n[yellow]Analysis interrupted by user. Cleaning up...[/yellow]")
            
            # Cancel any pending futures
            for future in futures:
                if not future.done():
                    future.cancel()
            
            # Wait for a short time for active tasks to complete or be cancelled
            wait(futures, timeout=5)
            
            # Update summary with partial results
            summary["status"] = "interrupted"
            summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed_time = time.time() - start_time
            summary["total_time_seconds"] = elapsed_time
            summary["completed_systems"] = len([r for r in summary["results"].values() 
                                               if r.get("status") == "completed"])
            summary["failed_systems"] = len([r for r in summary["results"].values() 
                                            if r.get("status") == "error"])
            
            # Save the partial summary
            with open(output_dir / "batch_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
            console.print(f"[yellow]Partial results saved to {output_dir}/batch_summary.json[/yellow]")
            raise
            
    finally:
        try:
            # Clean up multiprocessing resources
            if use_mpi and executor is not None:
                logger.info("Cleaning up multiprocessing resources...")
                
                # Cancel any pending futures first
                for future in futures:
                    if not future.done():
                        future.cancel()
                
                # Force immediate shutdown without waiting
                executor.shutdown(wait=False)
                
                # Clean up all multiprocessing resources using the dedicated function
                cleanup_multiprocessing_resources(logger)
                
            elif executor is not None:
                # For thread executor, just shutdown normally
                executor.shutdown(wait=False)
            
            # Calculate and add final stats to summary
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Add final stats to summary
            summary["status"] = "completed"
            summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            summary["total_time_seconds"] = elapsed_time
            summary["completed_systems"] = len([r for r in summary["results"].values() 
                                               if r.get("status") == "completed"])
            summary["failed_systems"] = len([r for r in summary["results"].values() 
                                            if r.get("status") == "error"])
            
            # Save the final summary
            with open(output_dir / "batch_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            # Show completion message
            console.print(f"\n[bold green]Batch analysis completed in {elapsed_time:.2f} seconds[/bold green]")
            console.print(f"[bold]Results saved to {output_dir}[/bold]\n")
            
            # Kill any remaining resource tracker processes at the very end
            from cluster_finder.utils.helpers import kill_resource_tracker
            kill_resource_tracker()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
    
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
    parser.add_argument("--mpi", nargs="?", const=-1, type=int, help="Enable multiprocessing with the specified number of cores (use all available if no value provided)")
    parser.add_argument("--no-pdf", action="store_true", help="Do not save PDF reports")
    parser.add_argument("--no-csv", action="store_true", help="Do not save CSV data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed logging information")
    
    args = parser.parse_args()
    
    # Handle MPI argument
    use_mpi = args.mpi is not None
    mpi_cores = None if args.mpi == -1 else args.mpi if args.mpi > 0 else None
    
    run_batch_analysis(
        api_key=args.api_key,
        output_dir=Path(args.output_dir),
        config_path=args.config,
        specific_tms=args.tms,
        specific_anions=args.anions,
        max_workers=args.max_workers,
        save_pdf=not args.no_pdf,
        save_csv=not args.no_csv,
        n_jobs_per_analysis=args.n_jobs,
        verbose=args.verbose,
        use_mpi=use_mpi,
        mpi_cores=mpi_cores
    )