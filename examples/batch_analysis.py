#!/usr/bin/env python
"""
Example script demonstrating the use of the refactored cluster_finder
for analyzing multiple transition metal-anion systems in parallel.
"""
import os
import time
from pathlib import Path
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.logging import RichHandler

from cluster_finder.analysis.batch import run_batch_analysis
from cluster_finder.utils.config_utils import load_config, get_element_combinations
from cluster_finder.utils.async_utils import get_api_key
from cluster_finder.utils.logger import get_logger

# Create a dedicated console for this script with full output enabled
console = Console(highlight=True)

# Configure logging with rich handler for better output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)

# Get logger for this script
logger = get_logger("batch_analysis")

def example_batch_systems():
    """Example of analyzing multiple TM-anion systems in parallel."""
    # Define your Materials Project API key
    API_KEY = get_api_key()
    
    # Create an output directory
    output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load the default configuration
    config = load_config()
    
    # You can override specific elements if needed
    specific_tms = ["Nb", "V"]
    specific_anions = ["Cl"]

    config["transition_metals"] = specific_tms
    config["anions"] = specific_anions
    
    # Get all combinations (or you could define them manually)
    systems = get_element_combinations(config)
    
    # Format each system as "TM-anion" for display
    formatted_systems = [f"{tm}-{anion}" for tm, anion in systems]
    
    # Display analysis configuration
    console.print(Panel.fit(
        f"[bold]Systems to analyze:[/bold] {', '.join(formatted_systems)}\n"
        f"[bold]Output directory:[/bold] {output_dir}",
        title="[bold cyan]Batch Analysis Configuration[/bold cyan]",
        border_style="green"
    ))
    
    # Set the number of parallel workers
    max_workers = min(4, len(systems))  # Use at most 4 workers or number of systems
    
    # Run batch analysis
    start_time = time.time()
    
    try:
        # Run the batch analysis directly
        # This allows the internal logging to display without interference
        result = run_batch_analysis(
            api_key=API_KEY,
            output_dir=output_dir,
            specific_tms=specific_tms,
            specific_anions=specific_anions,
            max_workers=max_workers,    # Number of parallel system analyses
            n_jobs_per_analysis=2,      # Number of parallel jobs within each analysis
            save_pdf=True,
            save_csv=True,
            verbose=True  # Enable verbose mode
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        return None
    except Exception as e:
        console.print(f"\n[red]Error during analysis: {e}[/red]")
        return None
    
    elapsed_time = time.time() - start_time
    
    # Create a rich table for the summary
    summary_table = Table(title="[bold]Batch Analysis Summary[/bold]")
    summary_table.add_column("Parameter", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Systems", str(len(systems)))
    summary_table.add_row("Completed Systems", str(result.get("completed_systems", 0)))
    summary_table.add_row("Failed Systems", str(result.get("failed_systems", 0)))
    summary_table.add_row("Total Time", f"{elapsed_time:.2f} seconds")
    
    console.print("\n")
    console.print(summary_table)
    console.print("\n[bold cyan]System-specific Results:[/bold cyan]")
    
    # Create a table for system-specific results
    results_table = Table()
    results_table.add_column("System", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Compounds", justify="right")
    results_table.add_column("With Clusters", justify="right")
    results_table.add_column("Time (s)", justify="right")
    
    for system_name, system_result in result.get("results", {}).items():
        status = system_result.get("status", "unknown")
        status_style = "green" if status == "completed" else "red"
        
        results_table.add_row(
            system_name,
            f"[{status_style}]{status}[/{status_style}]",
            str(system_result.get("compounds_count", 0)),
            str(system_result.get("compounds_with_clusters_count", 0)),
            f"{system_result.get('time_taken', 0):.2f}"
        )
    
    console.print(results_table)
    
    # Final completion message
    console.print(Panel.fit(
        f"[bold green]Analysis complete![/bold green]\n"
        f"Detailed results saved to: [cyan]{output_dir / 'batch_summary.json'}[/cyan]",
        border_style="green"
    ))
    
    return result


if __name__ == "__main__":
    example_batch_systems()