"""
Logger configuration for the cluster_finder package.
"""
import os
import sys
import logging
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

# Create console
console = Console()

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the package.
    
    Args:
        verbose (bool): Enable verbose logging (debug-level and INFO messages)
        log_file (str, optional): Path to log file
    """
    # In quiet mode, only show ERROR and above for console output
    # But maintain INFO level for the logger itself to allow child processes to log
    level = logging.INFO if verbose else logging.ERROR
    
    # Configure root logger to always accept INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set console quiet mode based on verbose flag
    console.quiet = not verbose
    
    # Create a separate console for logging to prevent overlap with status displays
    log_console = Console(stderr=True)  # Use stderr for logs
    
    # Configure the console handler with the user-specified level
    console_handler = RichHandler(
        console=log_console,  # Use separate console for logs
        rich_tracebacks=True,
        show_time=verbose,
        show_path=verbose,
        markup=True,
        log_time_format="[%X]" if verbose else None,
        omit_repeated_times=True,
        level=level
    )
    root_logger.addHandler(console_handler)
    
    # File handler if specified (this will always have detailed logs)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        
    # In non-verbose mode, suppress common verbose outputs
    if not verbose:
        # Suppress progress bars
        os.environ['TQDM_DISABLE'] = '1'
        
        # Suppress INFO logs from common verbose modules
        for module in ['urllib3', 'matplotlib', 'pymatgen', 'joblib', 'paramiko']:
            logging.getLogger(module).setLevel(logging.WARNING)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name or 'cluster_finder')

# Configure default logging and create the default logger instance
setup_logging()
logger = get_logger('cluster_finder')