"""
Logger configuration for the cluster_finder package.
"""
import os
import sys
import logging
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the package.
    
    Args:
        verbose (bool): Enable verbose logging (debug-level and INFO messages)
        log_file (str, optional): Path to log file
    """
    # By default, use WARNING level (quiet mode)
    # If verbose is True, use INFO level
    level = logging.INFO if verbose else logging.WARNING
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Rich handler format - simpler format for non-verbose mode
    rich_format = "%(message)s" if verbose else "[%(levelname)s] %(message)s"
    file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set console quiet mode based on verbose flag
    console.quiet = not verbose
    
    # Suppress progress bars in non-verbose mode
    if not verbose:
        os.environ['TQDM_DISABLE'] = '1'
    else:
        os.environ.pop('TQDM_DISABLE', None)
    
    # Console handler with rich formatting (this will handle all console output)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=verbose,
        show_path=verbose,
        markup=True,
        log_time_format="[%X]" if verbose else None,
        omit_repeated_times=True
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(rich_format))
    root_logger.addHandler(console_handler)
    
    # File handler if specified (this will always have detailed logs)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always keep detailed logs in file
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name (str, optional): Logger name. If None, returns the root logger.
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

# Configure default logging and create the default logger instance
setup_logging()
logger = get_logger('cluster_finder')