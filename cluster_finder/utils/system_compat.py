"""
System compatibility utilities for cluster_finder.
This module provides utility functions to check for system-level dependencies
and handle cases where they might be missing.
"""
import sys
import platform
from typing import Optional
import logging

# Import the console and logger from the logger module
from .logger import console, get_logger

# Get a logger instance for this module
logger = get_logger('cluster_finder.system_compat')

def has_lzma_support() -> bool:
    """
    Check if LZMA compression is supported in this Python installation.
    
    Returns:
        bool: True if LZMA is supported, False otherwise
    """
    try:
        import lzma
        return True
    except ImportError:
        return False

def has_distro_package() -> bool:
    """
    Check if the 'distro' package is available.
    
    Returns:
        bool: True if distro package is available, False otherwise
    """
    try:
        import distro
        return True
    except ImportError:
        return False

def get_linux_distro() -> str:
    """
    Attempt to detect the Linux distribution.
    
    Returns:
        str: Linux distribution ID or empty string if detection fails
    """
    linux_distro = ""
    # First try with distro package
    if has_distro_package():
        try:
            import distro
            linux_distro = distro.id()
            return linux_distro
        except Exception as e:
            logger.debug(f"Error detecting Linux distribution with distro package: {e}")
    
    # Fallback to reading /etc/os-release
    try:
        with open('/etc/os-release') as f:
            for line in f:
                if line.startswith('ID='):
                    linux_distro = line.split('=')[1].strip().strip('"')
                    break
    except Exception as e:
        logger.debug(f"Error reading /etc/os-release: {e}")
    
    return linux_distro

def check_dependency_support(verbose: bool = True) -> bool:
    """
    Check all system dependencies required by cluster_finder.
    
    Args:
        verbose: Whether to print warning messages for missing dependencies
        
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    all_supported = True
    
    # Check LZMA support
    if not has_lzma_support():
        all_supported = False
        if verbose:
            warn_lzma_missing()
    
    return all_supported

def warn_lzma_missing() -> None:
    """Display a warning about missing LZMA support with installation instructions using Rich formatting."""
    system = platform.system()
    
    # Use console.print for rich formatting with warning style
    console.print("\n[bold yellow]⚠️  Warning: Python LZMA support is missing![/bold yellow]")
    console.print("[yellow]This may cause errors when using compression features required by dependencies like pymatgen.[/yellow]")
    console.print("")
    console.print("[bold]To fix this issue:[/bold]")
    
    if system == "Darwin":  # macOS
        console.print("1. Install xz: [cyan]brew install xz[/cyan]")
    elif system == "Linux":
        # Get the Linux distribution using the new helper function
        linux_distro = get_linux_distro()
        
        if linux_distro.lower() in ['ubuntu', 'debian']:
            console.print("1. Install liblzma-dev: [cyan]sudo apt-get install liblzma-dev[/cyan]")
        elif linux_distro.lower() in ['centos', 'rhel', 'fedora']:
            console.print("1. Install xz-devel: [cyan]sudo yum install xz-devel[/cyan]")
        else:
            console.print("1. Install the LZMA development package for your distribution")
    elif system == "Windows":
        console.print("1. Reinstall Python using the official installer from [link]python.org[/link]")
        console.print("   Make sure to select the option to install all optional features")
    
    # Check if using pyenv
    if 'pyenv' in sys.executable:
        python_version = ".".join(map(str, sys.version_info[:3]))
        console.print("")
        if system == "Darwin":  # macOS
            console.print("[bold]For pyenv users:[/bold]")
            console.print(f"  [cyan]LDFLAGS=\"-L$(brew --prefix xz)/lib\" CPPFLAGS=\"-I$(brew --prefix xz)/include\" pyenv install {python_version}[/cyan]")
        else:
            console.print("[bold]For pyenv users: Reinstall Python with LZMA support[/bold]")
            console.print(f"  [cyan]pyenv install {python_version}[/cyan]")
    
    console.print("")
    console.print("2. After installing the system requirements, reinstall cluster-finder")
    console.print("")
    
    # Also log this to the logger for file logging if configured
    logger.warning("Python LZMA support is missing. This may cause errors with dependencies like pymatgen.")