#!/usr/bin/env python
"""
Configuration utilities for the cluster_finder package.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

def get_config_path() -> Path:
    """Get the path to the default configuration file."""
    # Get the directory where this file is located
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # Go up one level to the cluster_finder directory, then to config
    config_dir = current_dir.parent / "config"
    config_file = config_dir / "system_config.yaml"
    return config_file

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config file.
        
    Returns:
        Dictionary containing the configuration.
    """
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_element_combinations(config: Optional[Dict[str, Any]] = None) -> List[List[str]]:
    """
    Get all combinations of transition metals and anions from the configuration.
    
    Args:
        config: Configuration dictionary. If None, loads from the default config file.
        
    Returns:
        List of element combinations, where each combination is a list [transition_metal, anion].
    """
    if config is None:
        config = load_config()
    
    transition_metals = config.get('transition_metals', [])
    anions = config.get('anions', [])
    
    combinations = []
    for tm in transition_metals:
        for anion in anions:
            combinations.append([tm, anion])
    
    return combinations