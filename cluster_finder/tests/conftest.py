"""
Configuration for pytest fixtures.

This module contains fixtures and utilities for testing the cluster_finder package.
"""

import os
import pytest
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.periodic_table import Element

# Define directories
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


@pytest.fixture
def simple_cubic_structure():
    """
    Create a simple cubic structure with Fe atoms.
    
    Returns:
        Structure: A simple cubic structure with Fe atoms
    """
    # Create a simple cubic lattice with Fe atoms
    lattice = Lattice.cubic(3.0)  # 3 Å cube
    
    # Create a structure with Fe atoms at corners
    structure = Structure(
        lattice=lattice,
        species=['Fe', 'Fe', 'Fe', 'Fe', 'Fe', 'Fe', 'Fe', 'Fe'],
        coords=[
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ],
        coords_are_cartesian=False
    )
    
    return structure


@pytest.fixture
def binary_structure():
    """
    Create a binary structure with Fe and Co atoms.
    
    Returns:
        Structure: A binary structure with Fe and Co atoms
    """
    # Create a simple cubic lattice
    lattice = Lattice.cubic(3.0)  # 3 Å cube
    
    # Create a structure with Fe and Co atoms
    structure = Structure(
        lattice=lattice,
        species=['Fe', 'Co', 'Fe', 'Co', 'Fe', 'Co', 'Fe', 'Co'],
        coords=[
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ],
        coords_are_cartesian=False
    )
    
    return structure


@pytest.fixture
def connectivity_matrix():
    """
    Create a sample connectivity matrix.
    
    Returns:
        tuple: (connectivity matrix, transition metal indices)
    """
    # Create a simple connectivity matrix for 4 atoms
    # where atom 0 connects to 1 and 2, and atom 2 connects to 3
    matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    
    tm_indices = [0, 1, 2, 3]
    
    return matrix, tm_indices


@pytest.fixture
def sample_cluster(simple_cubic_structure):
    """
    Create a sample cluster of atoms.
    
    Returns:
        list: A list of sites representing a cluster
    """
    # Create a cluster with 3 atoms
    return [
        simple_cubic_structure.sites[0],
        simple_cubic_structure.sites[1],
        simple_cubic_structure.sites[2]
    ]


@pytest.fixture
def transition_metals():
    """
    Returns a list of transition metal symbols.
    
    Returns:
        list: List of transition metal symbols
    """
    return ["Fe", "Co", "Ni", "Cu"] 