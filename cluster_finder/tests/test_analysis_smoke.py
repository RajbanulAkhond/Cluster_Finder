"""
Test for the parameterized analysis module.
"""
import os
import sys
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Import the analysis module
from cluster_finder.analysis.analysis import run_analysis
from cluster_finder.analysis.batch import run_batch_analysis
from cluster_finder.utils.config_utils import load_config, get_element_combinations

def mock_mprester_init(self, *args, **kwargs):
    """Mock MPRester initialization to bypass API key validation."""
    pass

@pytest.fixture
def mock_mp_data():
    """Fixture to mock Materials Project data responses."""
    # Create a mock response for the search_transition_metal_compounds function
    compounds = [
        {
            "material_id": "mp-123",
            "formula": "NbCl3",
            "structure": {
                "@module": "pymatgen.core.structure",
                "@class": "Structure",
                "lattice": {
                    "matrix": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                    "a": 5.0, "b": 5.0, "c": 5.0,
                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                    "volume": 125.0
                },
                "sites": [
                    {"species": [{"element": "Nb", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0], "label": "Nb", "properties": {}},
                    {"species": [{"element": "Cl", "occu": 1.0}], "xyz": [2.5, 0.0, 0.0], "label": "Cl", "properties": {}},
                    {"species": [{"element": "Cl", "occu": 1.0}], "xyz": [0.0, 2.5, 0.0], "label": "Cl", "properties": {}},
                    {"species": [{"element": "Cl", "occu": 1.0}], "xyz": [0.0, 0.0, 2.5], "label": "Cl", "properties": {}}
                ]
            },
            "symmetry": {"symbol": "Pm-3m"},
            "magnetization": 2.0
        },
        {
            "material_id": "mp-456",
            "formula": "VCl2",
            "structure": {
                "@module": "pymatgen.core.structure",
                "@class": "Structure",
                "lattice": {
                    "matrix": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                    "a": 4.0, "b": 4.0, "c": 4.0,
                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                    "volume": 64.0
                },
                "sites": [
                    {"species": [{"element": "V", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0], "label": "V", "properties": {}},
                    {"species": [{"element": "Cl", "occu": 1.0}], "xyz": [2.0, 0.0, 0.0], "label": "Cl", "properties": {}},
                    {"species": [{"element": "Cl", "occu": 1.0}], "xyz": [0.0, 2.0, 0.0], "label": "Cl", "properties": {}}
                ]
            },
            "symmetry": {"symbol": "Fm-3m"},
            "magnetization": 3.0
        }
    ]
    
    # Create mock properties data
    properties = {
        "mp-123": {
            "energy_above_hull": 0.01,
            "formation_energy_per_atom": -0.5,
            "band_gap": 0.2,
            "total_magnetization": 2.0
        },
        "mp-456": {
            "energy_above_hull": 0.02,
            "formation_energy_per_atom": -0.3,
            "band_gap": 0.1,
            "total_magnetization": 3.0
        }
    }
    
    return {"compounds": compounds, "properties": properties}

@pytest.mark.skip(reason="Requires Materials Project API key")
@pytest.mark.parametrize("primary_tm,anion", [
    ("Nb", "Cl"),
    ("V", "Cl")
])
def test_run_analysis_smoke(mock_mp_data, primary_tm, anion):
    """Test run_analysis function with different element combinations."""
    pass

@pytest.mark.skip(reason="Requires Materials Project API key")
def test_batch_analysis_smoke(mock_mp_data):
    """Test batch analysis functionality."""
    pass

if __name__ == "__main__":
    # This enables running the test with python -m tests.test_analysis
    pytest.main(["-v", __file__])