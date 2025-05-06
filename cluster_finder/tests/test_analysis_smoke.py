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

@pytest.mark.parametrize("primary_tm,anion", [
    ("Nb", "Cl"),
    ("V", "Cl")
])
def test_run_analysis_smoke(mock_mp_data, primary_tm, anion):
    """
    Smoke test for run_analysis function with different element combinations.
    
    This test mocks the Materials Project API calls to test the analysis function
    without making real network requests.
    """
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Mock the API calls
        with patch("cluster_finder.utils.helpers.search_transition_metal_compounds") as mock_search, \
             patch("cluster_finder.utils.helpers.get_mp_properties_batch") as mock_properties:
            
            # Set up mock return values
            mock_search.return_value = [
                compound for compound in mock_mp_data["compounds"] 
                if primary_tm in compound["formula"] and anion in compound["formula"]
            ]
            
            mock_properties.return_value = mock_mp_data["properties"]
            
            # Create a simple config for testing
            test_config = {
                "element_filters": {"min_elements": 2, "max_elements": 4},
                "property_filters": {"min_magnetization": 0.01, "max_magnetization": 5.0},
                "mp_properties": ["energy_above_hull", "formation_energy_per_atom", "band_gap", "total_magnetization"],
                "analysis_params": {"top_n_compounds": 1}
            }
            
            # Run the analysis
            result = run_analysis(
                primary_tm=primary_tm,
                anion=anion,
                api_key="test_api_key",
                output_dir=output_dir,
                config=test_config,
                n_jobs=1,
                save_pdf=False,  # Skip PDF generation for faster testing
                save_csv=True
            )
            
            # Verify the result contains the expected keys
            assert "status" in result
            assert result["status"] == "completed"
            assert "system" in result
            assert result["system"] == f"{primary_tm}-{anion}"
            
            # Check for expected output files in the system directory
            system_dir = output_dir / f"{primary_tm}-{anion}"
            assert system_dir.exists()
            
            # Check if a log file was created
            log_files = list(system_dir.glob("*.log"))
            assert len(log_files) == 1
            
            # If CSV saving was enabled, check for the CSV file
            if "compounds_with_clusters_count" in result and result["compounds_with_clusters_count"] > 0:
                csv_file = system_dir / f"{primary_tm}-{anion}_analysis_results_summary.csv"
                assert csv_file.exists() or result["compounds_with_clusters_count"] == 0

def test_batch_analysis_smoke(mock_mp_data):
    """
    Smoke test for the batch analysis functionality.
    
    This test verifies that the batch processing correctly handles multiple systems
    and aggregates results.
    """
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Use a simplified config for testing
        test_config = {
            "transition_metals": ["Nb", "V"],
            "anions": ["Cl"],
            "element_filters": {"min_elements": 2, "max_elements": 4},
            "property_filters": {"min_magnetization": 0.01, "max_magnetization": 5.0},
            "mp_properties": ["energy_above_hull", "formation_energy_per_atom"],
            "analysis_params": {"top_n_compounds": 1}
        }
        
        # Mock the run_analysis function to avoid running the actual analysis
        with patch("cluster_finder.analysis.batch.run_analysis") as mock_run_analysis:
            # Set up mock return values
            mock_run_analysis.side_effect = lambda **kwargs: {
                "status": "completed",
                "system": f"{kwargs['primary_tm']}-{kwargs['anion']}",
                "compounds_count": 1,
                "compounds_with_clusters_count": 1,
                "time_taken": 0.5,
                "output_dir": str(kwargs['output_dir'] / f"{kwargs['primary_tm']}-{kwargs['anion']}"),
                "outputs": {
                    "pdf": None,
                    "summary_csv": str(kwargs['output_dir'] / f"{kwargs['primary_tm']}-{kwargs['anion']}" / 
                                     f"{kwargs['primary_tm']}-{kwargs['anion']}_analysis_results_summary.csv")
                }
            }
            
            # Run the batch analysis
            result = run_batch_analysis(
                api_key="test_api_key",
                output_dir=output_dir,
                config=test_config,
                max_workers=2,
                save_pdf=False,
                save_csv=True,
                n_jobs_per_analysis=1
            )
            
            # Verify the batch executed correctly
            assert "results" in result
            assert len(result["results"]) == 2  # Nb-Cl and V-Cl
            assert "Nb-Cl" in result["results"]
            assert "V-Cl" in result["results"]
            
            # Check that the summary file was created
            summary_file = output_dir / "batch_summary.json"
            assert summary_file.exists()
            
            # Verify the correct arguments were passed to run_analysis
            assert mock_run_analysis.call_count == 2
            
            call_args = [call[1] for call in mock_run_analysis.call_args_list]
            systems_called = [(args["primary_tm"], args["anion"]) for args in call_args]
            assert ("Nb", "Cl") in systems_called
            assert ("V", "Cl") in systems_called

if __name__ == "__main__":
    # This enables running the test with python -m tests.test_analysis
    pytest.main(["-v", __file__])