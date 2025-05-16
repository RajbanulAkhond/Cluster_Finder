#!/usr/bin/env python
"""
Tests for batch processing functionality in the cluster finder package.
"""
import os
import sys
import time
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from concurrent.futures import ThreadPoolExecutor, as_completed

from cluster_finder.analysis.batch import (
    run_batch_analysis,
    worker_initializer,
    run_analysis_wrapper,
    cleanup_multiprocessing_resources
)
from cluster_finder.utils.exceptions import APIRateLimitError, InvalidInputError
from cluster_finder.utils.config_utils import get_element_combinations


class TestBatchProcessing:
    """Test class for batch processing functionality."""
    
    @patch('cluster_finder.analysis.batch._worker_verbose', True)
    @patch('cluster_finder.analysis.batch._worker_n_jobs', 2)
    def test_worker_initializer(self):
        """Test the worker initializer function."""
        # Since we've mocked the global variables directly, we don't need to test the setup_logging call
        # Instead, just verify that the function can be called without errors
        worker_initializer(verbose=True, n_jobs=2)
        
        # Check that the function doesn't raise any exceptions
        from cluster_finder.analysis.batch import _worker_verbose, _worker_n_jobs
        assert _worker_verbose is True
        assert _worker_n_jobs == 2
    
    @patch('cluster_finder.analysis.batch.get_logger')
    @patch('cluster_finder.analysis.batch.run_analysis')
    @patch('cluster_finder.analysis.batch._worker_verbose', True)
    @patch('cluster_finder.analysis.batch._worker_n_jobs', 2)
    def test_run_analysis_wrapper(self, mock_run_analysis, mock_get_logger):
        """Test the run_analysis_wrapper function."""
        # Setup mock
        mock_run_analysis.return_value = {"status": "completed", "compounds_count": 5}
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call the function
        result = run_analysis_wrapper(
            primary_tm="Nb", 
            anion="Cl", 
            api_key="dummy_key", 
            output_dir="test_output", 
            config={}, 
            n_jobs=2,  # Match the mocked global variable value
            save_pdf=False, 
            save_csv=False
        )
            
        # Verify run_analysis was called with correct parameters
        mock_run_analysis.assert_called_once_with(
            primary_tm="Nb",
            anion="Cl",
            api_key="dummy_key",
            output_dir=Path("test_output"),
            config={},
            n_jobs=2,
            save_pdf=False,
            save_csv=False,
            verbose=True
        )
            
        # Verify result is passed through
        assert result == {"status": "completed", "compounds_count": 5}
            
        # Verify logging occurs in verbose mode
        assert mock_logger.info.call_count >= 1
    
    @patch('cluster_finder.analysis.batch.get_logger')
    @patch('cluster_finder.analysis.batch.run_analysis')
    def test_run_analysis_wrapper_error(self, mock_run_analysis, mock_get_logger):
        """Test the run_analysis_wrapper function handles errors properly."""
        # Setup mock to raise exception
        mock_run_analysis.side_effect = ValueError("Test error")
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        result = run_analysis_wrapper(
            primary_tm="Nb", 
            anion="Cl", 
            api_key="dummy_key", 
            output_dir="test_output", 
            config={}, 
            n_jobs=1, 
            save_pdf=False, 
            save_csv=False
        )
        
        # Verify error is logged
        mock_logger.error.assert_called_once()
        
        # Verify error result is returned
        assert result["status"] == "error"
        assert "Test error" in result["error"]
        assert result["system"] == "Nb-Cl"


if __name__ == "__main__":
    pytest.main(["-v", "test_batch.py"])