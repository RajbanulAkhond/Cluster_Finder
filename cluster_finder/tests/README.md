# Testing Strategy for Cluster Finder

This directory contains the test suite for the Cluster Finder package.

## Approach to Testing

### Mocking External Dependencies

To maintain robustness against changes in external dependencies, many of our tests use mocking to isolate our code from external libraries. This is particularly important for tests involving:

1. `pymatgen` - Which undergoes frequent updates that can introduce breaking changes
2. `mp-api` - Which depends on external services and may have version compatibility issues with pymatgen

For example, in `test_simple_search.py`, we mock the functions from `simple_search.py` rather than calling them directly, avoiding potential import errors related to `SymmetryUndeterminedError` and other external dependencies.

### Example of Mocking Approach

```python
# Instead of importing directly:
# from cluster_finder.core.simple_search import find_trimers

# We mock and patch the function:
@patch('cluster_finder.cluster_finder.core.simple_search.find_trimers')
def test_function(mock_find_trimers):
    # Configure the mock
    mock_find_trimers.return_value = expected_result
    
    # Import locally to avoid global import issues
    from cluster_finder.cluster_finder.core.simple_search import find_trimers
    
    # Test using the mock
    result = find_trimers(test_input)
    assert result == expected_result
```

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest test_simple_search.py

# Run with coverage
pytest --cov=cluster_finder
```

## Dependency Notes

If you encounter test failures related to library compatibility, please ensure you have the compatible versions installed as specified in the main `requirements.txt` file or use the test environment:

```bash
pip install -e ".[dev]"
```

This approach ensures tests can run even if APIs or function signatures change in external dependencies. 