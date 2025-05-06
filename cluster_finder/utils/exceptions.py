"""
Custom exceptions for the cluster_finder package.
"""

class ClusterFinderError(Exception):
    """Base exception class for cluster_finder package."""
    pass

class APIKeyError(ClusterFinderError):
    """Raised when there are issues with the Materials Project API key."""
    pass

class APIRateLimitError(ClusterFinderError):
    """Raised when Materials Project API rate limit is exceeded."""
    pass

class ConfigurationError(ClusterFinderError):
    """Raised when there is an error in configuration."""
    pass

class DataProcessingError(ClusterFinderError):
    """Raised when there is an error processing data."""
    pass

class ValidationError(ClusterFinderError):
    """Raised when validation fails."""
    pass

class MaterialNotFoundError(ClusterFinderError):
    """Raised when a material cannot be found in the Materials Project database."""
    pass

class PropertyNotFoundError(ClusterFinderError):
    """Raised when a requested material property is not available."""
    pass

class StructureError(ClusterFinderError):
    """Raised when there is an error processing crystal structures."""
    pass

class NoStructureFoundError(ClusterFinderError):
    """Raised when no structure data can be found for a given material."""
    pass

class ClusterAnalysisError(ClusterFinderError):
    """Raised when there is an error during cluster analysis."""
    pass

class VisualizationError(ClusterFinderError):
    """Raised when there is an error creating visualizations."""
    pass

class FileOperationError(ClusterFinderError):
    """Raised when there is an error with file operations."""
    pass

class BatchProcessingError(ClusterFinderError):
    """Raised when there is an error during batch processing."""
    pass

class NetworkError(ClusterFinderError):
    """Raised when there are network-related issues."""
    pass

class InvalidInputError(ClusterFinderError):
    """Raised when input validation fails."""
    pass