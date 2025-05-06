"""
Utility functions for the cluster_finder package.
"""

def has_lzma_support():
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