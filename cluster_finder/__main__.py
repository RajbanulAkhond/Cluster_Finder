"""
Main entry point for the cluster_finder package.
"""

# Import monkey patch module early to prevent resource tracker errors
try:
    from .utils import mppatches
except ImportError:
    pass

from .cli import main

if __name__ == '__main__':
    main()