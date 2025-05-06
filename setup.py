"""
Setup configuration for the cluster_finder package.
"""
from setuptools import setup, find_packages

setup(
    name="cluster_finder",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.4",
        "pymatgen>=2023.0",
        "networkx>=2.6",
        "pandas>=1.3",
        "mp-api>=0.33.3",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0.0",
        "scipy>=1.7",
        "tqdm>=4.65.0"
    ],
    entry_points={
        "console_scripts": [
            "cluster-finder=cluster_finder.cli:app"
        ]
    }
)