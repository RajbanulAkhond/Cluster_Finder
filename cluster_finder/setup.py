#!/usr/bin/env python
"""
Setup script for cluster_finder package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cluster_finder",
    version="0.1.0",
    author="Md. Rajbanul Akhond",
    author_email="makhond@iu.edu",
    description="A package for finding and analyzing atomic clusters in crystal structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/cluster_finder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11.11",
    install_requires=[
        "numpy>=1.20.0,<1.27.0",
        "matplotlib>=3.3.0,<4.0.0",
        "pymatgen>=2023.0.0,<=2024.9.0",
        "networkx>=2.5.0,<3.3.0",
        "pandas>=1.3.0,<2.3.0",
        "mp-api>=0.30.1,<=0.46.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "pylint>=2.12.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "cluster-finder=cluster_finder.cli:main",
        ],
    },
)
