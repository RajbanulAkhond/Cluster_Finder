from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cluster_finder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0,<1.27.0",
        "pandas>=2.0.3,<2.3.0",
        "networkx>=3.1,<3.5.0",
        "pymatgen>=2023.8.10,<2026.0.0",
        "matplotlib>=3.7.2,<3.11.0",
        "mp-api>=0.30.4,<0.50.0",
        "pytest>=7.4.0,<9.0.0",
        "pydantic>=1.10.8,<3.0.0"
    ],
    entry_points={
        'console_scripts': [
            'cluster-finder=cluster_finder.cli:main',
        ],
    },
    author="Md. Rajbanul Akhond",
    author_email="mdakhond@iu.edu",
    description="A package for finding and analyzing clusters in crystal structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RajbanulAkhond/Cluster_Finder/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research"
    ],
    python_requires=">=3.11.11",
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black",
            "isort",
            "pylint",
            "pytest-cov>=2.12.0",
            "mypy>=0.910"
        ]
    }
)