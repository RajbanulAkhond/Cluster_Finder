# Cluster Finder

Cluster Finder is a Python package for finding, analyzing, and visualizing atomic clusters in crystal structures, with a focus on transition metal clusters.

## Features

- Find clusters of atoms in crystal structures based on connectivity
- Analyze cluster properties (size, average distances, etc.)
- Visualize clusters and their connectivity
- Export results to various formats (CIF, CSV, JSON)
- Advanced cluster ranking with customizable property weights 
- Command-line interface for easy usage
- Extensible API for integration into other projects
- Comprehensive test suite 
- Detailed cluster statistics and summary reports

## Installation

### From PyPI (coming soon)

```bash
pip install cluster-finder
```

### From Source

```bash
git clone https://github.com/RajbanulAkhond/Cluster_Finder.git
cd Cluster_Finder
pip install -e .
```

## Dependencies

The package has been tested with the following versions:

```
numpy==1.26.4
matplotlib==3.9.4
pymatgen==2024.8.9
networkx==3.2.1
pandas==2.2.3
mp-api==0.45.3
pytest==8.0.2
pydantic>=2.0.0
```

These versions are pinned in the `requirements.txt` file. To install these specific versions:

```bash
pip install -r requirements.txt
```

Note: The package now requires Pydantic v2 or higher for improved data validation and serialization.

## Quick Start

### Command Line Usage

Find clusters in a structure file:

```bash
cluster-finder find structure.cif --elements Fe Co Ni --radius 3.2
```

Analyze clusters from a previous run:

```bash
cluster-finder analyze structure_clusters.json
```

Visualize clusters:

```bash
cluster-finder visualize structure_clusters.json --show
```

Rank clusters with custom properties:

```bash
cluster-finder analyze --rank results.csv --custom-props band_gap formation_energy_per_atom --weights "band_gap:2.0,formation_energy_per_atom:-1.5"
```

### Python API Usage

```python
import cluster_finder as cf
from pymatgen.core.structure import Structure

# Load structure
structure = Structure.from_file("structure.cif")

# Find transition metal clusters
connectivity_matrix, tm_indices = cf.create_connectivity_matrix(
    structure, 
    ["Fe", "Co", "Ni"],
    cutoff=3.2
)
graph = cf.structure_to_graph(connectivity_matrix)
clusters = cf.find_clusters(structure, graph, tm_indices)

# Analyze clusters
analyzed_clusters = cf.analyze_clusters(clusters, structure.lattice)

# Get cluster statistics
summary = cf.cluster_summary_stat(analyzed_clusters, structure)
print(summary)

# Visualize
import matplotlib.pyplot as plt
fig = cf.visualize_clusters_in_compound(structure, analyzed_clusters)
plt.show()

# Advanced: Rank clusters with custom properties
import pandas as pd
df = pd.read_csv("cluster_results.csv")

# Rank using default properties (min_avg_distance, point_group_order, etc.)
ranked_df = cf.rank_clusters(df)

# Rank with custom properties and weights
custom_props = ["band_gap", "formation_energy"]
weights = {"band_gap": 2.0, "formation_energy": -1.5}  # Positive favors higher values, negative favors lower values
ranked_df = cf.rank_clusters(
    df,
    custom_props=custom_props,
    prop_weights=weights
)

# Save ranked results
ranked_df.to_csv("ranked_results.csv")
```

## Documentation

For more detailed usage examples and API documentation, see the [example scripts](examples/) in the repository.

### Key Modules

- `core`: Core functionality for finding and analyzing clusters
  - `structure.py`: Structure manipulation and cluster identification
  - `utils.py`: Statistical analysis and data processing
- `visualization`: Functions for visualizing clusters and structures
- `analysis`: Advanced analysis of clusters and their properties
  - `postprocess.py`: Cluster ranking and classification
  - `dataframe.py`: Cluster data processing utilities
- `io`: Input/output utilities for various file formats
- `utils`: Helper functions and utilities

## Advanced Usage

### Cluster Ranking

The package includes a powerful ranking system for evaluating clusters based on various properties:

```python
from cluster_finder.analysis.postprocess import rank_clusters

# Rank by default properties (min_avg_distance, symmetry, stability)
ranked_df = rank_clusters("clusters_data.csv")

# Rank with custom properties and weights
ranked_df = rank_clusters(
    "clusters_data.csv",
    api_key="your_materials_project_api_key",  # Optional for retrieving properties
    custom_props=["band_gap", "formation_energy_per_atom"],
    prop_weights={
        "min_avg_distance": -2.0,              # Lower distances are better
        "energy_above_hull": -3.0,             # Lower energy is better
        "band_gap": 1.5,                       # Higher band gap is better
        "formation_energy_per_atom": -1.0      # Lower formation energy is better
    },
    include_default_ranking=True  # Whether to include default ranking criteria
)
```

Properties considered in ranking:
- **Default properties**:
  - `min_avg_distance`: Minimum average distance between atoms in cluster
  - `max_point_group_order`: Symmetry (higher order is better)
  - `space_group_order`: Space group order
  - `energy_above_hull`: Thermodynamic stability (lower is better)
  
- **Custom properties**:
  - Any material property available in the DataFrame or retrievable from the Materials Project
  - Set weights based on your research priorities (+ve weights favor higher values, -ve weights favor lower values)

## Development

### Running Tests

To run the test suite:

```bash
git clone https://github.com/RajbanulAkhond/Cluster_Finder.git
cd Cluster_Finder
pip install -e ".[dev]"
python -m pytest cluster_finder/tests/
```

The test suite includes:
- Core functionality tests
- Analysis tests
- CLI tests
- File I/O tests
- Utility tests

### Code Style

This project uses Black for code formatting, isort for import sorting, and pylint for linting:

```bash
# Format code
black cluster_finder tests

# Sort imports
isort cluster_finder tests

# Run linter
pylint cluster_finder
```

## Recent Updates

- Enhanced cluster ranking with support for custom properties and weights
- Improved materials property retrieval from the Materials Project API
- Added normalization of properties for balanced ranking
- Expanded test coverage for ranking functionalities
- Improved package structure and organization
- Enhanced documentation with comprehensive docstrings
- Fixed JSON serialization of PeriodicSite objects
- Updated cluster summary statistics with detailed output
- Added test structure for examples
- Fixed entry points in setup.py
- All tests passing successfully

## References

If you use this package in your research, please cite:

```bibtex
@software{akhond2024clusterfinder,
  author = {Akhond, Md. Rajbanul},
  title = {Cluster Finder: A Python package for finding atomic clusters in crystal structures},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/RajbanulAkhond/Cluster_Finder}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- Md. Rajbanul Akhond (makhond@iu.edu)
