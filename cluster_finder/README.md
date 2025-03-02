# Cluster Finder

Cluster Finder is a Python package for finding, analyzing, and visualizing atomic clusters in crystal structures, with a focus on transition metal clusters.

## Features

- Find clusters of atoms in crystal structures based on connectivity
- Analyze cluster properties (size, average distances, etc.)
- Visualize clusters and their connectivity
- Export results to various formats (CIF, CSV, JSON)
- Command-line interface for easy usage
- Extensible API for integration into other projects

## Installation

### From PyPI (recommended)

```bash
pip install cluster-finder
```

### From Source

```bash
git clone https://github.com/username/cluster_finder.git
cd cluster_finder
pip install -e .
```

## Dependencies

- numpy
- matplotlib
- pymatgen
- networkx
- pandas
- mp-api (Materials Project API)

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

# Visualize
import matplotlib.pyplot as plt
fig = cf.visualize_clusters_in_compound(structure, analyzed_clusters)
plt.show()
```

## Documentation

For more detailed usage examples and API documentation, see the [example scripts](examples/) in the repository.

### Key Modules

- `core`: Core functionality for finding and analyzing clusters
- `visualization`: Functions for visualizing clusters and structures
- `analysis`: Advanced analysis of clusters and their properties
- `io`: Input/output utilities for various file formats
- `utils`: Helper functions and utilities

## Development

### Running Tests

To run the test suite, first install the package with development dependencies:

```bash
git clone https://github.com/username/cluster_finder.git
cd cluster_finder
pip install -e ".[dev]"
```

Then run the tests using pytest:

```bash
pytest
```

For code coverage report:

```bash
pytest --cov=cluster_finder
```

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

## References

If you use this package in your research, please cite:

```bibtex
@software{akhond2023clusterfinder,
  author = {Akhond, Md. Rajbanul},
  title = {Cluster Finder: A Python package for finding atomic clusters in crystal structures},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/username/cluster_finder}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- Md. Rajbanul Akhond (makhond@iu.edu)
