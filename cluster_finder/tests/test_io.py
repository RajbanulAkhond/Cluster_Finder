from pymatgen.core.structure import Structure
import os
from cluster_finder.io.fileio import export_structure_to_cif

def test_export_structure_to_cif(tmp_path):
    """Test exporting structure to CIF."""
    # Create a simple structure
    lattice = [[1,0,0], [0,1,0], [0,0,1]]
    coords = [[0,0,0]]
    species = ["Fe"]
    structure = Structure(lattice, species, coords)
    
    # Export to CIF
    filename = str(tmp_path / "test.cif")
    export_structure_to_cif(structure, filename)
    
    # Check file exists
    assert os.path.exists(filename)
    
    # Try to read it back
    imported_structure = Structure.from_file(filename)
    assert len(imported_structure) == len(structure)
    assert imported_structure[0].specie.symbol == "Fe" 