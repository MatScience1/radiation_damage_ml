"""Unit tests for numerical stability of structural operations."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestNumericalStability:
    """Tests that structures remain numerically well defined."""

    def test_no_atom_overlap(self, engine, aluminum_atoms) -> None:
        """No two atoms in the defect structure should be closer than 2.0 A."""
        metadata = {"chemical_formula": aluminum_atoms.get_chemical_formula()}
        result = engine.generate_defect_structure(aluminum_atoms, metadata)
        atoms = result["atoms"]

        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        assert float(distances.min()) > 2.0

    def test_coordinates_in_valid_range(self, engine, aluminum_atoms) -> None:
        """All positions and cell vectors should be finite."""
        supercell = engine.create_supercell(aluminum_atoms)
        assert np.all(np.isfinite(supercell.get_positions()))
        assert np.all(np.isfinite(supercell.get_cell()))

    def test_positions_reasonable(self, engine, aluminum_atoms) -> None:
        """All atoms should lie inside the supercell in fractional space."""
        supercell = engine.create_supercell(aluminum_atoms)
        scaled = supercell.get_scaled_positions()
        assert np.all(scaled >= -1e-9)
        assert np.all(scaled < 1.0 + 1e-9)

    def test_lattice_parameters_physical(self, engine, aluminum_atoms) -> None:
        """Lattice lengths should be positive and angles within (0, 180)."""
        supercell = engine.create_supercell(aluminum_atoms)
        cellpar = supercell.cell.cellpar()
        assert np.all(cellpar[:3] > 0)
        assert np.all((cellpar[3:] > 0) & (cellpar[3:] < 180))
