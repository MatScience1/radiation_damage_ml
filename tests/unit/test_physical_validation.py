"""Unit tests for physical validation of structural operations."""

import pytest
import spglib

pytestmark = pytest.mark.unit


def _spacegroup_number(atoms) -> int:
    """Return the spacegroup number for an Atoms object via spglib."""
    dataset = spglib.get_symmetry_dataset(
        (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
        symprec=1e-3,
    )
    if isinstance(dataset, dict):
        return int(dataset["number"])
    return int(dataset.number)


class TestPhysicalValidation:
    """Tests that structural operations conserve physical quantities."""

    def test_mass_conservation_supercell(self, engine, aluminum_atoms) -> None:
        """Supercell creation should scale total mass by the cell factor."""
        original_mass = float(sum(aluminum_atoms.get_masses()))
        supercell = engine.create_supercell(aluminum_atoms)
        supercell_mass = float(sum(supercell.get_masses()))
        assert supercell_mass == pytest.approx(original_mass * 8, rel=1e-6)

    def test_composition_conservation(self, engine, aluminum_atoms) -> None:
        """Supercell creation should preserve the chemical composition."""
        supercell = engine.create_supercell(aluminum_atoms)
        assert set(supercell.get_chemical_symbols()) == {"Al"}
        assert len(supercell) == 8 * len(aluminum_atoms)

    def test_symmetry_preserved_in_supercell(self, engine, aluminum_atoms) -> None:
        """A 2x2x2 supercell of FCC aluminium should retain spacegroup 225."""
        supercell = engine.create_supercell(aluminum_atoms)
        assert _spacegroup_number(aluminum_atoms) == 225
        assert _spacegroup_number(supercell) == 225

    def test_vacancy_creation_removes_correct_element(
        self, engine, sample_config, supercell_atoms
    ) -> None:
        """Vacancy creation should remove exactly one atom of the target element."""
        index = 5
        removed = supercell_atoms.get_chemical_symbols()[index]
        before = supercell_atoms.get_chemical_symbols().count(removed)

        sample_config.mvp.vacancy_index = index
        result = engine.create_vacancy(supercell_atoms)

        after = result["atoms"].get_chemical_symbols().count(removed)
        assert after == before - 1
        assert result["vacancy_index"] == index
