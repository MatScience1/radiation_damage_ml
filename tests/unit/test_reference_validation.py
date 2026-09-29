"""Unit tests comparing pipeline output against literature reference values."""

import pytest

from pipeline.stage01_data_mining import LocalDataMiner

pytestmark = pytest.mark.unit

LITERATURE_AL_LATTICE = 4.046  # Angstrom, aluminium FCC lattice constant
AMU_TO_GRAM = 1.66054e-24
A3_TO_CM3 = 1e-24


class TestReferenceValidation:
    """Tests that computed values match known reference values for aluminium."""

    def test_lattice_constant_matches_literature(self, sample_config) -> None:
        """The aluminium lattice constant should match the literature value."""
        result = LocalDataMiner(sample_config).load_structure()
        lattice = result["metadata"]["lattice_parameters"]
        assert lattice["a"] == pytest.approx(LITERATURE_AL_LATTICE, abs=0.001)
        assert lattice["b"] == pytest.approx(LITERATURE_AL_LATTICE, abs=0.001)
        assert lattice["c"] == pytest.approx(LITERATURE_AL_LATTICE, abs=0.001)

    def test_density_calculation_correct(self, sample_config) -> None:
        """The computed density should match aluminium, about 2.70 g/cm^3."""
        atoms = LocalDataMiner(sample_config).load_structure()["atoms"]
        mass_amu = float(sum(atoms.get_masses()))
        volume_a3 = float(atoms.get_volume())
        density = mass_amu * AMU_TO_GRAM / (volume_a3 * A3_TO_CM3)
        assert density == pytest.approx(2.70, abs=0.05)

    def test_volume_per_atom_reasonable(self, sample_config) -> None:
        """Volume per atom should be about 16.6 A^3 for aluminium."""
        atoms = LocalDataMiner(sample_config).load_structure()["atoms"]
        volume_per_atom = float(atoms.get_volume()) / len(atoms)
        assert volume_per_atom == pytest.approx(16.6, abs=0.2)

    def test_mass_per_atom_correct(self, sample_config) -> None:
        """Mean atomic mass should match aluminium, about 26.98 amu."""
        atoms = LocalDataMiner(sample_config).load_structure()["atoms"]
        mean_mass = float(atoms.get_masses().mean())
        assert mean_mass == pytest.approx(26.98, rel=1e-2)
