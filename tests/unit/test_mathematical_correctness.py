"""Unit tests for mathematical correctness of supercell construction."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestMathematicalCorrectness:
    """Tests for the supercell transformation matrix and resulting geometry."""

    def test_transformation_matrix_determinant(self, sample_config) -> None:
        """The 2x2x2 transformation matrix should have determinant 8."""
        matrix = np.diag(sample_config.mvp.supercell_size)
        assert np.linalg.det(matrix) == pytest.approx(8.0, abs=1e-10)

    def test_volume_scales_correctly(self, engine, aluminum_atoms) -> None:
        """Cell volume should scale by the determinant of the transformation."""
        supercell = engine.create_supercell(aluminum_atoms)
        assert supercell.get_volume() == pytest.approx(
            aluminum_atoms.get_volume() * 8, rel=1e-6
        )

    def test_cell_angles_preserved(self, engine, aluminum_atoms) -> None:
        """A cubic input cell should retain 90 degree angles after expansion."""
        supercell = engine.create_supercell(aluminum_atoms)
        for angle in supercell.cell.cellpar()[3:]:
            assert angle == pytest.approx(90.0, abs=1e-6)

    def test_fractional_coordinates_valid(self, engine, aluminum_atoms) -> None:
        """All fractional coordinates should lie within the unit cell."""
        supercell = engine.create_supercell(aluminum_atoms)
        scaled = supercell.get_scaled_positions()
        assert np.all(scaled >= -1e-9)
        assert np.all(scaled < 1.0 + 1e-9)
