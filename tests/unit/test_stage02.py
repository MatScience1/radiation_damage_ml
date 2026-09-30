"""Unit tests for pipeline/stage02_defect_engineering.py."""

import json

import pytest
from ase import Atoms
from ase.io import read

from pipeline.stage02_defect_engineering import SimpleDefectEngine

pytestmark = pytest.mark.unit


class TestSimpleDefectEngine:
    """Tests for the SimpleDefectEngine class."""

    def test_create_supercell_multiplies_atoms(
        self, sample_config, aluminum_atoms
    ) -> None:
        """create_supercell should multiply the atom count correctly."""
        engine = SimpleDefectEngine(sample_config)
        supercell = engine.create_supercell(aluminum_atoms)
        assert len(supercell) == 32

    def test_create_supercell_preserves_structure(
        self, sample_config, aluminum_atoms
    ) -> None:
        """create_supercell should scale the cell by the supercell size."""
        engine = SimpleDefectEngine(sample_config)
        supercell = engine.create_supercell(aluminum_atoms)

        original_cell = aluminum_atoms.cell.cellpar()
        supercell_cell = supercell.cell.cellpar()

        assert supercell_cell[0] == pytest.approx(original_cell[0] * 2, abs=0.01)
        assert supercell_cell[1] == pytest.approx(original_cell[1] * 2, abs=0.01)
        assert supercell_cell[2] == pytest.approx(original_cell[2] * 2, abs=0.01)

    def test_create_vacancy_removes_one_atom(
        self, sample_config, supercell_atoms
    ) -> None:
        """create_vacancy should remove exactly one atom."""
        engine = SimpleDefectEngine(sample_config)
        initial_count = len(supercell_atoms)
        result = engine.create_vacancy(supercell_atoms)
        assert len(result["atoms"]) == initial_count - 1

    def test_create_vacancy_returns_correct_index(
        self, sample_config, supercell_atoms
    ) -> None:
        """create_vacancy should return the configured vacancy index."""
        sample_config.mvp.vacancy_index = 5
        engine = SimpleDefectEngine(sample_config)
        result = engine.create_vacancy(supercell_atoms)
        assert result["vacancy_index"] == 5
        assert len(result["atoms"]) == len(supercell_atoms) - 1

    def test_create_vacancy_raises_on_invalid_index(
        self, sample_config, supercell_atoms
    ) -> None:
        """create_vacancy should raise ValueError for an out of range index."""
        sample_config.mvp.vacancy_index = 100
        engine = SimpleDefectEngine(sample_config)
        with pytest.raises(ValueError, match="out of range"):
            engine.create_vacancy(supercell_atoms)

    def test_generate_defect_structure_writes_files(
        self, sample_config, aluminum_atoms
    ) -> None:
        """generate_defect_structure should write extXYZ and JSON files."""
        engine = SimpleDefectEngine(sample_config)
        metadata = {"chemical_formula": "Al", "num_atoms": 4}
        engine.generate_defect_structure(aluminum_atoms, metadata)

        material_name = sample_config.mvp.material_name
        defect_dir = sample_config.mvp.paths.defect_structures
        extxyz_file = defect_dir / f"{material_name}_vacancy.extxyz"
        json_file = defect_dir / f"{material_name}_vacancy.json"

        assert extxyz_file.exists()
        assert json_file.exists()

    def test_extxyz_contains_correct_metadata(
        self, sample_config, aluminum_atoms
    ) -> None:
        """The extXYZ file should carry the expected metadata tags."""
        engine = SimpleDefectEngine(sample_config)
        metadata = {"chemical_formula": "Al", "num_atoms": 4}
        engine.generate_defect_structure(aluminum_atoms, metadata)

        material_name = sample_config.mvp.material_name
        extxyz_file = (
            sample_config.mvp.paths.defect_structures
            / f"{material_name}_vacancy.extxyz"
        )
        atoms = read(str(extxyz_file))
        assert isinstance(atoms, Atoms)
        assert atoms.info["defect_type"] == "single_vacancy"
        assert list(atoms.info["supercell_size"]) == [2, 2, 2]
        assert "vacancy_index" in atoms.info

    def test_json_sidecar_contains_complete_metadata(
        self, sample_config, aluminum_atoms
    ) -> None:
        """The JSON sidecar should contain all required metadata fields."""
        engine = SimpleDefectEngine(sample_config)
        metadata = {"chemical_formula": "Al", "num_atoms": 4}
        engine.generate_defect_structure(aluminum_atoms, metadata)

        material_name = sample_config.mvp.material_name
        json_file = (
            sample_config.mvp.paths.defect_structures / f"{material_name}_vacancy.json"
        )
        with open(json_file) as handle:
            data = json.load(handle)

        required_fields = [
            "material_name",
            "defect_type",
            "vacancy_index",
            "supercell_size",
            "num_atoms_bulk",
            "num_atoms_defect",
        ]
        for field in required_fields:
            assert field in data

        assert data["num_atoms_bulk"] == 4
        assert data["num_atoms_defect"] == 31
        assert data["defect_type"] == "single_vacancy"

    def test_supercell_matrix_is_correct(self, sample_config, aluminum_atoms) -> None:
        """The supercell should be a 2x2x2 expansion of the input cell."""
        engine = SimpleDefectEngine(sample_config)
        supercell = engine.create_supercell(aluminum_atoms)
        assert len(supercell) == 32
        cell_params = supercell.cell.cellpar()
        original = aluminum_atoms.cell.cellpar()
        assert cell_params[0] == pytest.approx(original[0] * 2, abs=0.01)
