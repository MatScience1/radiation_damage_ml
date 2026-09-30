"""Integration tests for data integrity across output formats."""

import json

import pytest
from ase import Atoms
from ase.io import read

from pipeline.stage01_data_mining import LocalDataMiner
from pipeline.stage02_defect_engineering import SimpleDefectEngine

pytestmark = pytest.mark.integration


class TestDataIntegrity:
    """Tests that metadata and counts are consistent across outputs."""

    def _run_stages(self, sample_config):
        """Run stage 1 and stage 2 against the sample configuration."""
        stage1_result = LocalDataMiner(sample_config).load_structure()
        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )
        return stage1_result, stage2_result

    def test_metadata_consistency_between_formats(self, sample_config) -> None:
        """JSON and extXYZ metadata should agree on defect fields."""
        self._run_stages(sample_config)
        name = sample_config.mvp.material_name
        defect_dir = sample_config.mvp.paths.defect_structures

        with open(defect_dir / f"{name}_vacancy.json") as handle:
            json_metadata = json.load(handle)
        atoms = read(str(defect_dir / f"{name}_vacancy.extxyz"))
        assert isinstance(atoms, Atoms)

        assert atoms.info["defect_type"] == json_metadata["defect_type"]
        assert atoms.info["vacancy_index"] == json_metadata["vacancy_index"]
        assert list(atoms.info["supercell_size"]) == json_metadata["supercell_size"]

    def test_atom_count_consistency(self, sample_config) -> None:
        """Atom counts should agree across metadata, extXYZ, and return value."""
        stage1_result, stage2_result = self._run_stages(sample_config)
        name = sample_config.mvp.material_name
        defect_dir = sample_config.mvp.paths.defect_structures

        with open(defect_dir / f"{name}_vacancy.json") as handle:
            metadata = json.load(handle)
        atoms = read(str(defect_dir / f"{name}_vacancy.extxyz"))

        assert metadata["num_atoms_defect"] == len(atoms) == len(stage2_result["atoms"])
        assert metadata["num_atoms_bulk"] == len(stage1_result["atoms"])

    def test_file_paths_are_valid(self, sample_config) -> None:
        """All expected output files should exist and be non-empty."""
        self._run_stages(sample_config)
        name = sample_config.mvp.material_name
        raw = sample_config.mvp.paths.raw_structures
        defect = sample_config.mvp.paths.defect_structures

        expected = [
            raw / f"{name}.cif",
            raw / f"{name}.json",
            defect / f"{name}_vacancy.extxyz",
            defect / f"{name}_vacancy.json",
        ]
        for path in expected:
            assert path.exists()
            assert path.is_file()
            assert path.stat().st_size > 0

    def test_json_files_are_valid(self, sample_config) -> None:
        """All JSON outputs should parse to dictionaries."""
        self._run_stages(sample_config)
        name = sample_config.mvp.material_name
        raw = sample_config.mvp.paths.raw_structures
        defect = sample_config.mvp.paths.defect_structures

        for path in [raw / f"{name}.json", defect / f"{name}_vacancy.json"]:
            with open(path) as handle:
                data = json.load(handle)
            assert isinstance(data, dict)
