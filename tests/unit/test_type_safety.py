"""Unit tests verifying return value and attribute types."""

from pathlib import Path

import pytest
from ase import Atoms

from pipeline.config import MVPConfig
from pipeline.stage01_data_mining import LocalDataMiner
from pipeline.stage02_defect_engineering import SimpleDefectEngine

pytestmark = pytest.mark.unit


class TestTypeSafety:
    """Tests that public functions return the documented types."""

    def test_stage1_returns_correct_types(self, sample_config) -> None:
        """Stage 1 should return a dict with an Atoms object and a dict."""
        result = LocalDataMiner(sample_config).load_structure()
        assert isinstance(result, dict)
        assert isinstance(result["atoms"], Atoms)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["metadata"]["material_name"], str)
        assert isinstance(result["metadata"]["num_atoms"], int)
        assert isinstance(result["metadata"]["lattice_parameters"], dict)

    def test_stage2_returns_correct_types(self, sample_config) -> None:
        """Stage 2 should return a dict with an Atoms object and a dict."""
        stage1_result = LocalDataMiner(sample_config).load_structure()
        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )
        assert isinstance(stage2_result, dict)
        assert isinstance(stage2_result["atoms"], Atoms)
        assert isinstance(stage2_result["metadata"], dict)
        assert isinstance(stage2_result["metadata"]["vacancy_index"], int)
        assert isinstance(stage2_result["metadata"]["num_atoms_defect"], int)

    def test_config_attributes_have_correct_types(self) -> None:
        """MVPConfig attributes should have the expected types."""
        config = MVPConfig()
        assert isinstance(config.material_file, Path)
        assert isinstance(config.material_name, str)
        assert isinstance(config.supercell_size, list)
        assert isinstance(config.vacancy_index, int)
        assert isinstance(config.cutoff_radius, float)
        assert isinstance(config.dataset_file, Path)

    def test_metadata_fields_have_correct_types(self, sample_config) -> None:
        """Stage 1 metadata fields should have the expected types."""
        metadata = LocalDataMiner(sample_config).load_structure()["metadata"]
        assert isinstance(metadata["material_name"], str)
        assert isinstance(metadata["num_atoms"], int)
        assert isinstance(metadata["chemical_formula"], str)
        assert isinstance(metadata["lattice_parameters"], dict)
        assert isinstance(metadata["source"], str)
        assert isinstance(metadata["file_path"], str)
        for value in metadata["lattice_parameters"].values():
            assert isinstance(value, float)
