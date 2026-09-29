"""Unit tests for pipeline/stage01_data_mining.py."""

import json
from pathlib import Path

import pytest

from pipeline.stage01_data_mining import LocalDataMiner

pytestmark = pytest.mark.unit


class TestLocalDataMiner:
    """Tests for the LocalDataMiner class."""

    def test_init_creates_directories(self, sample_config) -> None:
        """Initialisation should create the raw_structures directory."""
        LocalDataMiner(sample_config)
        assert sample_config.mvp.paths.raw_structures.exists()

    def test_load_structure_reads_cif_correctly(self, sample_config) -> None:
        """load_structure should read the CIF and return an Atoms object."""
        miner = LocalDataMiner(sample_config)
        result = miner.load_structure()
        assert "atoms" in result
        assert "metadata" in result
        assert len(result["atoms"]) == 4

    def test_load_structure_extracts_correct_metadata(self, sample_config) -> None:
        """load_structure should extract all required metadata fields."""
        miner = LocalDataMiner(sample_config)
        result = miner.load_structure()
        metadata = result["metadata"]
        required_fields = [
            "material_name",
            "num_atoms",
            "chemical_formula",
            "lattice_parameters",
            "source",
            "file_path",
        ]
        for field in required_fields:
            assert field in metadata

    def test_lattice_parameters_match_expected(self, sample_config) -> None:
        """Lattice parameters should match known aluminium values."""
        miner = LocalDataMiner(sample_config)
        result = miner.load_structure()
        lattice = result["metadata"]["lattice_parameters"]
        assert lattice["a"] == pytest.approx(4.046, abs=0.01)
        assert lattice["b"] == pytest.approx(4.046, abs=0.01)
        assert lattice["c"] == pytest.approx(4.046, abs=0.01)
        assert lattice["alpha"] == pytest.approx(90.0, abs=0.1)
        assert lattice["beta"] == pytest.approx(90.0, abs=0.1)
        assert lattice["gamma"] == pytest.approx(90.0, abs=0.1)

    def test_load_structure_writes_output_files(self, sample_config) -> None:
        """load_structure should write CIF and JSON sidecar files."""
        miner = LocalDataMiner(sample_config)
        miner.load_structure()

        material_name = sample_config.mvp.material_name
        raw_dir = sample_config.mvp.paths.raw_structures
        cif_file = raw_dir / f"{material_name}.cif"
        json_file = raw_dir / f"{material_name}.json"

        assert cif_file.exists()
        assert json_file.exists()

        with open(json_file) as handle:
            data = json.load(handle)
        assert data["num_atoms"] == 4

    def test_load_structure_raises_on_missing_file(self, sample_config) -> None:
        """load_structure should raise FileNotFoundError for a missing CIF."""
        sample_config.mvp.material_file = Path("nonexistent.cif")
        miner = LocalDataMiner(sample_config)
        with pytest.raises(FileNotFoundError):
            miner.load_structure()

    def test_chemical_formula_is_correct(self, sample_config) -> None:
        """The chemical formula should contain aluminium."""
        miner = LocalDataMiner(sample_config)
        result = miner.load_structure()
        assert "Al" in result["metadata"]["chemical_formula"]
