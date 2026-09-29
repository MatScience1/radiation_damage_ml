"""Integration tests for data flow between pipeline stages."""

import numpy as np
import pytest

from pipeline.stage01_data_mining import LocalDataMiner
from pipeline.stage02_defect_engineering import SimpleDefectEngine

pytestmark = pytest.mark.integration


class TestDataFlow:
    """Tests for data flow between stage 1 and stage 2."""

    def test_stage1_to_stage2_atoms_transfer(self, sample_config) -> None:
        """The atoms object should transfer correctly across stages."""
        miner = LocalDataMiner(sample_config)
        stage1_result = miner.load_structure()

        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )

        assert "atoms" in stage2_result
        assert "metadata" in stage2_result
        assert len(stage2_result["atoms"]) == 31

    def test_metadata_preservation_across_stages(self, sample_config) -> None:
        """Metadata should be preserved and extended across stages."""
        miner = LocalDataMiner(sample_config)
        stage1_result = miner.load_structure()

        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )

        assert (
            stage2_result["metadata"]["material_name"]
            == stage1_result["metadata"]["material_name"]
        )
        assert "defect_type" in stage2_result["metadata"]
        assert "vacancy_index" in stage2_result["metadata"]

    def test_physical_validity_of_defect_structure(self, sample_config) -> None:
        """The defect structure should contain no overlapping atoms."""
        miner = LocalDataMiner(sample_config)
        stage1_result = miner.load_structure()

        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )

        atoms = stage2_result["atoms"]
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        assert float(distances.min()) > 1.0

    def test_supercell_dimensions_match_configuration(self, sample_config) -> None:
        """The defective supercell should match the configured dimensions."""
        miner = LocalDataMiner(sample_config)
        stage1_result = miner.load_structure()

        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )

        defect_cell = stage2_result["atoms"].cell.cellpar()
        original_cell = stage1_result["atoms"].cell.cellpar()

        assert defect_cell[0] == pytest.approx(original_cell[0] * 2, abs=0.01)
        assert defect_cell[1] == pytest.approx(original_cell[1] * 2, abs=0.01)
        assert defect_cell[2] == pytest.approx(original_cell[2] * 2, abs=0.01)
