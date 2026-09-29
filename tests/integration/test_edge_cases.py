"""Integration tests for edge cases across the pipeline."""

import pytest
from ase.io import write

from pipeline.stage01_data_mining import LocalDataMiner
from pipeline.stage02_defect_engineering import SimpleDefectEngine

pytestmark = pytest.mark.integration


class TestEdgeCases:
    """Tests for non-cubic materials, asymmetric supercells, and index bounds."""

    def test_handles_non_cubic_crystal(
        self, sample_config, hcp_magnesium_atoms, temp_output_dir
    ) -> None:
        """The pipeline should handle a non-cubic (HCP) crystal."""
        test_cif = temp_output_dir / "Mg.cif"
        write(str(test_cif), hcp_magnesium_atoms)

        sample_config.mvp.material_file = test_cif
        sample_config.mvp.material_name = "magnesium"

        stage1_result = LocalDataMiner(sample_config).load_structure()
        assert len(stage1_result["atoms"]) == 2

        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )
        assert len(stage2_result["atoms"]) == 15

    def test_handles_asymmetric_supercell(self, sample_config) -> None:
        """The pipeline should handle an asymmetric supercell size."""
        sample_config.mvp.supercell_size = [2, 3, 4]

        stage1_result = LocalDataMiner(sample_config).load_structure()
        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )
        assert len(stage2_result["atoms"]) == 4 * 2 * 3 * 4 - 1

    def test_handles_large_vacancy_index(self, sample_config) -> None:
        """Removing the last atom index should produce a valid structure."""
        sample_config.mvp.vacancy_index = 31

        stage1_result = LocalDataMiner(sample_config).load_structure()
        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )
        assert len(stage2_result["atoms"]) == 31

    def test_handles_first_vacancy_index(self, sample_config) -> None:
        """Removing the first atom index should produce a valid structure."""
        sample_config.mvp.vacancy_index = 0

        stage1_result = LocalDataMiner(sample_config).load_structure()
        engine = SimpleDefectEngine(sample_config)
        stage2_result = engine.generate_defect_structure(
            stage1_result["atoms"],
            stage1_result["metadata"],
        )
        assert len(stage2_result["atoms"]) == 31
