"""Integration tests for reproducibility of pipeline outputs."""

import hashlib
import json

import numpy as np
import pytest

from pipeline.stage01_data_mining import LocalDataMiner

pytestmark = pytest.mark.integration


def _md5(path) -> str:
    """Return the MD5 hex digest of a file."""
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


class TestReproducibility:
    """Tests that repeated runs produce identical results."""

    def test_pipeline_is_deterministic(self, sample_config) -> None:
        """Two runs should produce identical positions and metadata."""
        result1 = LocalDataMiner(sample_config).load_structure()
        result2 = LocalDataMiner(sample_config).load_structure()
        assert np.allclose(
            result1["atoms"].get_positions(), result2["atoms"].get_positions()
        )
        assert result1["metadata"] == result2["metadata"]

    def test_file_hashes_are_stable(self, sample_config) -> None:
        """The generated CIF should be byte identical across runs."""
        cif_file = (
            sample_config.mvp.paths.raw_structures
            / f"{sample_config.mvp.material_name}.cif"
        )
        LocalDataMiner(sample_config).load_structure()
        hash_first = _md5(cif_file)
        LocalDataMiner(sample_config).load_structure()
        hash_second = _md5(cif_file)
        assert hash_first == hash_second

    def test_metadata_identical_across_runs(self, sample_config) -> None:
        """The generated JSON metadata should be identical across runs."""
        json_file = (
            sample_config.mvp.paths.raw_structures
            / f"{sample_config.mvp.material_name}.json"
        )
        LocalDataMiner(sample_config).load_structure()
        first = json.loads(json_file.read_text())
        LocalDataMiner(sample_config).load_structure()
        second = json.loads(json_file.read_text())
        assert first == second
