"""Unit tests for pipeline/config.py."""

from pathlib import Path

import pytest

from pipeline.config import PROJECT_ROOT, BasePaths, FullConfig, MVPConfig, load_config

pytestmark = pytest.mark.unit


class TestBasePaths:
    """Tests for the BasePaths dataclass."""

    def test_default_paths_are_anchored_to_project_root(self) -> None:
        """Default paths should be absolute and under the project root."""
        paths = BasePaths()
        candidates = [
            paths.data_dir,
            paths.raw_structures,
            paths.defect_structures,
            paths.evaluated,
            paths.graphs,
            paths.output,
        ]
        for path in candidates:
            assert path.is_absolute()
            assert str(path).startswith(str(PROJECT_ROOT))

    def test_all_required_paths_exist(self) -> None:
        """All required path attributes should be defined."""
        paths = BasePaths()
        required_attrs = [
            "data_dir",
            "raw_structures",
            "defect_structures",
            "evaluated",
            "graphs",
            "output",
        ]
        for attr in required_attrs:
            assert hasattr(paths, attr)
            assert isinstance(getattr(paths, attr), Path)


class TestMVPConfig:
    """Tests for the MVPConfig dataclass."""

    def test_default_values_are_sensible(self) -> None:
        """Default configuration values should be reasonable."""
        config = MVPConfig()
        assert config.supercell_size == [2, 2, 2]
        assert config.defect_type == "single_vacancy"
        assert config.vacancy_index == 0
        assert config.evaluation_backend == "ase_emt"
        assert config.cutoff_radius == 5.0
        assert config.storage_format == "pytorch"

    def test_material_file_default_is_anchored(self) -> None:
        """The default material file should be the checked-in Al.cif."""
        config = MVPConfig()
        assert config.material_file.is_absolute()
        assert config.material_file == PROJECT_ROOT / "materials" / "Al.cif"

    def test_create_directories_creates_all_folders(self, sample_config) -> None:
        """create_directories should create all required output folders."""
        sample_config.mvp.create_directories()
        paths = sample_config.mvp.paths
        assert paths.raw_structures.exists()
        assert paths.defect_structures.exists()
        assert paths.evaluated.exists()
        assert paths.graphs.exists()
        assert paths.output.exists()

    def test_create_directories_is_idempotent(self, sample_config) -> None:
        """Calling create_directories twice should not raise."""
        sample_config.mvp.create_directories()
        sample_config.mvp.create_directories()
        assert sample_config.mvp.paths.raw_structures.exists()


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_returns_defaults_without_path(self) -> None:
        """load_config should return defaults when no path is provided."""
        config = load_config()
        assert isinstance(config, FullConfig)
        assert isinstance(config.mvp, MVPConfig)

    def test_load_config_handles_missing_file(self) -> None:
        """load_config should fall back to defaults for a missing file."""
        config = load_config(Path("nonexistent.yaml"))
        assert isinstance(config.mvp, MVPConfig)
