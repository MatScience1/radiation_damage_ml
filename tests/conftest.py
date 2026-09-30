"""Shared pytest fixtures for the radiation_damage_ml MVP test suite."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, make_supercell

from pipeline.config import FullConfig
from pipeline.stage02_defect_engineering import SimpleDefectEngine


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory (the parent of the tests directory)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_config(project_root: Path, tmp_path: Path) -> FullConfig:
    """Return a FullConfig whose inputs and outputs are test isolated.

    Inputs point at the checked-in materials/Al.cif; all output directories
    point inside a pytest temporary directory so tests never touch data/.
    """
    config = FullConfig()
    config.mvp.material_file = project_root / "materials" / "Al.cif"
    config.mvp.material_name = "test_aluminum"
    config.mvp.paths.data_dir = tmp_path / "data"
    config.mvp.paths.raw_structures = tmp_path / "data" / "raw_structures"
    config.mvp.paths.defect_structures = tmp_path / "data" / "defect_structures"
    config.mvp.paths.evaluated = tmp_path / "data" / "evaluated"
    config.mvp.paths.graphs = tmp_path / "data" / "graphs"
    config.mvp.paths.output = tmp_path / "outputs"
    return config


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def aluminum_atoms() -> Atoms:
    """Return the conventional cubic aluminium FCC cell (4 atoms, a=4.046 A)."""
    return bulk("Al", "fcc", a=4.046, cubic=True)


@pytest.fixture
def supercell_atoms(aluminum_atoms: Atoms) -> Atoms:
    """Return a 2x2x2 supercell of the aluminium structure (32 atoms)."""
    supercell: Atoms = make_supercell(aluminum_atoms, np.diag([2, 2, 2]))
    return supercell


@pytest.fixture
def hcp_magnesium_atoms() -> Atoms:
    """Return a hexagonal close-packed magnesium structure (2 atoms)."""
    return bulk("Mg", "hcp", a=3.21, c=5.21)


@pytest.fixture
def engine(sample_config) -> SimpleDefectEngine:
    """Return a SimpleDefectEngine bound to the test sample configuration."""
    return SimpleDefectEngine(sample_config)
