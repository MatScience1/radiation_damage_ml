"""
config.py - Central configuration for the radiation damage ML pipeline.

Defines the directory layout and stage-specific settings for the MVP, plus
placeholder configuration for the advanced stages (V1 ADP, V2 GRACE).

All default paths are anchored to the project root resolved from this file,
so the pipeline behaves identically regardless of the working directory it
is launched from.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Project root is the directory that contains the pipeline/ package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass
class BasePaths:
    """Base directory structure for the pipeline.

    Every path is anchored to the project root rather than the current
    working directory.
    """

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_structures: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw_structures"
    )
    defect_structures: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "defect_structures"
    )
    evaluated: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "evaluated")
    graphs: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "graphs")
    output: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")


@dataclass
class MVPConfig:
    """Configuration for the MVP stage, the minimal working version."""

    paths: BasePaths = field(default_factory=BasePaths)

    # Input material: a local CIF file shipped with the repository.
    material_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "materials" / "Al.cif"
    )
    material_name: str = "aluminum"

    # Supercell generation.
    supercell_size: List[int] = field(default_factory=lambda: [2, 2, 2])

    # Defect generation. The MVP supports single vacancies only.
    defect_type: str = "single_vacancy"
    vacancy_index: int = 0

    # Evaluation settings. The MVP uses the ASE built-in EMT potential,
    # which is valid for a small set of metals including aluminium.
    evaluation_backend: str = "ase_emt"
    cutoff_radius: float = 5.0

    # Storage. The MVP writes a single PyTorch file.
    storage_format: str = "pytorch"
    dataset_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "dataset.pt"
    )

    def create_directories(self) -> None:
        """Create all directories required by the MVP pipeline."""
        self.paths.raw_structures.mkdir(parents=True, exist_ok=True)
        self.paths.defect_structures.mkdir(parents=True, exist_ok=True)
        self.paths.evaluated.mkdir(parents=True, exist_ok=True)
        self.paths.graphs.mkdir(parents=True, exist_ok=True)
        self.paths.output.mkdir(parents=True, exist_ok=True)


@dataclass
class FullConfig:
    """Top level configuration for the MVP pipeline.

    Only the MVP stage is present. The V1 (ADP) and V2 (GRACE) configuration
    classes were removed with the rest of the advanced code; recover them from
    git history when those stages are reintroduced.
    """

    mvp: MVPConfig = field(default_factory=MVPConfig)


def load_config(config_path: Optional[Path] = None) -> FullConfig:
    """Load configuration from file or return the built-in defaults.

    Args:
        config_path: Optional path to a configuration file. YAML/JSON loading
            is not implemented in the MVP; when provided, the argument is
            validated but ignored in favour of defaults.

    Returns:
        A fully populated FullConfig instance.
    """
    if config_path is not None and config_path.exists():
        # TODO: Implement YAML/JSON loading for V1.
        pass
    return FullConfig()
