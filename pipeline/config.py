"""
config.py — Central configuration for the radiation damage ML pipeline.

All paths, physical constants, and hyperparameters are defined here.
Override any value via environment variables (loaded from .env by python-dotenv).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional at import time

# ---------------------------------------------------------------------------
# Root paths  (config.py lives inside pipeline/, so BASE_DIR = project root)
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw_structures")
DEFECT_DIR    = os.path.join(DATA_DIR, "defect_structures")
EVALUATED_DIR = os.path.join(DATA_DIR, "evaluated")
GRAPH_DIR     = os.path.join(DATA_DIR, "graphs")
LMDB_PATH     = os.path.join(DATA_DIR, "dataset.lmdb")
HDF5_PATH     = os.path.join(DATA_DIR, "dataset.h5")
LOG_DIR       = os.path.join(BASE_DIR, "logs")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")

for _d in [RAW_DIR, DEFECT_DIR, EVALUATED_DIR, GRAPH_DIR, LOG_DIR, OUTPUTS_DIR]:
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Materials Project API
# ---------------------------------------------------------------------------
@dataclass
class MPConfig:
    api_key: str = os.environ.get("MP_API_KEY", "YOUR_MP_API_KEY")

    # Target chemical systems — HEAs and reference materials
    chemsys_queries: List[str] = field(default_factory=lambda: [
        "Si",               # crystalline silicon baseline
        "Fe-Ni-Cr",         # stainless-steel proxy HEA
        "Mo-Nb-Ta-W",       # BCC refractory HEA (matches ADP potential)
        "Al-Co-Cr-Fe-Ni",   # Cantor alloy family
        "Ti-Zr-Hf-V-Nb",   # HCP/BCC HEA
        "W-Mo-Nb-Zr-Ti-Ta", # matches WMoNbZrTiTa ADP potential
    ])

    # Quality filters
    is_stable: bool              = False  # allow metastable phases
    energy_above_hull_max: float = 0.1   # eV/atom
    nsites_max: int              = 80

    fields: List[str] = field(default_factory=lambda: [
        "material_id", "formula_pretty", "structure",
        "energy_per_atom", "formation_energy_per_atom",
        "band_gap", "nsites", "density", "symmetry",
        "theoretical", "energy_above_hull",
    ])


# ---------------------------------------------------------------------------
# Defect Engineering
# ---------------------------------------------------------------------------
@dataclass
class DefectConfig:
    supercell_size: List[int]       = field(default_factory=lambda: [3, 3, 3])
    thermal_sigma: float            = 0.05  # Angstrom, Debye-Waller amplitude
    interstitial_species: List[str] = field(default_factory=lambda: ["He", "H"])
    divacancy_max_dist: float       = 5.0   # Angstrom
    n_thermal_snapshots: int        = 3
    rng_seed: int                   = 42


# ---------------------------------------------------------------------------
# MLIP — Machine Learning Interatomic Potential
# ---------------------------------------------------------------------------
@dataclass
class MLIPConfig:
    """
    Controls which MLIP backend is used for high-accuracy evaluations.

    Supported backends:
        "mace"     — MACE-MP universal potential (public, pip-installable).
                     Requires: mace-torch
        "grace"    — GRACE (ICAMS/RUB), near-DFT on transition metals.
                     Requires: grace-tensorpotential (install from source)
        "sevennet" — SevenNet-0 universal MLIP.
                     Requires: sevenn

    model_path:
        Path to a downloaded checkpoint file.
        If empty, MACE will auto-download the MACE-MP-0 medium checkpoint.
        For GRACE, must point to the .pth file.

    device:
        "cpu"  — safe on any cloud instance
        "cuda" — requires GPU + matching torch+cuda build

    dtype:
        "float64" — recommended for energy/force accuracy
        "float32" — faster, slightly less accurate
    """
    backend: str    = os.environ.get("MLIP_BACKEND", "mace")
    model_path: str = os.environ.get("MLIP_MODEL", "")
    device: str     = os.environ.get("MLIP_DEVICE", "cpu")
    dtype: str      = "float64"


# ---------------------------------------------------------------------------
# Energy / Force Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    # --- LAMMPS (fast path, classical ADP) ---
    lammps_cmd: str          = os.environ.get("LAMMPS_CMD", "lmp")
    # WMoNbZrTiTa ADP potential (NIST IPRP)
    adp_potential_path: str  = os.environ.get(
        "ADP_POTENTIAL",
        os.path.join(BASE_DIR, "potentials", "WMoNbZrTiTa.nist.adp.txt")
    )
    # Species order must match the ADP file header exactly
    adp_species_order: List[str] = field(
        default_factory=lambda: ["W", "Mo", "Nb", "Zr", "Ti", "Ta"]
    )

    # --- MLIP (accurate path) ---
    mlip: MLIPConfig         = field(default_factory=MLIPConfig)

    # Routing: use MLIP when n_vacancies > this threshold
    use_mlip_for_complex: bool = True
    mlip_complexity_threshold: int = 2  # vacancies

    # --- Ray parallelism ---
    ray_num_cpus: Optional[int] = None  # None = auto-detect
    ray_num_gpus: int           = 0
    batch_size: int             = 64


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
@dataclass
class GraphConfig:
    cutoff_radius: float     = 6.0   # Angstrom
    max_neighbors: int       = 32
    include_edge_vectors: bool = True

    node_feature_keys: List[str] = field(default_factory=lambda: [
        "atomic_number", "site_volume", "electronegativity",
        "covalent_radius", "is_defect_site",
    ])


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
@dataclass
class StorageConfig:
    backend: str              = os.environ.get("STORAGE_BACKEND", "lmdb")
    lmdb_map_size: int        = 1 << 40   # 1 TB virtual address space
    hdf5_chunk_size: int      = 256
    hdf5_compression: str     = "lzf"    # fast, no external dep
    webdataset_shard_size: int = 1000
    webdataset_output_dir: str = os.path.join(DATA_DIR, "shards")


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
@dataclass
class LoaderConfig:
    batch_size: int     = 32
    num_workers: int    = 4
    pin_memory: bool    = True
    prefetch_factor: int = 2
    train_split: float  = 0.80
    val_split: float    = 0.10
    # test_split = 1 - train - val implicitly


# ---------------------------------------------------------------------------
# Unified pipeline config (singleton)
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    mp:      MPConfig      = field(default_factory=MPConfig)
    defect:  DefectConfig  = field(default_factory=DefectConfig)
    eval:    EvalConfig    = field(default_factory=EvalConfig)
    graph:   GraphConfig   = field(default_factory=GraphConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    loader:  LoaderConfig  = field(default_factory=LoaderConfig)


CFG = PipelineConfig()
