# Radiation Damage ML Dataset Pipeline

A cloud-optimised Python pipeline to generate, process, and store a
graph neural network training dataset for predicting radiation damage
and point defect formation in crystalline materials.

Supports HEAs in the W-Mo-Nb-Zr-Ti-Ta system and other refractory/stainless
alloys. Energy/force evaluation uses a two-tier approach:
- **Fast path**: LAMMPS + ADP potential (`WMoNbZrTiTa.nist.adp.txt`)
- **Accurate path**: MLIP — MACE-MP-0, GRACE, or SevenNet (configurable)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE STAGES                          │
├──────────┬──────────────────────────────────────────────────────┤
│  Stage 1 │  DATA MINING  (stage01_data_mining.py)               │
│          │  Materials Project API                               │
│          │  ├── W-Mo-Nb-Zr-Ti-Ta (matches ADP potential)        │
│          │  ├── Fe-Ni-Cr, Mo-Nb-Ta-W, Al-Co-Cr-Fe-Ni, Si       │
│          │  └── Filter: E_above_hull < 0.1 eV/atom             │
│          │  Output: CIF + JSON → data/raw_structures/           │
├──────────┼──────────────────────────────────────────────────────┤
│  Stage 2 │  DEFECT ENGINEERING  (stage02_defect_engineering.py) │
│          │  Supercell 3×3×3 from each bulk structure            │
│          │  ├── Single vacancies (per unique element)           │
│          │  ├── Divacancies (nearest-neighbour pairs)           │
│          │  ├── Interstitials (He, H at largest void)           │
│          │  └── Thermal displacements (σ=0.05 Å, 3 snapshots)  │
│          │  Output: extXYZ + JSON → data/defect_structures/     │
├──────────┼──────────────────────────────────────────────────────┤
│  Stage 3 │  ENERGY / FORCE EVALUATION  (stage03_energy_evaluation.py)│
│          │  ┌─ Fast path  (n_vac ≤ 2) ────────────────────┐    │
│          │  │  LAMMPS + WMoNbZrTiTa ADP (metal units)     │    │
│          │  └─────────────────────────────────────────────┘    │
│          │  ┌─ Accurate path  (n_vac > 2) ───────────────┐     │
│          │  │  MLIP: MACE-MP-0 | GRACE | SevenNet        │     │
│          │  │  Select via MLIP_BACKEND env var            │     │
│          │  └─────────────────────────────────────────────┘    │
│          │  Parallelised via Ray (cluster) or multiprocessing   │
│          │  Output: NPZ + JSON → data/evaluated/               │
├──────────┼──────────────────────────────────────────────────────┤
│  Stage 4 │  GRAPH TRANSFORMATION  (stage04_graph_transform.py)  │
│          │  Nodes: Z, V_voronoi, EN, r_cov, is_defect_site      │
│          │  Edges: r_ij, unit vector (PBC-aware, cutoff 6 Å)    │
│          │  Targets: E_total, F_i, E_formation                  │
│          │  Output: graphs_raw.pkl → data/graphs/               │
├──────────┼──────────────────────────────────────────────────────┤
│  Stage 5 │  CLOUD STORAGE  (stage05_storage.py)                 │
│          │  ├── LMDB       — random access, NFS-safe            │
│          │  ├── HDF5       — chunked + LZF compressed           │
│          │  └── WebDataset — sharded .tar for S3/GCS streaming  │
│          │  Output: dataset.lmdb / dataset.h5 / shards/*.tar    │
└──────────┴──────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Cloud instance setup (Ubuntu 22.04 LTS)

```bash
cp .env.example .env
# Edit .env: set MP_API_KEY and ADP_POTENTIAL path

chmod +x setup_env.sh
./setup_env.sh          # installs deps, builds LAMMPS, downloads MACE-MP-0
./setup_env.sh --run    # setup + run pipeline immediately
./setup_env.sh --run --gpu  # with CUDA support
```

### 2. Manual pipeline execution

```bash
source ~/venv/radiation_pipeline/bin/activate
export $(cat .env | xargs)   # load env vars

python run_pipeline.py                        # full pipeline
python run_pipeline.py --start-stage 3        # resume from stage 3
python run_pipeline.py --only-stage 4 --force # re-run stage 4
STORAGE_BACKEND=hdf5 python run_pipeline.py --only-stage 5
```

### 3. Selecting the MLIP backend

```bash
# MACE-MP-0 (default, auto-downloads)
MLIP_BACKEND=mace python run_pipeline.py

# GRACE (requires manual checkpoint download — see potentials/README.md)
MLIP_BACKEND=grace MLIP_MODEL=potentials/grace_model.pth python run_pipeline.py

# SevenNet-0 (auto-downloads 7net-0 universal model)
MLIP_BACKEND=sevennet python run_pipeline.py
```

### 4. Ray cluster (multi-node)

```bash
ray start --head --port=6379                # head node
ray start --address=<head-ip>:6379          # each worker

RAY_ADDRESS=ray://<head-ip>:10001 python run_pipeline.py
```

---

## Configuration

All parameters live in `pipeline/config.py`. Key options:

| Parameter | Default | Description |
|---|---|---|
| `mp.chemsys_queries` | Fe-Ni-Cr, W-Mo-Nb-Zr-Ti-Ta, ... | Systems to download |
| `defect.supercell_size` | [3,3,3] | Supercell multiplier |
| `defect.thermal_sigma` | 0.05 Å | Thermal displacement RMS |
| `eval.adp_species_order` | W Mo Nb Zr Ti Ta | Must match ADP file header |
| `eval.mlip_complexity_threshold` | 2 | Use MLIP when n_vac > N |
| `eval.mlip.backend` | mace | mace / grace / sevennet |
| `graph.cutoff_radius` | 6.0 Å | Graph edge cutoff |
| `storage.backend` | lmdb | lmdb / hdf5 / webdataset |

All overridable via `.env` without editing Python files.

---

## Training Integration

```python
from pipeline.stage06_dataloader import get_loaders
import torch

train_loader, val_loader, test_loader = get_loaders(backend="lmdb")

# Multi-GPU DDP
train_loader, val_loader, test_loader = get_loaders(
    backend="lmdb", distributed=True, rank=local_rank, world_size=world_size,
)

for batch in train_loader:
    batch = batch.to(device)
    # batch.x          (N, 5)    node features
    # batch.z          (N,)      atomic numbers
    # batch.pos        (N, 3)    positions, Å
    # batch.forces     (N, 3)    forces, eV/Å
    # batch.y          (B,)      total energies, eV
    # batch.edge_index (2, E)
    # batch.edge_attr  (E, 4)    [dist, dx, dy, dz]
```

---

## Potentials

See `potentials/README.md` for download links and configuration for:
- WMoNbZrTiTa ADP (NIST IPRP)
- MACE-MP-0 (auto-download)
- GRACE (ICAMS/RUB)
- SevenNet-0

---

## Outputs

See `outputs/RESULTS_DESCRIPTION.txt` for a detailed description of every
output file: physical meaning, units, array shapes, and loading examples.

---

## File Layout

```
radiation_damage_ml/
├── pipeline/
│   ├── config.py                      # unified configuration + MLIP settings
│   ├── stage01_data_mining.py         # MP download + filter
│   ├── stage02_defect_engineering.py  # ASE supercell + defect generation
│   ├── stage03_energy_evaluation.py   # LAMMPS ADP / MLIP + Ray
│   ├── stage04_graph_transform.py     # ASE → PyG Data
│   ├── stage05_storage.py             # LMDB / HDF5 / WebDataset
│   └── stage06_dataloader.py          # GPU streaming loaders
├── potentials/
│   └── README.md                      # download instructions
├── outputs/
│   └── RESULTS_DESCRIPTION.txt        # detailed output reference
├── logs/
├── data/                              # generated (git-ignored)
│   ├── raw_structures/
│   ├── defect_structures/
│   ├── evaluated/
│   ├── graphs/
│   └── shards/
├── run_pipeline.py                    # master orchestrator
├── setup_env.sh                       # cloud bootstrap script
├── requirements.txt
├── .env.example                       # env var template
├── .gitignore
└── README.md
```

---

## Known Issues Fixed (vs. original code)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `run_pipeline.py` | `importlib.import_module` cannot import modules starting with a digit (`01_data_mining`) — `ModuleNotFoundError` at runtime | Renamed all stage files to `stage01_...`; load by file path via `importlib.util.spec_from_file_location` |
| 2 | `stage02_defect_engineering.py` | `sc.append(Atoms(...))` passes `Atoms` (plural) where `Atom` (singular) is required — `TypeError` on every interstitial | Changed to `sc.append(Atom(species, position=void_pos))` |
| 3 | `stage01_data_mining.py` | `logging.FileHandler(os.path.join(CFG.__class__.__name__, "mining.log"))` evaluates to path `"PipelineConfig/mining.log"` — `FileNotFoundError` | Removed the dead-code conditional and cleaned up logging setup |
| 4 | `stage04_graph_transform.py` | `_ELECTRONEGATIVITY` dict had duplicate keys `Al`, `Hf`, `Zr`, `V` (silently discarded last assignment) | Removed duplicates; single canonical entry per element |
| 5 | All stages | Stage files named `01_*.py` are not valid Python identifiers; importing with `importlib` requires workaround | Consistent `stage0N_` prefix; file-path-based loading in orchestrator |
