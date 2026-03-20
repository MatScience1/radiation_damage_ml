"""
stage02_defect_engineering.py — Programmatic radiation defect generation via ASE.

For each bulk structure in RAW_DIR:
  1. Build a 3×3×3 supercell.
  2. Generate:
       - Single vacancies    (one per unique element species)
       - Divacancies         (nearest-neighbour pairs, up to 4)
       - Interstitials       (He, H inserted at the largest Voronoi void)
  3. Apply Gaussian thermal displacements (Debye-Waller approximation).
  4. Serialise augmented Atoms + metadata as JSON + extXYZ.

Output  →  data/defect_structures/
    <defect_id>.xyz          — atomic positions in extXYZ format
    <defect_id>.json         — defect metadata
    defect_manifest.json     — ordered list of all defect IDs produced
"""

import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms, Atom                   # BUG FIX: import Atom (singular)
from ase.build import make_supercell
from ase.io import read as ase_read, write as ase_write
from ase.geometry import get_distances

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CFG, RAW_DIR, DEFECT_DIR

log = logging.getLogger("stage02_defect_engineering")

RNG = np.random.default_rng(CFG.defect.rng_seed)


# ---------------------------------------------------------------------------
# Supercell construction
# ---------------------------------------------------------------------------
def build_supercell(atoms: Atoms, size: List[int]) -> Atoms:
    """Build an orthorhombic supercell; size = [nx, ny, nz]."""
    sc = make_supercell(atoms, np.diag(size))
    sc.info["supercell_matrix"] = size
    return sc


# ---------------------------------------------------------------------------
# Unique site identification (species-based, one index per element)
# ---------------------------------------------------------------------------
def unique_site_indices(atoms: Atoms) -> List[int]:
    """
    Return one representative atom index per chemical species.
    Sufficient for supercell vacancy generation where all sites of the
    same element are crystallographically equivalent within the supercell.
    """
    seen: Dict[str, int] = {}
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        if sym not in seen:
            seen[sym] = i
    return list(seen.values())


# ---------------------------------------------------------------------------
# Vacancy generation
# ---------------------------------------------------------------------------
def create_vacancy(atoms: Atoms, site_idx: int) -> Tuple[Atoms, dict]:
    """Remove atom at site_idx; return (new_Atoms, metadata)."""
    sc               = atoms.copy()
    removed_species  = sc.get_chemical_symbols()[site_idx]
    removed_position = sc.get_positions()[site_idx].copy()
    del sc[site_idx]
    meta = {
        "defect_type":      "vacancy",
        "removed_species":  [removed_species],
        "removed_position": removed_position.tolist(),
        "site_index":       site_idx,
        "n_vacancies":      1,
        "n_interstitials":  0,
    }
    return sc, meta


def create_divacancy(atoms: Atoms, idx1: int, idx2: int) -> Tuple[Atoms, dict]:
    """Remove two atoms (higher index deleted first to preserve lower index)."""
    sc    = atoms.copy()
    sym   = sc.get_chemical_symbols()
    pos   = sc.get_positions()
    specs = [sym[idx1], sym[idx2]]
    poss  = [pos[idx1].tolist(), pos[idx2].tolist()]
    for idx in sorted([idx1, idx2], reverse=True):
        del sc[idx]
    meta = {
        "defect_type":       "divacancy",
        "removed_species":   specs,
        "removed_positions": poss,
        "site_indices":      [idx1, idx2],
        "n_vacancies":       2,
        "n_interstitials":   0,
    }
    return sc, meta


# ---------------------------------------------------------------------------
# Nearest-neighbour pair finder (PBC-aware)
# ---------------------------------------------------------------------------
def find_nn_pairs(
    atoms: Atoms, max_dist: float = 5.0, max_pairs: int = 4
) -> List[Tuple[int, int]]:
    """
    Return up to max_pairs nearest-neighbour index pairs within max_dist Å.
    Uses ASE get_distances for minimum-image convention.
    """
    pos  = atoms.get_positions()
    cell = atoms.get_cell()
    pbc  = atoms.get_pbc()

    _, dists = get_distances(pos, cell=cell, pbc=pbc)
    heap: List[Tuple[float, int, int]] = []
    n = len(atoms)

    for i, j in combinations(range(n), 2):
        d = dists[i, j]
        if 0.5 < d < max_dist:
            heap.append((d, i, j))

    heap.sort(key=lambda x: x[0])
    return [(i, j) for _, i, j in heap[:max_pairs]]


# ---------------------------------------------------------------------------
# Interstitial insertion
# ---------------------------------------------------------------------------
def find_void_center(atoms: Atoms) -> np.ndarray:
    """
    Find the approximate void centre via coarse fractional-coordinate grid:
    the grid point whose minimum distance to any atom is maximal.
    """
    cell = np.array(atoms.get_cell())
    pos  = atoms.get_positions()
    grid = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j].reshape(3, -1).T
    cart = grid @ cell
    min_dists = np.min(
        np.linalg.norm(cart[:, None, :] - pos[None, :, :], axis=-1), axis=1
    )
    return cart[np.argmax(min_dists)]


def create_interstitial(atoms: Atoms, species: str) -> Tuple[Atoms, dict]:
    """
    Insert a foreign atom at the largest void centre.

    FIX: use ase.Atom (singular) with Atoms.append().
         Previously, Atoms (plural) was passed, which raises TypeError.
    """
    sc       = atoms.copy()
    void_pos = find_void_center(sc)
    sc.append(Atom(species, position=void_pos))   # correct: Atom, not Atoms
    meta = {
        "defect_type":       "interstitial",
        "inserted_species":  [species],
        "inserted_position": void_pos.tolist(),
        "n_vacancies":       0,
        "n_interstitials":   1,
    }
    return sc, meta


# ---------------------------------------------------------------------------
# Thermal displacements (Debye-Waller)
# ---------------------------------------------------------------------------
def apply_thermal_displacements(atoms: Atoms, sigma: float) -> Atoms:
    """Add isotropic Gaussian displacements ~ N(0, sigma) Å to all positions."""
    sc   = atoms.copy()
    disp = RNG.normal(0.0, sigma, size=sc.get_positions().shape)
    sc.set_positions(sc.get_positions() + disp)
    sc.info["thermal_sigma"] = sigma
    return sc


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _defect_id(base_id: str, defect_type: str, counter: int) -> str:
    return f"{base_id}__{defect_type}__{counter:04d}"


def save_defect(atoms: Atoms, meta: dict, base_id: str, counter: int) -> str:
    """Write extXYZ + JSON sidecar; return defect ID string."""
    did  = _defect_id(base_id, meta["defect_type"], counter)
    stem = os.path.join(DEFECT_DIR, did)
    ase_write(stem + ".xyz", atoms, format="extxyz")
    meta = {
        **meta,
        "defect_id": did,
        "base_id":   base_id,
        "n_atoms":   len(atoms),
        "cell":      atoms.get_cell().tolist(),
        "species":   atoms.get_chemical_symbols(),
    }
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return did


# ---------------------------------------------------------------------------
# Per-structure defect generation
# ---------------------------------------------------------------------------
def generate_defects_for_structure(record: dict) -> List[str]:
    """
    Build supercell from record and generate all defect variants.
    Returns list of defect IDs.
    """
    mid      = record["material_id"]
    cif_path = os.path.join(RAW_DIR, mid + ".cif")
    if not os.path.exists(cif_path):
        log.warning("CIF not found for %s — skipping", mid)
        return []

    try:
        bulk = ase_read(cif_path)
    except Exception as exc:
        log.error("Cannot read %s: %s", cif_path, exc)
        return []

    sc = build_supercell(bulk, CFG.defect.supercell_size)
    sc.info["material_id"] = mid
    log.info("%s → supercell %s (%d atoms)", mid, CFG.defect.supercell_size, len(sc))

    created_ids: List[str] = []
    counter = 0

    def _snapshots(atoms: Atoms, meta: dict) -> None:
        nonlocal counter
        for _ in range(CFG.defect.n_thermal_snapshots):
            displaced = apply_thermal_displacements(atoms, CFG.defect.thermal_sigma)
            m = {**meta, "thermal_snapshot": True,
                 "thermal_sigma": CFG.defect.thermal_sigma}
            created_ids.append(save_defect(displaced, m, mid, counter))
            counter += 1

    # 1. Single vacancies (one per unique element species)
    for site_idx in unique_site_indices(sc):
        vac, meta = create_vacancy(sc, site_idx)
        _snapshots(vac, meta)

    # 2. Divacancies (nearest-neighbour pairs)
    for idx1, idx2 in find_nn_pairs(sc, max_dist=CFG.defect.divacancy_max_dist):
        div, meta = create_divacancy(sc, idx1, idx2)
        _snapshots(div, meta)

    # 3. Interstitials (He, H at largest void)
    for species in CFG.defect.interstitial_species:
        intl, meta = create_interstitial(sc, species)
        _snapshots(intl, meta)

    log.info("%s → created %d defect structures", mid, len(created_ids))
    return created_ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    log.info("=== Stage 2: Defect Engineering ===")

    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        log.error("manifest.json not found — run stage01 first")
        sys.exit(1)

    with open(manifest_path) as fh:
        records = json.load(fh)

    all_ids: List[str] = []
    for i, record in enumerate(records):
        log.info("[%d/%d] %s (%s)", i + 1, len(records),
                 record["material_id"], record["formula"])
        all_ids.extend(generate_defects_for_structure(record))

    defect_manifest = os.path.join(DEFECT_DIR, "defect_manifest.json")
    with open(defect_manifest, "w") as fh:
        json.dump(all_ids, fh, indent=2)
    log.info("Defect manifest written: %d structures total", len(all_ids))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    main()
