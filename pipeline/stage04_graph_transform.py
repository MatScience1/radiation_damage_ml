"""
stage04_graph_transform.py — Convert annotated ASE Atoms → PyTorch Geometric Data.

Node features  (N × 5, float32, all normalised to ~[0, 1]):
    atomic_number      / 118
    site_volume        / 50   (Voronoi volume in Å³)
    electronegativity  / 4    (Pauling scale)
    covalent_radius    / 3    (Å)
    is_defect_site           (binary: 1 if within 3 Å of a vacancy/interstitial)

Edge features  (E × 4, float32):
    interatomic distance (Å)
    unit displacement vector dx, dy, dz

Targets:
    y               — total system energy (eV), shape (1,)
    forces          — per-atom forces (eV/Å), shape (N, 3)
    formation_energy — point defect formation energy (eV), shape (1,)

Output  →  data/graphs/
    graphs_raw.pkl   — Python list of torch_geometric.data.Data objects
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from ase import Atoms
from ase.io import read as ase_read
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import NeighborList
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CFG, DEFECT_DIR, EVALUATED_DIR, GRAPH_DIR

log = logging.getLogger("stage04_graph_transform")

ADAPTOR = AseAtomsAdaptor()

# ---------------------------------------------------------------------------
# Elemental property lookup — Pauling electronegativities
# FIX: removed duplicate keys (Al, Hf, Zr, V appeared twice in original)
# ---------------------------------------------------------------------------
_ELECTRONEGATIVITY: Dict[str, float] = {
    "H":  2.20, "He": 0.00,
    "Li": 0.98, "Be": 1.57, "B":  2.04, "C":  2.55, "N":  3.04,
    "O":  3.44, "F":  3.98, "Ne": 0.00,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P":  2.19,
    "S":  2.58, "Cl": 3.16, "Ar": 0.00,
    "K":  0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V":  1.63,
    "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y":  1.22, "Zr": 1.33, "Nb": 1.60,
    "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20,
    "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05,
    "Te": 2.10, "I":  2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Hf": 1.30, "Ta": 1.50,
    "W":  2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Tl": 2.04, "Pb": 2.33, "Bi": 2.02,
}


def get_electronegativity(symbol: str) -> float:
    return _ELECTRONEGATIVITY.get(symbol, 1.5)


def get_covalent_radius(symbol: str) -> float:
    z = atomic_numbers.get(symbol, 1)
    r = covalent_radii[z]
    return float(r) if not np.isnan(r) else 1.5


# ---------------------------------------------------------------------------
# Voronoi site volumes
# ---------------------------------------------------------------------------
def compute_site_volumes(atoms: Atoms) -> np.ndarray:
    """
    Per-atom Voronoi volumes via pymatgen VoronoiNN.
    Falls back to uniform volume / N on failure (e.g. for isolated atoms).
    Note: can be slow for large supercells (>500 atoms).
    """
    try:
        structure = ADAPTOR.get_structure(atoms)
        vnn  = VoronoiNN(allow_pathological=True)
        vols = np.array([
            sum(d["volume"]
                for d in vnn.get_voronoi_polyhedra(structure, i).values())
            for i in range(len(structure))
        ], dtype=np.float32)
    except Exception:
        vol_per_atom = atoms.get_volume() / len(atoms)
        vols = np.full(len(atoms), vol_per_atom, dtype=np.float32)
    return vols


# ---------------------------------------------------------------------------
# Defect site mask
# ---------------------------------------------------------------------------
def get_defect_mask(atoms: Atoms, defect_meta: dict,
                    radius: float = 3.0) -> np.ndarray:
    """
    Binary flag: 1.0 for atoms within `radius` Å of any vacancy /
    interstitial site; 0.0 otherwise.
    """
    mask = np.zeros(len(atoms), dtype=np.float32)
    defect_positions: List[np.ndarray] = []

    removed = defect_meta.get("removed_position")
    if removed is not None:
        defect_positions.append(np.array(removed))
    for p in defect_meta.get("removed_positions", []):
        defect_positions.append(np.array(p))

    ins = defect_meta.get("inserted_position")
    if ins is not None:
        defect_positions.append(np.array(ins))

    if not defect_positions:
        return mask

    pos = atoms.get_positions()
    for d_pos in defect_positions:
        dists = np.linalg.norm(pos - d_pos, axis=1)
        mask[dists < radius] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Node feature matrix
# ---------------------------------------------------------------------------
def build_node_features(atoms: Atoms, defect_meta: dict) -> torch.Tensor:
    """
    Returns float32 tensor of shape (N, 5):
        [atomic_number/118, site_volume/50, electronegativity/4,
         covalent_radius/3, is_defect_site]
    All columns normalised to roughly [0, 1].
    """
    symbols = atoms.get_chemical_symbols()
    an      = atoms.get_atomic_numbers().astype(np.float32) / 118.0
    vols    = compute_site_volumes(atoms) / 50.0
    en      = np.array([get_electronegativity(s) for s in symbols],
                       dtype=np.float32) / 4.0
    cr      = np.array([get_covalent_radius(s) for s in symbols],
                       dtype=np.float32) / 3.0
    dm      = get_defect_mask(atoms, defect_meta)

    return torch.from_numpy(np.stack([an, vols, en, cr, dm], axis=1))


# ---------------------------------------------------------------------------
# Edge construction — radius graph with PBC
# ---------------------------------------------------------------------------
def build_edges(atoms: Atoms, cutoff: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index (2, E) and edge_attr (E, 4) = [dist, dx, dy, dz].
    PBC-aware via ASE NeighborList minimum-image convention.
    """
    pos  = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    n    = len(atoms)

    nl = NeighborList(
        [cutoff / 2.0] * n,
        skin=0.0, self_interaction=False, bothways=True, primitive=True,
    )
    nl.update(atoms)

    src_list, dst_list, dist_list, vec_list = [], [], [], []

    for i in range(n):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            r_ij = pos[j] + offset @ cell - pos[i]
            d    = float(np.linalg.norm(r_ij))
            if d < 1e-6 or d > cutoff:
                continue
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(d)
            vec_list.append(r_ij / d)

    if not src_list:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 4), dtype=torch.float32))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    dists      = torch.tensor(dist_list, dtype=torch.float32).unsqueeze(1)
    vecs       = torch.tensor(vec_list,  dtype=torch.float32)
    edge_attr  = torch.cat([dists, vecs], dim=1)   # (E, 4)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Graph constructor
# ---------------------------------------------------------------------------
def atoms_to_graph(
    atoms:       Atoms,
    defect_meta: dict,
    eval_result: dict,
) -> Data:
    x                    = build_node_features(atoms, defect_meta)
    edge_index, edge_attr = build_edges(atoms, CFG.graph.cutoff_radius)
    pos                  = torch.from_numpy(atoms.get_positions().astype(np.float32))
    z                    = torch.from_numpy(atoms.get_atomic_numbers().astype(np.int64))
    cell                 = torch.from_numpy(np.array(atoms.get_cell()).astype(np.float32))
    y_energy             = torch.tensor([eval_result["total_energy_eV"]], dtype=torch.float32)

    # Forces from NPZ sidecar
    npz_path = os.path.join(EVALUATED_DIR, defect_meta["defect_id"] + ".npz")
    if os.path.exists(npz_path):
        forces = torch.from_numpy(np.load(npz_path)["forces"].astype(np.float32))
    else:
        forces = torch.zeros((len(atoms), 3), dtype=torch.float32)

    fe = eval_result.get("formation_energy_eV")
    y_formation = torch.tensor(
        [fe if fe is not None else float("nan")], dtype=torch.float32
    )

    return Data(
        x=x,
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_energy,
        forces=forces,
        formation_energy=y_formation,
        cell=cell,
        num_nodes=len(atoms),
        defect_id=defect_meta["defect_id"],
        base_id=defect_meta.get("base_id", ""),
        defect_type=defect_meta.get("defect_type", ""),
        n_vacancies=int(defect_meta.get("n_vacancies", 0)),
        backend=eval_result.get("backend", ""),
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_all() -> List[Data]:
    eval_manifest_path = os.path.join(EVALUATED_DIR, "eval_manifest.json")
    if not os.path.exists(eval_manifest_path):
        log.error("eval_manifest.json not found — run stage03 first")
        sys.exit(1)

    with open(eval_manifest_path) as fh:
        eval_results = json.load(fh)

    log.info("Converting %d evaluated structures to graphs", len(eval_results))
    graphs: List[Data] = []
    skipped = 0

    for i, result in enumerate(eval_results):
        did       = result["defect_id"]
        xyz_path  = os.path.join(DEFECT_DIR, did + ".xyz")
        meta_path = os.path.join(DEFECT_DIR, did + ".json")

        if not os.path.exists(xyz_path) or not os.path.exists(meta_path):
            skipped += 1
            continue
        try:
            atoms = ase_read(xyz_path, format="extxyz")
        except Exception as exc:
            log.warning("Cannot read %s: %s", xyz_path, exc)
            skipped += 1
            continue

        with open(meta_path) as fh:
            defect_meta = json.load(fh)

        try:
            graph = atoms_to_graph(atoms, defect_meta, result)
        except Exception as exc:
            log.warning("Graph construction failed for %s: %s", did, exc)
            skipped += 1
            continue

        graphs.append(graph)
        if (i + 1) % 500 == 0:
            log.info("  Processed %d / %d (skipped %d)",
                     i + 1, len(eval_results), skipped)

    log.info("Graph conversion: %d graphs, %d skipped", len(graphs), skipped)
    return graphs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    log.info("=== Stage 4: Graph Transformation ===")
    graphs = process_all()

    if not graphs:
        log.error("No graphs produced — check upstream stages")
        sys.exit(1)

    out_path = os.path.join(GRAPH_DIR, "graphs_raw.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(graphs, fh)
    log.info("Saved %d graphs to %s", len(graphs), out_path)

    s = graphs[0]
    log.info("Sample graph — nodes: %d  edges: %d  node_feat: %s  edge_feat: %s",
             s.num_nodes, s.num_edges, tuple(s.x.shape), tuple(s.edge_attr.shape))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    main()
