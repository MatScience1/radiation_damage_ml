"""
stage01_data_mining.py — DFT baseline extraction from the Materials Project.

Downloads bulk crystalline structures for high-entropy alloys and silicon,
filters for high-quality DFT data, and serialises to disk in JSON + CIF.

Output  →  data/raw_structures/
    <material_id>.cif     — crystal structure (VESTA-compatible)
    <material_id>.json    — metadata: energy, formation energy, lattice params
    manifest.json         — consolidated list of all accepted records
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CFG, RAW_DIR

log = logging.getLogger("stage01_data_mining")


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------
def passes_quality_filter(entry: dict) -> bool:
    """Reject entries that do not meet minimum DFT quality criteria."""
    if entry.get("energy_above_hull", 1.0) > CFG.mp.energy_above_hull_max:
        return False
    if entry.get("nsites", 0) > CFG.mp.nsites_max:
        return False
    if entry.get("energy_per_atom") is None:
        return False
    if entry.get("formation_energy_per_atom") is None:
        return False
    return True


def extract_lattice_features(structure: Structure) -> Dict:
    """Return lattice parameters, angles, volume, and symmetry as a flat dict."""
    latt = structure.lattice
    sga  = SpacegroupAnalyzer(structure, symprec=0.1)
    return {
        "a": latt.a, "b": latt.b, "c": latt.c,
        "alpha": latt.alpha, "beta": latt.beta, "gamma": latt.gamma,
        "volume": latt.volume,
        "spacegroup_number": sga.get_space_group_number(),
        "spacegroup_symbol": sga.get_space_group_symbol(),
        "crystal_system":    sga.get_crystal_system(),
    }


def extract_cohesive_energy(entry: dict) -> Optional[float]:
    """
    Approximate cohesive energy per atom.
    For alloys, formation_energy_per_atom is stored directly;
    for elemental solids, -energy_per_atom is a binding-energy proxy.
    """
    epa = entry.get("energy_per_atom")
    return -float(epa) if epa is not None else None


# ---------------------------------------------------------------------------
# Main download routine
# ---------------------------------------------------------------------------
def download_structures(api_key: str, chemsys_list: List[str]) -> List[dict]:
    """
    Query Materials Project for each chemsys, apply quality filters,
    persist CIF + JSON, and return list of enriched metadata dicts.
    """
    records: List[dict] = []
    adaptor = AseAtomsAdaptor()

    with MPRester(api_key) as mpr:
        for chemsys in chemsys_list:
            log.info("Querying MP for chemsys: %s", chemsys)
            try:
                results = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=CFG.mp.fields,
                    all_fields=False,
                )
            except Exception as exc:
                log.error("MP query failed for %s: %s", chemsys, exc)
                continue

            log.info("  Got %d raw entries for %s", len(results), chemsys)
            accepted = 0

            for doc in results:
                entry: dict = {f: getattr(doc, f, None) for f in CFG.mp.fields}
                structure: Optional[Structure] = entry.get("structure")
                if structure is None:
                    continue

                entry["material_id"]                = str(entry["material_id"])
                entry["energy_per_atom"]            = float(entry["energy_per_atom"] or 0)
                entry["formation_energy_per_atom"]  = float(entry["formation_energy_per_atom"] or 0)
                entry["energy_above_hull"]          = float(entry.get("energy_above_hull") or 0)
                entry["nsites"]                     = int(structure.num_sites)

                if not passes_quality_filter(entry):
                    log.debug("  Filtered: %s", entry["material_id"])
                    continue

                lattice_feats = extract_lattice_features(structure)
                cohesive_e    = extract_cohesive_energy(entry)

                record = {
                    "material_id":                entry["material_id"],
                    "formula":                    entry.get("formula_pretty", ""),
                    "chemsys":                    chemsys,
                    "nsites":                     entry["nsites"],
                    "energy_per_atom":            entry["energy_per_atom"],
                    "formation_energy_per_atom":  entry["formation_energy_per_atom"],
                    "energy_above_hull":          entry["energy_above_hull"],
                    "cohesive_energy_per_atom":   cohesive_e,
                    "band_gap":                   float(entry.get("band_gap") or 0.0),
                    "density":                    float(entry.get("density") or 0.0),
                    "frac_coords":                structure.frac_coords.tolist(),
                    "species":                    [str(s) for s in structure.species],
                    "lattice_matrix":             structure.lattice.matrix.tolist(),
                    **lattice_feats,
                }
                records.append(record)
                accepted += 1
                _save_record(record, structure)

            log.info("  Accepted %d / %d for %s", accepted, len(results), chemsys)
            time.sleep(0.5)  # respect MP rate limits

    log.info("Total accepted structures: %d", len(records))
    return records


def _save_record(record: dict, structure: Structure) -> None:
    """Write CIF and JSON sidecar for one structure."""
    mid  = record["material_id"]
    stem = os.path.join(RAW_DIR, mid)
    structure.to(fmt="cif", filename=stem + ".cif")
    meta = {k: v for k, v in record.items() if not isinstance(v, Structure)}
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    log.info("=== Stage 1: Data Mining ===")
    records = download_structures(
        api_key=CFG.mp.api_key,
        chemsys_list=CFG.mp.chemsys_queries,
    )

    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(records, fh, indent=2)
    log.info("Manifest written: %s (%d entries)", manifest_path, len(records))

    if records:
        energies = np.array([r["energy_per_atom"] for r in records])
        log.info(
            "Energy/atom — mean: %.3f  std: %.3f  min: %.3f  max: %.3f eV",
            energies.mean(), energies.std(), energies.min(), energies.max(),
        )
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    main()
