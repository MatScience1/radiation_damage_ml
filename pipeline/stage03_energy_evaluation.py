"""
stage03_energy_evaluation.py — Parallelised energy/force evaluation.

Routing logic:
  n_vacancies <= threshold  →  LAMMPS + ADP (WMoNbZrTiTa.nist.adp.txt)  [fast]
  n_vacancies >  threshold  →  MLIP (GRACE / MACE / SevenNet)             [accurate]

MLIP backend is selected via CFG.eval.mlip.backend (or MLIP_BACKEND env var).
ADP species order for WMoNbZrTiTa must match the potential file header.

Output  →  data/evaluated/
    <defect_id>.npz    — positions, forces, stress arrays (float32)
    <defect_id>.json   — scalar results: energies, backend, defect metadata
    eval_manifest.json — list of all result dicts for stage 4 consumption
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CFG, DEFECT_DIR, EVALUATED_DIR, RAW_DIR

log = logging.getLogger("stage03_energy_evaluation")


# ---------------------------------------------------------------------------
# LAMMPS ADP calculator (fast path)
# ---------------------------------------------------------------------------
def build_lammps_calculator(atoms: Atoms):
    """
    Construct LAMMPS ASE calculator using the WMoNbZrTiTa ADP potential.

    The species list passed to pair_coeff must match the potential file header
    exactly.  CFG.eval.adp_species_order defines that order; elements not
    present in the structure are still listed (LAMMPS requires the full list).
    """
    from ase.calculators.lammpsrun import LAMMPS

    pot_path     = CFG.eval.adp_potential_path
    species_line = " ".join(CFG.eval.adp_species_order)

    if not os.path.exists(pot_path):
        raise FileNotFoundError(
            f"ADP potential not found: {pot_path}\n"
            "Download from NIST IPRP and set ADP_POTENTIAL in .env"
        )

    params = {
        "pair_style": "adp",
        "pair_coeff": [f"* * {pot_path} {species_line}"],
        "units":      "metal",
        "atom_style": "atomic",
    }
    return LAMMPS(
        command=CFG.eval.lammps_cmd,
        parameters=params,
        keep_tmp_files=False,
        tmp_dir=os.path.join(EVALUATED_DIR, "lammps_tmp"),
    )


# ---------------------------------------------------------------------------
# MLIP calculator (accurate path)
# ---------------------------------------------------------------------------
def build_mlip_calculator(atoms: Atoms):
    """
    Load the configured MLIP backend as an ASE calculator.

    Supported backends (set CFG.eval.mlip.backend or MLIP_BACKEND env var):
        "mace"     — MACE-MP-0 universal potential (auto-downloads medium ckpt)
        "grace"    — GRACE via grace-tensorpotential package
        "sevennet" — SevenNet-0 universal MLIP

    If model_path is empty, MACE uses its built-in auto-download.
    For GRACE and SevenNet, model_path must point to a valid checkpoint.
    """
    backend    = CFG.eval.mlip.backend.lower()
    model_path = CFG.eval.mlip.model_path
    device     = CFG.eval.mlip.device
    dtype      = CFG.eval.mlip.dtype

    if backend == "grace":
        return _build_grace_calculator(model_path, device)

    if backend == "sevennet":
        return _build_sevennet_calculator(model_path, device)

    # Default: MACE
    return _build_mace_calculator(model_path, device, dtype)


def _build_mace_calculator(model_path: str, device: str, dtype: str):
    try:
        from mace.calculators import MACECalculator, mace_mp
    except ImportError:
        raise ImportError(
            "mace-torch not installed. Run: pip install mace-torch"
        )

    if model_path:
        log.info("MACE: loading checkpoint from %s", model_path)
        return MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype=dtype,
        )
    else:
        log.info("MACE: auto-downloading MACE-MP-0 medium checkpoint")
        return mace_mp(model="medium", default_dtype=dtype, device=device)


def _build_grace_calculator(model_path: str, device: str):
    try:
        from grace_tensorpotential.calculator import GRACECalculator
    except ImportError:
        raise ImportError(
            "grace-tensorpotential not installed.\n"
            "Install from: https://github.com/ICAMS/grace-tensorpotential\n"
            "Or switch to MLIP_BACKEND=mace."
        )
    if not model_path:
        raise ValueError(
            "GRACE backend requires MLIP_MODEL path to a .pth checkpoint.\n"
            "Set MLIP_MODEL=/path/to/grace_model.pth in .env"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GRACE model not found: {model_path}")

    log.info("GRACE: loading checkpoint from %s on %s", model_path, device)
    return GRACECalculator(model_path=model_path, device=device)


def _build_sevennet_calculator(model_path: str, device: str):
    try:
        from sevenn.sevennet_calculator import SevenNetCalculator
    except ImportError:
        raise ImportError(
            "sevenn not installed. Run: pip install sevenn\n"
            "Or switch to MLIP_BACKEND=mace."
        )
    ckpt = model_path or "7net-0"  # SevenNet built-in universal model ID
    log.info("SevenNet: loading model '%s' on %s", ckpt, device)
    return SevenNetCalculator(ckpt, device=device)


# ---------------------------------------------------------------------------
# Pristine supercell reference energy (for formation energy)
# ---------------------------------------------------------------------------
_pristine_cache: Dict[str, float] = {}


def get_pristine_energy(base_id: str) -> Optional[float]:
    """
    Derive total pristine supercell energy from Materials Project metadata.
    E_pristine = energy_per_atom × n_primitive_sites × supercell_volume_factor
    Result is cached per base_id within this process.
    """
    if base_id in _pristine_cache:
        return _pristine_cache[base_id]

    json_path = os.path.join(RAW_DIR, base_id + ".json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as fh:
        meta = json.load(fh)

    epa    = meta.get("energy_per_atom")
    nsites = meta.get("nsites", 1)
    if epa is None:
        return None

    sc_factor       = int(np.prod(CFG.defect.supercell_size))
    e_pristine      = float(epa) * nsites * sc_factor
    _pristine_cache[base_id] = e_pristine
    return e_pristine


# ---------------------------------------------------------------------------
# Defect formation energy
# ---------------------------------------------------------------------------
def compute_formation_energy(
    e_defect:   float,
    e_pristine: float,
    defect_meta: dict,
    chemical_potentials: Optional[Dict[str, float]] = None,
) -> float:
    """
    E_f = E_defect − E_pristine
          + Σ_{removed} μ_i  − Σ_{inserted} μ_i

    Chemical potentials default to zero when not provided.
    """
    mu       = chemical_potentials or {}
    delta_mu = 0.0

    for sp in defect_meta.get("removed_species", []):
        delta_mu += mu.get(sp, 0.0)
    for sp in defect_meta.get("inserted_species", []):
        delta_mu -= mu.get(sp, 0.0)

    return e_defect - e_pristine + delta_mu


# ---------------------------------------------------------------------------
# Core evaluation (runs inside each Ray worker / multiprocessing worker)
# ---------------------------------------------------------------------------
def evaluate_single(defect_id: str) -> Optional[dict]:
    """
    Load one defect structure, run the appropriate calculator, compute
    formation energy, and write NPZ + JSON results.
    """
    xyz_path  = os.path.join(DEFECT_DIR, defect_id + ".xyz")
    meta_path = os.path.join(DEFECT_DIR, defect_id + ".json")

    if not os.path.exists(xyz_path) or not os.path.exists(meta_path):
        log.error("Missing files for %s", defect_id)
        return None

    with open(meta_path) as fh:
        defect_meta = json.load(fh)

    try:
        atoms = ase_read(xyz_path, format="extxyz")
    except Exception as exc:
        log.error("Cannot read %s: %s", xyz_path, exc)
        return None

    n_vac     = defect_meta.get("n_vacancies", 0)
    use_mlip  = (
        CFG.eval.use_mlip_for_complex
        and n_vac > CFG.eval.mlip_complexity_threshold
    )

    try:
        if use_mlip:
            calc    = build_mlip_calculator(atoms)
            backend = f"mlip_{CFG.eval.mlip.backend}"
        else:
            calc    = build_lammps_calculator(atoms)
            backend = "lammps_adp"

        atoms.calc = calc
        energy = atoms.get_potential_energy()   # eV
        forces = atoms.get_forces()             # eV/Å,  (N, 3)
        stress = atoms.get_stress()             # Voigt, eV/Å³

    except Exception as exc:
        log.error("Calculator failed for %s: %s", defect_id, exc)
        log.debug(traceback.format_exc())
        return None

    base_id     = defect_meta.get("base_id", "")
    e_pristine  = get_pristine_energy(base_id)
    e_formation = None
    if e_pristine is not None:
        e_formation = compute_formation_energy(energy, e_pristine, defect_meta)

    result = {
        "defect_id":           defect_id,
        "base_id":             base_id,
        "backend":             backend,
        "total_energy_eV":     float(energy),
        "energy_per_atom_eV":  float(energy / len(atoms)),
        "formation_energy_eV": float(e_formation) if e_formation is not None else None,
        "max_force_eV_A":      float(np.linalg.norm(forces, axis=1).max()),
        "rms_force_eV_A":      float(np.sqrt((forces ** 2).mean())),
        "n_atoms":             len(atoms),
        **{k: defect_meta.get(k) for k in [
            "defect_type", "n_vacancies", "n_interstitials",
            "thermal_snapshot", "thermal_sigma",
        ]},
    }

    out_stem = os.path.join(EVALUATED_DIR, defect_id)
    np.savez_compressed(
        out_stem + ".npz",
        positions=atoms.get_positions().astype(np.float32),
        forces=forces.astype(np.float32),
        stress=stress.astype(np.float32),
        atomic_numbers=atoms.get_atomic_numbers(),
        cell=np.array(atoms.get_cell()).astype(np.float32),
    )
    with open(out_stem + ".json", "w") as fh:
        json.dump(result, fh, indent=2)

    return result


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------
if RAY_AVAILABLE:
    @ray.remote(num_cpus=1)
    def _ray_evaluate(defect_id: str) -> Optional[dict]:
        return evaluate_single(defect_id)


# ---------------------------------------------------------------------------
# Batch dispatcher
# ---------------------------------------------------------------------------
def run_batch(defect_ids: List[str]) -> List[dict]:
    """Dispatch evaluations via Ray cluster, local Ray, or multiprocessing."""
    results = []

    if RAY_AVAILABLE:
        ray_address = os.environ.get("RAY_ADDRESS", None)
        if ray_address:
            log.info("Connecting to Ray cluster at %s", ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            log.info("Starting local Ray (CPUs: %s)",
                     CFG.eval.ray_num_cpus or "auto")
            ray.init(num_cpus=CFG.eval.ray_num_cpus,
                     num_gpus=CFG.eval.ray_num_gpus,
                     ignore_reinit_error=True)

        futures = []
        for i in range(0, len(defect_ids), CFG.eval.batch_size):
            chunk = defect_ids[i:i + CFG.eval.batch_size]
            futures.extend([_ray_evaluate.remote(did) for did in chunk])
            log.info("Submitted %d–%d / %d",
                     i, min(i + CFG.eval.batch_size, len(defect_ids)),
                     len(defect_ids))

        results = ray.get(futures)
        ray.shutdown()

    else:
        log.warning("Ray unavailable — falling back to multiprocessing")
        from multiprocessing import Pool, cpu_count
        n_proc = CFG.eval.ray_num_cpus or cpu_count()
        log.info("Multiprocessing with %d workers", n_proc)
        with Pool(processes=n_proc) as pool:
            results = pool.map(evaluate_single, defect_ids)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    log.info("=== Stage 3: Energy Evaluation ===")
    log.info("Fast path  : LAMMPS ADP  (%s)", CFG.eval.adp_potential_path)
    log.info("Accurate   : MLIP backend=%s  model=%s",
             CFG.eval.mlip.backend,
             CFG.eval.mlip.model_path or "<auto>")
    log.info("Switch at  : n_vacancies > %d", CFG.eval.mlip_complexity_threshold)

    defect_manifest = os.path.join(DEFECT_DIR, "defect_manifest.json")
    if not os.path.exists(defect_manifest):
        log.error("defect_manifest.json not found — run stage02 first")
        sys.exit(1)

    with open(defect_manifest) as fh:
        defect_ids: List[str] = json.load(fh)

    log.info("Evaluating %d defect structures", len(defect_ids))
    os.makedirs(os.path.join(EVALUATED_DIR, "lammps_tmp"), exist_ok=True)

    results = run_batch(defect_ids)

    eval_manifest = os.path.join(EVALUATED_DIR, "eval_manifest.json")
    with open(eval_manifest, "w") as fh:
        json.dump(results, fh, indent=2)

    form_ens   = [r["formation_energy_eV"] for r in results
                  if r.get("formation_energy_eV") is not None]
    max_forces = [r["max_force_eV_A"] for r in results]

    log.info("Evaluated %d / %d successfully", len(results), len(defect_ids))
    if form_ens:
        log.info("Formation energy — mean: %.3f  std: %.3f  min: %.3f  max: %.3f eV",
                 np.mean(form_ens), np.std(form_ens),
                 np.min(form_ens), np.max(form_ens))
    log.info("Max force        — mean: %.4f  max: %.4f eV/Å",
             np.mean(max_forces), np.max(max_forces))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    main()
