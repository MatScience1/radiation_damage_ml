"""
run_pipeline.py — Master orchestrator for the radiation damage ML pipeline.

Runs all six stages in sequence.  Checkpoint detection allows resuming
from any stage without reprocessing earlier outputs.

IMPORTANT — original bug fixed:
    Python cannot import modules whose names start with a digit (01_data_mining).
    Stages are now named stage01_data_mining etc. and imported via
    importlib.util.spec_from_file_location to load by file path, which
    sidesteps any remaining naming constraints entirely.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --start-stage 3   # resume from stage 3
    python run_pipeline.py --only-stage 4    # run stage 4 only
    python run_pipeline.py --only-stage 5 --force  # re-run even if checkpoint exists

    STORAGE_BACKEND=hdf5 python run_pipeline.py
    MLIP_BACKEND=grace   python run_pipeline.py
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.config import (
    CFG, RAW_DIR, DEFECT_DIR, EVALUATED_DIR, GRAPH_DIR, LMDB_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline_run.log", mode="a"),
    ],
)
log = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Stage registry
# (stage_num → (display_name, filename_stem_in_pipeline/))
# ---------------------------------------------------------------------------
STAGES = {
    1: ("Data Mining",          "stage01_data_mining"),
    2: ("Defect Engineering",   "stage02_defect_engineering"),
    3: ("Energy Evaluation",    "stage03_energy_evaluation"),
    4: ("Graph Transformation", "stage04_graph_transform"),
    5: ("Cloud Storage",        "stage05_storage"),
    6: ("Dataloader Check",     "stage06_dataloader"),
}

# Checkpoint files that signal a stage completed successfully
STAGE_CHECKPOINTS = {
    1: os.path.join(RAW_DIR,       "manifest.json"),
    2: os.path.join(DEFECT_DIR,    "defect_manifest.json"),
    3: os.path.join(EVALUATED_DIR, "eval_manifest.json"),
    4: os.path.join(GRAPH_DIR,     "graphs_raw.pkl"),
    5: LMDB_PATH,
    6: os.path.join(GRAPH_DIR,     "splits.json"),
}

PIPELINE_DIR = Path(__file__).resolve().parent


def checkpoint_exists(stage: int) -> bool:
    path = STAGE_CHECKPOINTS.get(stage)
    return path is not None and os.path.exists(path)


def load_stage_module(stage_num: int):
    """
    Load a stage module by file path.
    This avoids importlib.import_module limitations with numeric/hyphenated names.
    """
    _, file_stem = STAGES[stage_num]
    stage_file   = PIPELINE_DIR / "pipeline" / f"{file_stem}.py"
    if not stage_file.exists():
        raise FileNotFoundError(f"Stage file not found: {stage_file}")
    spec   = importlib.util.spec_from_file_location(file_stem, stage_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_stage(stage_num: int) -> None:
    name, _ = STAGES[stage_num]
    log.info("=" * 60)
    log.info("  STAGE %d: %s", stage_num, name)
    log.info("=" * 60)
    t0 = time.time()
    try:
        module = load_stage_module(stage_num)
        module.main()
    except Exception as exc:
        log.error("Stage %d FAILED: %s", stage_num, exc, exc_info=True)
        raise
    log.info("Stage %d completed in %.1f s", stage_num, time.time() - t0)


def parse_args():
    p = argparse.ArgumentParser(description="Radiation Damage ML Pipeline")
    p.add_argument("--start-stage", type=int, default=1,
                   help="Resume from this stage (1–6)")
    p.add_argument("--only-stage", type=int, default=None,
                   help="Run only this single stage")
    p.add_argument("--force", action="store_true",
                   help="Re-run stages even when checkpoint exists")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    stages_to_run = (
        [args.only_stage]
        if args.only_stage is not None
        else list(range(args.start_stage, 7))
    )

    log.info("Pipeline configuration:")
    log.info("  Stages        : %s", stages_to_run)
    log.info("  MP chemsys    : %s", CFG.mp.chemsys_queries)
    log.info("  Supercell     : %s", CFG.defect.supercell_size)
    log.info("  Cutoff        : %.1f Å", CFG.graph.cutoff_radius)
    log.info("  Storage       : %s", CFG.storage.backend)
    log.info("  MLIP backend  : %s  (threshold n_vac > %d)",
             CFG.eval.mlip.backend, CFG.eval.mlip_complexity_threshold)
    log.info("  ADP potential : %s", CFG.eval.adp_potential_path)

    t_pipeline = time.time()
    for stage_num in stages_to_run:
        if not args.force and checkpoint_exists(stage_num):
            log.info("Stage %d: checkpoint found — skipping (--force to override)",
                     stage_num)
            continue
        run_stage(stage_num)

    total = time.time() - t_pipeline
    log.info("=" * 60)
    log.info("  Pipeline complete in %.1f s (%.1f min)", total, total / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
