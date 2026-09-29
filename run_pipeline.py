"""
run_pipeline.py - Master orchestrator for the radiation damage ML pipeline.

MVP scope: runs stage 1 (local CIF load) and stage 2 (single vacancy
generation). Advanced stages 3 to 6 are intentionally excluded from the MVP
and remain in the repository as reference for later versions.
"""

from pipeline import stage01_data_mining, stage02_defect_engineering
from pipeline.config import load_config


def main() -> None:
    """Run the complete MVP pipeline."""
    print("=" * 60)
    print("Radiation Damage ML Pipeline - MVP Version")
    print("=" * 60)

    config = load_config()
    config.mvp.create_directories()

    print("\n[Stage 1] Loading material structure...")
    stage1_result = stage01_data_mining.run_stage01()

    print("\n[Stage 2] Creating defect structure...")
    stage2_result = stage02_defect_engineering.run_stage02(stage1_result)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully (MVP stages)")
    print("=" * 60)
    print(
        f"Defect structure: {stage2_result['metadata']['material_name']} "
        f"single_vacancy, {stage2_result['metadata']['num_atoms_defect']} atoms"
    )


if __name__ == "__main__":
    main()
