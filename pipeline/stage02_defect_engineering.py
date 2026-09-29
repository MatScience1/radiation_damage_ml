"""
Stage 2: Defect Engineering (MVP version).

Creates a single vacancy in a supercell built from the bulk structure
produced by stage 1. Writes the defective structure to
data/defect_structures/ as extXYZ plus a JSON metadata sidecar.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from pipeline.config import FullConfig, load_config


class SimpleDefectEngine:
    """Creates single vacancy defects in crystal structures."""

    def __init__(self, config: FullConfig) -> None:
        """Initialise the engine and ensure output directories exist.

        Creating the directories here keeps the stage self-contained and
        independent of the orchestrator.

        Args:
            config: Fully populated pipeline configuration.
        """
        self.config = config
        self.config.mvp.create_directories()

    def create_supercell(self, atoms: Atoms) -> Atoms:
        """Build an orthorhombic supercell from the primitive structure.

        Args:
            atoms: Bulk crystal structure.

        Returns:
            The supercell as a new Atoms object.

        Raises:
            RuntimeError: If the supercell cannot be constructed.
        """
        try:
            matrix = np.diag(self.config.mvp.supercell_size)
            supercell = make_supercell(atoms, matrix)
        except Exception as exc:
            raise RuntimeError(f"Failed to create supercell: {exc}") from exc

        print(f"Created supercell: {self.config.mvp.supercell_size}")
        print(f"Supercell atoms: {len(supercell)}")

        return supercell

    def create_vacancy(self, atoms: Atoms) -> Dict[str, Any]:
        """Create a single vacancy by removing one atom.

        Args:
            atoms: Supercell from which to remove an atom.

        Returns:
            A dictionary with keys "atoms" (defective Atoms) and
            "vacancy_index" (int).

        Raises:
            ValueError: If the configured vacancy index is out of range.
            RuntimeError: If the atom cannot be removed.
        """
        vacancy_index: int = self.config.mvp.vacancy_index

        if vacancy_index < 0 or vacancy_index >= len(atoms):
            raise ValueError(
                f"Vacancy index {vacancy_index} out of range "
                f"for a structure with {len(atoms)} atoms."
            )

        try:
            defective = atoms.copy()
            del defective[vacancy_index]
        except Exception as exc:
            raise RuntimeError(f"Failed to create vacancy: {exc}") from exc

        print(f"Created vacancy at index {vacancy_index}")
        print(f"Remaining atoms: {len(defective)}")

        return {"atoms": defective, "vacancy_index": vacancy_index}

    def generate_defect_structure(
        self, atoms: Atoms, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a defective structure from the bulk structure.

        Args:
            atoms: Bulk crystal structure from stage 1.
            metadata: Bulk metadata dictionary from stage 1.

        Returns:
            A dictionary with keys "atoms" (defective Atoms) and "metadata"
            (defect metadata).

        Raises:
            RuntimeError: If the defect structure cannot be written.
        """
        supercell = self.create_supercell(atoms)
        defect_result = self.create_vacancy(supercell)
        defect_atoms: Atoms = defect_result["atoms"]
        vacancy_index: int = defect_result["vacancy_index"]
        material_name: str = self.config.mvp.material_name

        output_file: Path = (
            self.config.mvp.paths.defect_structures / f"{material_name}_vacancy.extxyz"
        )

        defect_atoms.info["material_name"] = material_name
        defect_atoms.info["defect_type"] = "single_vacancy"
        defect_atoms.info["vacancy_index"] = vacancy_index
        defect_atoms.info["supercell_size"] = self.config.mvp.supercell_size

        try:
            defect_atoms.write(str(output_file), format="extxyz")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write defect structure to {output_file}: {exc}"
            ) from exc

        defect_metadata: Dict[str, Any] = {
            "material_name": material_name,
            "defect_type": "single_vacancy",
            "vacancy_index": vacancy_index,
            "supercell_size": self.config.mvp.supercell_size,
            "num_atoms_bulk": len(atoms),
            "num_atoms_defect": len(defect_atoms),
            "bulk_formula": metadata["chemical_formula"],
        }

        metadata_file: Path = (
            self.config.mvp.paths.defect_structures / f"{material_name}_vacancy.json"
        )
        try:
            with open(metadata_file, "w") as handle:
                json.dump(defect_metadata, handle, indent=2)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write defect metadata to {metadata_file}: {exc}"
            ) from exc

        print(f"Saved defect structure to {output_file}")

        return {"atoms": defect_atoms, "metadata": defect_metadata}


def run_stage02(stage1_result: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for stage 2."""
    config = load_config()
    engine = SimpleDefectEngine(config)
    return engine.generate_defect_structure(
        stage1_result["atoms"],
        stage1_result["metadata"],
    )


if __name__ == "__main__":
    raise SystemExit(
        "Run stage02 through run_pipeline.py; it requires the stage 1 result."
    )
