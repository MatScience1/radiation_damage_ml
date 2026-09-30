"""
Stage 1: Data Mining (MVP version).

Loads crystal structures from a local CIF file instead of querying the
Materials Project API. Writes the structure to data/raw_structures/ together
with a JSON metadata sidecar.
"""

import json
from pathlib import Path
from typing import Any, Dict

from ase.io import read, write

from pipeline.config import FullConfig, load_config


class LocalDataMiner:
    """Loads crystal structures from local CIF files."""

    def __init__(self, config: FullConfig) -> None:
        """Initialise the miner and ensure the output directories exist.

        Args:
            config: Fully populated pipeline configuration.
        """
        self.config = config
        self.config.mvp.create_directories()

    def load_structure(self) -> Dict[str, Any]:
        """Load a crystal structure from the configured local CIF file.

        Returns:
            A dictionary with keys "atoms" (ase.Atoms) and "metadata" (dict).

        Raises:
            FileNotFoundError: If the configured CIF file does not exist.
            RuntimeError: If the structure cannot be read or serialised.
        """
        material_file: Path = self.config.mvp.material_file
        material_name: str = self.config.mvp.material_name

        if not material_file.exists():
            raise FileNotFoundError(
                f"Material file not found: {material_file}. "
                "Provide a valid CIF file via MVPConfig.material_file."
            )

        try:
            atoms_or_list = read(str(material_file))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read structure from {material_file}: {exc}"
            ) from exc

        if isinstance(atoms_or_list, list):
            atoms = atoms_or_list[0]
        else:
            atoms = atoms_or_list

        if len(atoms) == 0:
            raise RuntimeError(
                f"Structure read from {material_file} contains no atoms."
            )

        output_file: Path = (
            self.config.mvp.paths.raw_structures / f"{material_name}.cif"
        )
        try:
            write(str(output_file), atoms)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write structure to {output_file}: {exc}"
            ) from exc

        cellpar = atoms.cell.cellpar()
        metadata: Dict[str, Any] = {
            "material_name": material_name,
            "num_atoms": len(atoms),
            "chemical_formula": atoms.get_chemical_formula(),
            "lattice_parameters": {
                "a": float(cellpar[0]),
                "b": float(cellpar[1]),
                "c": float(cellpar[2]),
                "alpha": float(cellpar[3]),
                "beta": float(cellpar[4]),
                "gamma": float(cellpar[5]),
            },
            "source": "local_file",
            "file_path": str(material_file),
        }

        metadata_file: Path = (
            self.config.mvp.paths.raw_structures / f"{material_name}.json"
        )
        try:
            with open(metadata_file, "w") as handle:
                json.dump(metadata, handle, indent=2)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write metadata to {metadata_file}: {exc}"
            ) from exc

        print(f"Loaded structure: {metadata['chemical_formula']}")
        print(f"Number of atoms: {metadata['num_atoms']}")

        return {"atoms": atoms, "metadata": metadata}


def run_stage01() -> Dict[str, Any]:
    """Main entry point for stage 1."""
    config = load_config()
    miner = LocalDataMiner(config)
    return miner.load_structure()


if __name__ == "__main__":
    run_stage01()
