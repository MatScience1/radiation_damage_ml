"""End-to-end tests for the complete MVP pipeline."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


class TestPipelineExecution:
    """Tests for full pipeline execution via run_pipeline.py."""

    def test_pipeline_runs_from_project_root(self, project_root: Path) -> None:
        """The pipeline should run successfully from the project root."""
        result = subprocess.run(
            [sys.executable, "run_pipeline.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Pipeline completed successfully" in result.stdout

    def test_pipeline_runs_from_arbitrary_directory(
        self, project_root: Path, tmp_path: Path
    ) -> None:
        """The pipeline should run from any directory and write to the root."""
        result = subprocess.run(
            [sys.executable, str(project_root / "run_pipeline.py")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (project_root / "data" / "raw_structures").exists()
        assert (project_root / "data" / "defect_structures").exists()

    def test_all_expected_outputs_created(self, project_root: Path) -> None:
        """The pipeline should create all expected output files."""
        subprocess.run(
            [sys.executable, "run_pipeline.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        expected_files = [
            "data/raw_structures/aluminum.cif",
            "data/raw_structures/aluminum.json",
            "data/defect_structures/aluminum_vacancy.extxyz",
            "data/defect_structures/aluminum_vacancy.json",
        ]
        for relative in expected_files:
            assert (project_root / relative).exists(), f"Missing: {relative}"

    def test_pipeline_handles_missing_input_gracefully(
        self, project_root: Path
    ) -> None:
        """A missing input CIF should produce a clear error and non-zero exit."""
        input_file = project_root / "materials" / "Al.cif"
        backup_file = project_root / "materials" / "Al.cif.backup"

        if input_file.exists():
            input_file.rename(backup_file)

        try:
            result = subprocess.run(
                [sys.executable, "run_pipeline.py"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            combined = (result.stderr + result.stdout).lower()
            assert "filenotfounderror" in combined or "not found" in combined
        finally:
            if backup_file.exists():
                backup_file.rename(input_file)
