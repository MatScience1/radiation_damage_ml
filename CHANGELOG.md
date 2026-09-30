# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic
Versioning. Each task in this repository increments the version.

## [0.5.1] - 2026-09-29

Fix mypy internal error in CI on Python 3.9 and 3.10.

### Changed
- `mypy.ini`: removed `follow_imports = skip` for `ase`, `numpy`, `torch`, and
  `spglib`, keeping only `ignore_missing_imports`. Skipping imports can leave
  unresolved placeholder types, which is the likely trigger of the cache
  serialization error.
- `pipeline/stage01_data_mining.py`: `load_structure` narrows the
  `ase.io.read` result (`Atoms | list[Atoms]`) to a single `Atoms`.
- `pipeline/stage02_defect_engineering.py`: `create_supercell` annotates the
  `make_supercell` result as `Atoms`.
- `tests/conftest.py`: `supercell_atoms` annotates the `make_supercell` result
  as `Atoms`.
- `tests/unit/test_physical_validation.py`: `_spacegroup_number` takes an
  `Atoms` argument and reads the spglib dataset defensively.
- `tests/unit/test_stage02.py` and `tests/integration/test_data_integrity.py`:
  assert the `ase.io.read` result is `Atoms` before accessing `.info`.

### Fixed
- `mypy.ini`: added `incremental = False` to the `[mypy]` section. This avoids
  the mypy cache serialization error, `AssertionError: Internal error:
  unresolved placeholder type None`, observed on Python 3.9 and 3.10.
- `.github/workflows/ci.yml`: the mypy step removes `.mypy_cache` and runs
  `mypy pipeline/ --no-incremental --cache-dir=/dev/null --show-error-codes`,
  so no module cache is written.
- `.pre-commit-config.yaml`: the mypy hook runs with `--no-incremental`.

### Notes
- `mypy==1.11.2` was already pinned in `requirements-dev.txt`, keeping the CI
  and local versions consistent.
- `.mypy_cache/` was already present in `.gitignore`.
- Version bumped 0.5.0 to 0.5.1.

## [0.5.0] - 2026-09-29

Project narrative and README rewrite.

### Changed
- `README.md`: rewritten to explain the project goal and scientific
  motivation, the current MVP scope, an implementation diagram, the data flow,
  the roadmap, and separate strengths and limitations sections. Added a mermaid
  flowchart of the pipeline stages and marked planned stages explicitly.

### Fixed
- `.pre-commit-config.yaml`: the mypy hook now installs `pytest` so fixture
  decorators are typed in the hook environment.
- `mypy.ini`: added `disallow_untyped_decorators = False` to `[mypy-tests.*]`
  so pytest fixture functions are accepted under the strict decorator check.

### Notes
- Documentation only for the source pipeline. No source, test, or runtime
  behaviour changed.

## [0.4.0] - 2026-09-29

Enterprise hardening: CI, pre-commit, static typing, packaging, benchmarks.

### Added
- `.github/workflows/ci.yml`: matrix job on Python 3.9, 3.10, 3.11. Caches pip,
  installs `requirements-dev.txt`, runs `ruff check .`, `mypy pipeline/`,
  `pytest --cov=pipeline --cov-report=xml`, and uploads coverage to Codecov.
- `pyproject.toml`: `[build-system]` via setuptools and wheel, `[project]`
  metadata (name `radiation-damage-ml`, `requires-python = ">=3.9"`), `dev`
  optional dependencies, `benchmark` console script, and setuptools package
  discovery for `pipeline*` and `benchmarks*`.
- `setup.py`: setuptools shim that delegates to `pyproject.toml`.
- `mypy.ini`: strict settings for the pipeline and relaxed settings for tests.
- `.pre-commit-config.yaml`: hooks for trailing whitespace, end of file, YAML,
  JSON, large files, merge conflicts, debug statements, ruff, mypy, black, and
  isort.
- `benchmarks/__init__.py` and `benchmarks/benchmark_pipeline.py` with
  functions `benchmark_stage`, `run_benchmarks`, and `main`.
- `benchmarks/README.md`: metrics, baseline, targets, and scaling notes.
- `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`,
  `tests/e2e/__init__.py`: make the test tree a packageable hierarchy.
- `pipeline/__init__.py`: added `__version__ = "0.4.0"`.

### Changed
- `requirements-dev.txt`: pinned toolchain. `mypy==1.11.2`, `ruff==0.6.9`,
  `black==24.8.0`, `isort==5.13.2`, `pre-commit==3.8.0`, `build==1.2.2`, plus
  `pytest>=8.0`, `pytest-cov>=5.0`, `spglib>=2.0`.
- `pyproject.toml`: ruff configuration moved to the `[tool.ruff.lint]` table
  with `select = ["E", "W", "F", "I", "B", "C4", "UP"]` and
  `ignore = ["E501", "B008", "UP006", "UP007", "UP035"]`; `tests/*` ignores
  `S101`. Added `[tool.black]` and `[tool.isort]` sections.
- `README.md`: added Development setup, Code quality, Continuous integration,
  Packaging, and Performance sections.
- `benchmarks/README.md`: baseline table updated to measured values, stage 1
  about 0.30 s and 0.7 MB, stage 2 about 0.04 s and 0.2 MB, total about 0.35 s.
- `tests/integration/test_data_flow.py`: in
  `test_physical_validity_of_defect_structure`, replaced the boolean mask
  `distances[~np.eye(...)]` with `np.fill_diagonal(distances, np.inf)`.
- `.gitignore`: added `coverage.xml`, `build/`, `dist/`, `.mypy_cache/`,
  `.ruff_cache/`, and `benchmarks/benchmark_results.json`.
- Applied black and isort formatting; ruff auto-fixed
  `tests/unit/test_physical_validation.py` by removing an unused `numpy`
  import.

### Fixed
- `benchmarks/benchmark_pipeline.py`: `benchmark_stage` return annotation
  corrected from `Tuple[float, float]` to `Tuple[float, float, Any]` to match
  the three returned values.
- `mypy.ini`: added `disallow_incomplete_defs = False` to `[mypy-tests.*]` so
  untyped fixture parameters are accepted, and `follow_imports = skip` for
  `ase`, `numpy`, `torch`, and `spglib` so their stubs are not parsed.
- Made `tests/` a package so the `[mypy-tests.*]` override matches the test
  modules during `mypy tests/`.

### Notes
- Version bumped 0.3.0 to 0.4.0.
- Mypy targets Python 3.12 for stub compatibility; runtime compatibility with
  3.9 to 3.11 is enforced by the CI test matrix.
- The documentation workflow from the hardening specification was intentionally
  omitted; Markdown documentation is deemed sufficient at MVP stage.

## [0.3.0] - 2026-09-29

Second test suite: physical, mathematical, and data-integrity verification.

### Added
- `tests/unit/test_physical_validation.py`: `_spacegroup_number` helper plus
  `test_mass_conservation_supercell`, `test_composition_conservation`,
  `test_symmetry_preserved_in_supercell`, and
  `test_vacancy_creation_removes_correct_element`.
- `tests/unit/test_mathematical_correctness.py`:
  `test_transformation_matrix_determinant`, `test_volume_scales_correctly`,
  `test_cell_angles_preserved`, `test_fractional_coordinates_valid`.
- `tests/unit/test_numerical_stability.py`: `test_no_atom_overlap`,
  `test_coordinates_in_valid_range`, `test_positions_reasonable`,
  `test_lattice_parameters_physical`.
- `tests/unit/test_reference_validation.py`:
  `test_lattice_constant_matches_literature`, `test_density_calculation_correct`,
  `test_volume_per_atom_reasonable`, `test_mass_per_atom_correct`.
- `tests/unit/test_type_safety.py`: `test_stage1_returns_correct_types`,
  `test_stage2_returns_correct_types`,
  `test_config_attributes_have_correct_types`,
  `test_metadata_fields_have_correct_types`.
- `tests/integration/test_reproducibility.py`: `_md5` helper plus
  `test_pipeline_is_deterministic`, `test_file_hashes_are_stable`,
  `test_metadata_identical_across_runs`.
- `tests/integration/test_edge_cases.py`: `test_handles_non_cubic_crystal`,
  `test_handles_asymmetric_supercell`, `test_handles_large_vacancy_index`,
  `test_handles_first_vacancy_index`.
- `tests/integration/test_data_integrity.py`: `_run_stages` helper plus
  `test_metadata_consistency_between_formats`, `test_atom_count_consistency`,
  `test_file_paths_are_valid`, `test_json_files_are_valid`.
- `tests/conftest.py`: added `hcp_magnesium_atoms` fixture (HCP Mg, 2 atoms)
  and `engine` fixture (a `SimpleDefectEngine` bound to `sample_config`).

### Changed
- Test count increased from 32 to 63: 44 unit, 15 integration, 4 end-to-end.
- `requirements-dev.txt`: added `spglib>=2.0` for spacegroup validation.
- `tests/README.md`: documented the new verification layers, per-file counts,
  coverage, reproducibility guarantees, edge cases, and limitations.
- Test modules use `sample_config.mvp.*` accessors and derive output file names
  from `material_name`.

### Fixed
- `test_extxyz_contains_correct_metadata` in `tests/unit/test_stage02.py`
  compares `supercell_size` as `list(...)`, because extXYZ round-trips the value
  to a NumPy array.
- Symmetry validation uses `spglib.get_symmetry_dataset` directly, because
  `ase.spacegroup.get_spacegroup` is deprecated and requires `spglib`.

### Notes
- CIF output was verified byte-deterministic, allowing the file hash test to
  hash the CIF directly. Coverage remains 83 percent over `pipeline`.

## [0.2.0] - 2026-09-29

First test suite: pytest pyramid with unit, integration, and end-to-end tests.

### Added
- `tests/conftest.py`: fixtures `project_root`, `sample_config` (a `FullConfig`
  writing into `tmp_path`), `temp_output_dir`, `aluminum_atoms` (FCC Al,
  a=4.046, 4 atoms), and `supercell_atoms` (2x2x2, 32 atoms).
- `tests/unit/test_config.py`: `BasePaths` and `MVPConfig` defaults,
  `create_directories` presence and idempotency, and `load_config` behaviour.
- `tests/unit/test_stage01.py`: CIF parsing, atom count, lattice parameters via
  `pytest.approx`, metadata fields, output file writing, `FileNotFoundError`,
  and chemical formula.
- `tests/unit/test_stage02.py`: supercell atom count and cell scaling, vacancy
  removal, index validation, extXYZ and JSON outputs and their metadata.
- `tests/integration/test_data_flow.py`: cross-stage atom and metadata
  transfer, physical validity via minimum image distance, supercell dimensions.
- `tests/e2e/test_pipeline.py`: `run_pipeline.py` from the project root, from
  an arbitrary directory, expected output creation, and missing input handling.
- `pytest.ini`: `testpaths = tests`, `pythonpath = .`, registered `unit`,
  `integration`, and `e2e` markers, and default `addopts`.

### Changed
- `pipeline/stage02_defect_engineering.py`: `SimpleDefectEngine.__init__` now
  calls `config.mvp.create_directories()`, making the stage self-contained.
- `README.md`: rewritten to a concise professional format with overview,
  development status, structure, strengths, weaknesses, and roadmap.
- `requirements-dev.txt`: added `pytest>=8.0` and `pytest-cov>=5.0`.
- `tests/README.md`: initial test documentation.

### Fixed
- `LocalDataMiner` and `SimpleDefectEngine` writes no longer depend on the
  orchestrator pre-creating output directories.

## [0.1.0] - 2026-09-29

MVP refactor: local CIF input, single vacancy, clean tree.

### Added
- `CHANGELOG.md`.
- `materials/Al.cif`: aluminium FCC sample, a=4.046 Angstrom, 4 atoms.
- `pipeline/config.py`: `PROJECT_ROOT` derived from `__file__`, dataclasses
  `BasePaths`, `MVPConfig`, and `FullConfig`, and `load_config`.
- `pipeline/stage01_data_mining.py`: `LocalDataMiner.__init__` and
  `load_structure` (reads a local CIF via `ase.io.read`, writes a normalised
  CIF and a JSON sidecar via `ase.io.write`, raises `FileNotFoundError` and
  `RuntimeError`), and `run_stage01`.
- `pipeline/stage02_defect_engineering.py`: `SimpleDefectEngine.create_supercell`
  (`make_supercell` with `np.diag(supercell_size)`), `create_vacancy` (index
  bounds check, atom deletion), `generate_defect_structure` (extXYZ plus JSON
  sidecar), and `run_stage02`.
- `.gitignore` for `__pycache__`, virtual environments, and generated data.

### Changed
- `pipeline/config.py`: replaced environment-driven configuration with
  dataclasses; all default paths anchored to the project root via `__file__`.
- `pipeline/stage01_data_mining.py`: replaced the Materials Project API path
  with local CIF loading.
- `pipeline/stage02_defect_engineering.py`: reduced to supercell construction
  and a single vacancy; removed divacancy, interstitial, and thermal snapshot
  logic.
- `run_pipeline.py`: reduced to a minimal MVP orchestrator running stage 1 then
  stage 2 and importing directly from the `pipeline` package.
- `requirements.txt`: reduced to `numpy>=1.21.0`, `ase>=3.22.0`,
  `torch>=1.9.0`.
- `outputs/RESULTS_DESCRIPTION.txt`: rewritten for the MVP outputs.
- `README.md`: rewritten to MVP scope with a deferred-features section.

### Removed
- `pipeline/stage03_energy_evaluation.py`,
  `pipeline/stage04_graph_transform.py`, `pipeline/stage05_storage.py`, and
  `pipeline/stage06_dataloader.py`.
- `setup_env.sh` and the `potentials/` directory with its README.
- `ADPConfig` and `GRACEConfig` dataclasses from `pipeline/config.py`.
- Materials Project and pymatgen coupling from stage 1.

### Notes
- All removed code is recoverable from git history and is documented in the
  README as deferred to V1 and later.
