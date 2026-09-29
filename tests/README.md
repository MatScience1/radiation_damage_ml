# Test Suite

Documentation for the radiation_damage_ml MVP test suite.

## Scope

The suite covers the two implemented stages:

- Stage 1: [`pipeline/stage01_data_mining.py`](../pipeline/stage01_data_mining.py),
  local CIF loading.
- Stage 2: [`pipeline/stage02_defect_engineering.py`](../pipeline/stage02_defect_engineering.py),
  supercell construction and single vacancy generation.

Tests follow the test pyramid: many fast unit tests, a moderate number of
integration tests, and a small number of end-to-end tests.

## Layout

```
tests/
├── conftest.py                          # shared fixtures
├── unit/
│   ├── test_config.py                   # configuration dataclasses (8)
│   ├── test_stage01.py                  # LocalDataMiner (7)
│   ├── test_stage02.py                  # SimpleDefectEngine (9)
│   ├── test_physical_validation.py      # mass, composition, symmetry (4)
│   ├── test_mathematical_correctness.py # matrix, volume, angles (4)
│   ├── test_numerical_stability.py      # overlap, finite values (4)
│   ├── test_reference_validation.py     # literature values (4)
│   └── test_type_safety.py              # return and attribute types (4)
├── integration/
│   ├── test_data_flow.py                # stage 1 to stage 2 flow (4)
│   ├── test_reproducibility.py          # determinism and hashes (3)
│   ├── test_edge_cases.py               # non-cubic, asymmetric, indices (4)
│   └── test_data_integrity.py           # cross-format consistency (4)
└── e2e/
    └── test_pipeline.py                 # full pipeline via subprocess (4)
```

Totals: 63 tests, split 44 unit, 15 integration, 4 end-to-end.

## Verification layers

- Physical validation: total mass scales by the supercell factor, chemical
  composition is preserved, the spacegroup is retained after expansion (via
  spglib), and vacancy removal removes one atom of the target element.
- Mathematical correctness: the transformation matrix determinant is 8, cell
  volume scales by the determinant, cubic cell angles are preserved, and all
  fractional coordinates lie within the unit cell.
- Numerical stability: no atom overlaps below 2.0 A, positions and cell values
  are finite, atoms lie inside the cell in fractional space, and lattice
  lengths and angles are physically valid.
- Reference validation: the aluminium lattice constant, density (about 2.70
  g/cm^3), volume per atom (about 16.6 A^3), and mean atomic mass (about 26.98
  amu) match literature values.
- Type safety: stage 1 and stage 2 return dictionaries containing `Atoms` and
  `dict` objects, and configuration and metadata fields carry the documented
  types.
- Reproducibility: repeated runs yield identical positions, metadata, and
  byte-identical CIF output. The ASE CIF writer is deterministic for a fixed
  structure.
- Edge cases: a non-cubic HCP crystal (magnesium), an asymmetric supercell
  (2x3x4), and both the first and last valid vacancy indices.
- Data integrity: JSON and extXYZ metadata agree, atom counts are consistent
  across all outputs, every output path exists and is non-empty, and all JSON
  outputs parse.

## Fixtures

Defined in [`tests/conftest.py`](conftest.py):

- `project_root`: absolute path to the repository root.
- `sample_config`: a `FullConfig` whose inputs point at `materials/Al.cif` and
  whose output directories live inside a pytest temporary directory. Tests
  never write into the repository `data/` directory through this fixture.
- `temp_output_dir`: the pytest temporary directory.
- `aluminum_atoms`: conventional cubic aluminium FCC cell, 4 atoms, a=4.046 A.
- `supercell_atoms`: 2x2x2 supercell of `aluminum_atoms`, 32 atoms.
- `hcp_magnesium_atoms`: hexagonal close-packed magnesium, 2 atoms.
- `engine`: a `SimpleDefectEngine` bound to `sample_config`.

## Running the tests

```bash
# Install runtime and test dependencies
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt

# Run everything
pytest

# Run by tier
pytest -m unit
pytest -m integration
pytest -m e2e

# Coverage over the pipeline package
pytest --cov=pipeline --cov-report=term-missing
```

`pytest.ini` sets `testpaths = tests`, `pythonpath = .`, and registers the
`unit`, `integration`, and `e2e` markers.

## Coverage

Last measured with pytest 9 and coverage 7 on Python 3.12.

```
Name                                     Stmts   Miss  Cover
--------------------------------------------------------------
pipeline/__init__.py                         0      0   100%
pipeline/config.py                          37      1    97%
pipeline/stage01_data_mining.py             43     11    74%
pipeline/stage02_defect_engineering.py      62     12    81%
--------------------------------------------------------------
TOTAL                                      142     24    83%
```

Uncovered lines are almost entirely defensive error branches, for example
failed `ase.io` reads and JSON write failures, which are intentionally hard to
trigger in a normal run.

## Known limitations

- The end-to-end tests execute `run_pipeline.py` directly, so they write into
  the repository `data/` directory. That directory is git-ignored. Unit and
  integration tests are fully isolated through `tmp_path`.
- Symmetry validation depends on `spglib`, declared in `requirements-dev.txt`.
  It is not required to run the pipeline itself.
- No numerical regression baselines are stored yet, so tests assert structural
  and reference correctness rather than pinned energy or force values.
- There is no continuous integration configuration in the repository yet.
- The installed ASE version emits `DeprecationWarning` messages from a NumPy
  2.5 shape assignment, and spglib emits a deprecation warning about error
  handling. These are external to this project and do not affect results.

## Future test additions

When later stages are reintroduced, extend the suite as follows.

- V1 (LAMMPS ADP and energy evaluation): unit tests for the calculator builder
  with a mocked LAMMPS interface, numerical tolerance tests for formation
  energy, and integration tests over the evaluated NPZ and JSON outputs.
- V1 (graph transform): shape and dtype assertions on node and edge feature
  tensors, periodic boundary condition edge tests, and a small reference graph.
- V1 (storage): round-trip tests for LMDB and HDF5 writers against their
  readers.
- V2 (MLIP and Materials Project): isolate external services behind interfaces
  and test them with recorded fixtures rather than live calls.
- Additional defect types (divacancy, interstitial, thermal snapshots) with
  per-partition parametrised tests.
- A CI workflow that runs `pytest -m "unit or integration"` on every push and
  the full suite on merge.
