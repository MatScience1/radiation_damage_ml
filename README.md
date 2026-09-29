# Radiation Damage ML Pipeline

A pipeline that loads a crystalline structure and generates a single-vacancy
defect from it. This is the foundation for a larger effort to build graph
neural network training datasets for radiation damage in crystalline
materials.

## Development status

MVP. Two stages are implemented and covered by a 32-test suite. The pipeline
is a correct, self-contained baseline, but it does not yet produce energies,
forces, or graph datasets.

## Structure

```
radiation_damage_ml/
├── pipeline/
│   ├── config.py                      # configuration dataclasses
│   ├── stage01_data_mining.py         # local CIF loading
│   └── stage02_defect_engineering.py  # supercell and single vacancy
├── materials/Al.cif                   # sample aluminium FCC input
├── tests/                             # unit, integration, end-to-end tests
├── outputs/RESULTS_DESCRIPTION.txt    # output reference
├── run_pipeline.py                    # orchestrator (stage 1 then stage 2)
├── requirements.txt                   # runtime dependencies
├── requirements-dev.txt               # test dependencies
└── CHANGELOG.md
```

Data flow:

```
materials/Al.cif ──► Stage 1 (LocalDataMiner) ──► data/raw_structures/
                                                      │
                                                      ▼
                       data/defect_structures/ ◄── Stage 2 (SimpleDefectEngine)
```

Stage 1 reads the configured CIF, writes a normalised CIF and a JSON metadata
sidecar. Stage 2 builds a supercell, removes one atom, and writes an extXYZ
file and a JSON sidecar.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

Expected outputs:

```
data/raw_structures/aluminum.cif
data/raw_structures/aluminum.json
data/defect_structures/aluminum_vacancy.extxyz
data/defect_structures/aluminum_vacancy.json
```

Configuration lives in [`pipeline/config.py`](pipeline/config.py) in the
`MVPConfig` dataclass: input file, supercell size, vacancy index, and output
paths. All paths are anchored to the project root, so the pipeline runs from
any working directory.

## Strengths

- Minimal dependency set (numpy, ase, torch) and no external data source.
- Deterministic layout anchored to the project root rather than the current
  working directory.
- Clear separation between stages, each self-contained and independently
  testable.
- Explicit error handling with actionable messages.
- Test suite organised by the test pyramid with coverage reporting.

## Weaknesses

- One material and one defect type (single vacancy) only.
- No energy, force, or formation-energy evaluation.
- No graph dataset or training-ready output.
- End-to-end tests write into the repository `data/` directory.
- No continuous integration configuration yet.

## Roadmap

- V1: LAMMPS with an ADP potential for energies and forces, additional defect
  types (divacancy, interstitial), graph transformation to PyTorch Geometric,
  and LMDB or HDF5 storage.
- V2: GRACE or MACE machine-learning interatomic potentials, Materials Project
  integration, and thermal displacement snapshots.
- V3: Full defect catalogue, large-scale parallel evaluation, and streaming
  dataloaders.

## Further information

- [`CHANGELOG.md`](CHANGELOG.md): change history, including what was removed
  from the MVP and deferred to V1 and later.
- [`tests/README.md`](tests/README.md): test structure, how to run the suite,
  coverage, and known limitations.
- [`outputs/RESULTS_DESCRIPTION.txt`](outputs/RESULTS_DESCRIPTION.txt):
  description of every output file and its fields.

## Development setup

```bash
git clone https://github.com/MatScience1/radiation_damage_ml.git
cd radiation_damage_ml
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Supported interpreters: Python 3.9, 3.10, and 3.11.

## Code quality

```bash
pytest                      # full test suite
ruff check .                # lint
mypy pipeline/              # static type check
black .                     # formatting
isort .                     # import ordering
pre-commit run --all-files  # run every hook
```

Tool configuration lives in [`pyproject.toml`](pyproject.toml) (black, isort,
ruff) and [`mypy.ini`](mypy.ini) (mypy).

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request to main. It executes on Python 3.9, 3.10, and 3.11, runs ruff and
mypy, runs the test suite with coverage, and uploads coverage to Codecov.

## Packaging

The project is installable as a Python package.

```bash
python -m build        # build sdist and wheel into dist/
pip install -e .       # editable install
pip install .          # regular install
```

Installing the package exposes the `benchmark` console script.

## Performance

See [`benchmarks/README.md`](benchmarks/README.md). The MVP pipeline runs in
under one second, and results are written to
`benchmarks/benchmark_results.json`.

```bash
python benchmarks/benchmark_pipeline.py
```
