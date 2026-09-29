# Performance Benchmarks

Performance benchmarks for the radiation damage ML pipeline.

## Running benchmarks

```bash
python benchmarks/benchmark_pipeline.py
```

When the package is installed, the same entry point is available as:

```bash
benchmark
```

## Metrics collected

- Execution time (seconds)
- Peak memory usage (MB)
- Number of atoms processed

Results are written to `benchmarks/benchmark_results.json`.

## Baseline performance

MVP stage, 2x2x2 supercell with a single vacancy. Values are approximate,
depend on the machine, and are measured by tracemalloc, which tracks
Python-level allocations only. Stage 1 includes interpreter warm-up on the
first call.

| Stage | Time (s) | Memory (MB) | Atoms |
|-------|----------|-------------|-------|
| Stage 1 (load CIF) | about 0.30 | about 0.7 | 4 |
| Stage 2 (defect) | about 0.04 | about 0.2 | 31 |
| Total | about 0.35 | about 0.7 | 31 |

## Performance targets

- Stage 1: under 1 second for a single CIF file.
- Stage 2: under 1 second for a 2x2x2 supercell.
- Total pipeline: under 2 seconds for the MVP workflow.

## Scaling considerations

- 3x3x3 supercell: about 3.4 times the atom count, roughly linear time.
- 4x4x4 supercell: about 8 times the atom count, roughly linear time.
- Memory usage scales linearly with the atom count.

## Future benchmarks

V1 and V2 will add benchmarks for LAMMPS evaluation, MLIP inference, graph
transformation, and storage I/O.
