#!/usr/bin/env python3
"""Performance benchmarks for the radiation damage ML pipeline.

Measures execution time and peak memory for each pipeline stage and writes
the results to benchmarks/benchmark_results.json.
"""

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from pipeline.config import load_config
from pipeline.stage01_data_mining import LocalDataMiner
from pipeline.stage02_defect_engineering import SimpleDefectEngine


def benchmark_stage(func: Callable[..., Any], *args: Any) -> Tuple[float, float, Any]:
    """Run a callable and return (execution_seconds, peak_memory_mb, result)."""
    tracemalloc.start()
    start_time = time.perf_counter()
    result = func(*args)
    execution_time = time.perf_counter() - start_time
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024.0 / 1024.0
    return execution_time, peak_memory_mb, result


def run_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Run all pipeline benchmarks and return the collected metrics."""
    config = load_config()
    config.mvp.create_directories()

    results: Dict[str, Dict[str, Any]] = {}

    miner = LocalDataMiner(config)
    time1, mem1, result1 = benchmark_stage(miner.load_structure)
    results["stage01_data_mining"] = {
        "execution_time_seconds": time1,
        "peak_memory_mb": mem1,
        "num_atoms_loaded": len(result1["atoms"]),
    }

    engine = SimpleDefectEngine(config)
    time2, mem2, result2 = benchmark_stage(
        engine.generate_defect_structure,
        result1["atoms"],
        result1["metadata"],
    )
    results["stage02_defect_engineering"] = {
        "execution_time_seconds": time2,
        "peak_memory_mb": mem2,
        "num_atoms_defect": len(result2["atoms"]),
    }

    results["total"] = {
        "execution_time_seconds": time1 + time2,
        "peak_memory_mb": max(mem1, mem2),
        "final_num_atoms": len(result2["atoms"]),
    }
    return results


def main() -> None:
    """Run the benchmarks, print the metrics, and save the results."""
    print("=" * 60)
    print("Radiation Damage ML Pipeline - Performance Benchmarks")
    print("=" * 60)

    results = run_benchmarks()

    for stage_name, metrics in results.items():
        print(f"\n{stage_name}:")
        print(f"  Execution time: {metrics['execution_time_seconds']:.4f} seconds")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        if "num_atoms_loaded" in metrics:
            print(f"  Atoms loaded: {metrics['num_atoms_loaded']}")
        if "num_atoms_defect" in metrics:
            print(f"  Defect atoms: {metrics['num_atoms_defect']}")

    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "benchmark_results.json"
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nResults saved to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
