"""Benchmark script for Rust eigendecomposition memory usage.

Measures actual peak RSS versus estimated memory to validate/update
memory estimation functions for faer backend.

Run: python tests/benchmark_rust_memory.py
"""

import gc
import json
import os
from datetime import datetime
from typing import NamedTuple

import numpy as np
import psutil

# Available sample sizes (increase for high-memory machines)
SAMPLE_SIZES = [1_000, 5_000, 10_000, 25_000]


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark run."""

    backend: str
    n_samples: int
    estimated_gb: float
    peak_rss_gb: float
    time_seconds: float
    overhead_ratio: float  # peak_rss / estimated


def get_rss_gb() -> float:
    """Get current RSS in GB."""
    return psutil.Process().memory_info().rss / 1e9


def benchmark_eigendecomp_memory(
    n_samples: int,
    backend: str,
) -> BenchmarkResult:
    """Benchmark eigendecomposition memory usage.

    Args:
        n_samples: Matrix size (n x n).
        backend: Either "numpy" or "rust".

    Returns:
        BenchmarkResult with actual vs estimated memory.
    """
    import time

    from jamma.core.memory import estimate_eigendecomp_memory

    # Force garbage collection before measurement
    gc.collect()
    gc.collect()

    # Create symmetric kinship-like matrix
    np.random.seed(42)
    A = np.random.randn(n_samples, n_samples)
    K = (A + A.T) / 2
    del A
    gc.collect()

    rss_before = get_rss_gb()
    estimated_gb = estimate_eigendecomp_memory(n_samples)

    # Set backend
    os.environ["JAMMA_BACKEND"] = (
        f"jax.{backend}" if backend != "numpy" else "jax.numpy"
    )

    # Import after setting backend
    from jamma.core.backend import get_compute_backend

    get_compute_backend.cache_clear()

    start_time = time.perf_counter()

    if backend == "numpy":
        eigenvalues, eigenvectors = np.linalg.eigh(K)
    else:
        import jamma_core

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K, threshold=0.0)

    elapsed = time.perf_counter() - start_time

    # Measure peak RSS (note: this is after eigendecomp, not during)
    rss_after = get_rss_gb()
    peak_rss_gb = rss_after  # Approximation; true peak may be higher during computation

    # Calculate overhead vs estimate
    # Estimate is for eigendecomp workspace; actual includes K + U
    matrix_memory_gb = 2 * (n_samples**2 * 8 / 1e9)  # K + U
    effective_peak = peak_rss_gb - rss_before + matrix_memory_gb
    overhead_ratio = effective_peak / estimated_gb if estimated_gb > 0 else 0

    # Cleanup
    del K, eigenvalues, eigenvectors
    gc.collect()

    return BenchmarkResult(
        backend=backend,
        n_samples=n_samples,
        estimated_gb=estimated_gb,
        peak_rss_gb=peak_rss_gb,
        time_seconds=elapsed,
        overhead_ratio=overhead_ratio,
    )


def run_benchmarks(
    sample_sizes: list[int] | None = None,
    backends: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run benchmarks across all sample sizes and backends."""
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES
    if backends is None:
        backends = ["numpy", "rust"]

    results = []

    for n in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking n_samples={n:,}")
        print(f"{'='*60}")

        for backend in backends:
            try:
                print(f"\n  Backend: {backend}")
                result = benchmark_eigendecomp_memory(n, backend)
                results.append(result)

                print(f"    Time: {result.time_seconds:.2f}s")
                print(f"    Peak RSS: {result.peak_rss_gb:.2f}GB")
                print(f"    Estimated: {result.estimated_gb:.2f}GB")
                print(f"    Overhead ratio: {result.overhead_ratio:.2f}x")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            gc.collect()
            gc.collect()

    return results


def main():
    """Run benchmarks and save results."""
    from jamma.core.backend import is_rust_available

    print("Rust/faer Memory Benchmark")
    print("=" * 60)
    print(f"System memory: {psutil.virtual_memory().total / 1e9:.1f}GB")
    print(f"Available: {psutil.virtual_memory().available / 1e9:.1f}GB")
    print(f"Rust available: {is_rust_available()}")

    backends = ["numpy"]
    if is_rust_available():
        backends.append("rust")
    else:
        print("\nWARNING: Rust backend not available, numpy-only benchmark")

    results = run_benchmarks(backends=backends)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "system_memory_gb": psutil.virtual_memory().total / 1e9,
        "results": [r._asdict() for r in results],
    }

    output_path = "output/benchmark_rust_memory.json"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    # Summary table
    print("\n\nSummary")
    print("=" * 80)
    print(
        f"{'Backend':<10} {'Samples':>10} {'Time (s)':>10} {'RSS (GB)':>10} "
        f"{'Est (GB)':>10} {'Ratio':>8}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r.backend:<10} {r.n_samples:>10,} {r.time_seconds:>10.2f} "
            f"{r.peak_rss_gb:>10.2f} {r.estimated_gb:>10.2f} {r.overhead_ratio:>8.2f}"
        )


if __name__ == "__main__":
    main()
