#!/usr/bin/env python3
"""Benchmark jlinalg operations vs system BLAS (via numpy).

Usage:
    uv run python scripts/bench_jlinalg.py
    uv run python scripts/bench_jlinalg.py --runs 10
    uv run python scripts/bench_jlinalg.py --sizes 1000,4000,10000
    uv run python scripts/bench_jlinalg.py --skip-eigh

Requires jlinalg C extension compiled. Reports GFLOPS and throughput ratios.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _best_time(fn, n_warmup: int, n_runs: int) -> float:
    """Run fn with warmup, return best elapsed time in seconds."""
    for _ in range(n_warmup):
        fn()
    best = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def _bench_dgemm(sizes: list[int], n_runs: int) -> list[dict]:
    """Benchmark dgemm at given square sizes."""
    from jamma.jlinalg import dgemm

    results = []
    for sz in sizes:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((sz, sz))
        B = rng.standard_normal((sz, sz))

        t_jlinalg = _best_time(
            lambda _a=A, _b=B: dgemm(_a, _b), n_warmup=2, n_runs=n_runs
        )
        t_numpy = _best_time(
            lambda _a=A, _b=B: np.matmul(_a, _b), n_warmup=2, n_runs=n_runs
        )

        flops = 2.0 * sz * sz * sz
        gf_jlinalg = flops / t_jlinalg / 1e9
        gf_numpy = flops / t_numpy / 1e9
        ratio = gf_jlinalg / gf_numpy

        results.append(
            {
                "size": f"{sz}",
                "gf_jlinalg": gf_jlinalg,
                "gf_numpy": gf_numpy,
                "ratio": ratio,
            }
        )
    return results


def _bench_dsyrk(sizes: list[tuple[int, int]], n_runs: int) -> list[dict]:
    """Benchmark dsyrk at given (N, K) sizes."""
    from jamma.jlinalg import dsyrk

    results = []
    for n, k in sizes:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, k))

        # Reduce runs for large sizes
        actual_runs = min(n_runs, 3) if n >= 10000 else n_runs

        t_jlinalg = _best_time(lambda _x=X: dsyrk(_x), n_warmup=2, n_runs=actual_runs)
        t_numpy = _best_time(
            lambda _x=X: np.dot(_x, _x.T), n_warmup=2, n_runs=actual_runs
        )

        # GFLOPS: N*N*K for dsyrk (standard convention)
        flops = float(n) * n * k
        gf_jlinalg = flops / t_jlinalg / 1e9
        gf_numpy = flops / t_numpy / 1e9
        ratio = gf_jlinalg / gf_numpy

        results.append(
            {
                "size": f"{n}x{k}",
                "gf_jlinalg": gf_jlinalg,
                "gf_numpy": gf_numpy,
                "ratio": ratio,
            }
        )
    return results


def _bench_eigh(sizes: list[int], n_runs: int) -> list[dict]:
    """Benchmark eigh at given sizes."""
    from jamma.jlinalg import eigh

    results = []
    for sz in sizes:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((sz, sz))
        K = A @ A.T  # symmetric positive definite

        t_jlinalg = _best_time(lambda _k=K: eigh(_k.copy()), n_warmup=1, n_runs=n_runs)
        t_numpy = _best_time(
            lambda _k=K: np.linalg.eigh(_k.copy()), n_warmup=1, n_runs=n_runs
        )

        ratio = t_numpy / t_jlinalg  # time ratio (higher = jlinalg faster)

        results.append(
            {
                "size": f"{sz}",
                "t_jlinalg": t_jlinalg,
                "t_numpy": t_numpy,
                "ratio": ratio,
            }
        )
    return results


def _print_gflops_table(title: str, results: list[dict]) -> None:
    """Print a GFLOPS comparison table."""
    print(f"\n{title}")
    print(f"  {'Size':<14} {'jlinalg':>10} {'numpy':>10}  {'Ratio':>7}")
    for r in results:
        print(
            f"  {r['size']:<14} {r['gf_jlinalg']:>8.1f} GF"
            f" {r['gf_numpy']:>8.1f} GF  {r['ratio']:>6.2f}x"
        )


def _print_time_table(title: str, results: list[dict]) -> None:
    """Print a seconds comparison table."""
    print(f"\n{title}")
    print(f"  {'Size':<10} {'jlinalg':>10} {'numpy':>10}  {'Ratio':>7}")
    for r in results:
        print(
            f"  {r['size']:<10} {r['t_jlinalg']:>8.3f}s"
            f" {r['t_numpy']:>8.3f}s  {r['ratio']:>6.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jlinalg operations vs system BLAS (via numpy)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed iterations per operation (default: 5)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Override dgemm sizes (comma-separated, e.g. 1000,4000,10000)",
    )
    parser.add_argument(
        "--skip-eigh",
        action="store_true",
        help="Skip eigh benchmark (slow at large sizes)",
    )
    args = parser.parse_args()

    # Import jlinalg info
    from jamma.jlinalg import blas_backend, jlinalg_isa

    print(f"jlinalg Benchmark (ISA: {jlinalg_isa}, Backend: {blas_backend})")
    print("=" * 60)

    # dgemm sizes
    dgemm_sizes = [1000, 1410, 4000]
    if args.sizes:
        dgemm_sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # dsyrk sizes: (N, K)
    dsyrk_sizes = [(4000, 2000), (10000, 5000)]

    # eigh sizes
    eigh_sizes = [200, 500, 1000, 1940]

    # Run benchmarks
    dgemm_results = _bench_dgemm(dgemm_sizes, args.runs)
    _print_gflops_table("dgemm (C = A @ B)", dgemm_results)

    dsyrk_results = _bench_dsyrk(dsyrk_sizes, args.runs)
    _print_gflops_table("dsyrk (K = X @ X.T, symmetric)", dsyrk_results)

    if not args.skip_eigh:
        eigh_results = _bench_eigh(eigh_sizes, args.runs)
        _print_time_table("eigh (eigendecomposition)", eigh_results)

    print()


if __name__ == "__main__":
    main()
