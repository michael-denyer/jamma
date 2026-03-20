"""Benchmark lazy vs standard eigendecomposition on mouse_hs1940.

Compares:
1. eigendecompose_kinship (standard) — returns eigenvalues + full U matrix
2. eigendecompose_kinship_lazy — returns LazyEigen with rotate()

For a fair comparison, the lazy path also rotates W and y (the same work
that runners do post-eigendecomp), so we measure the full cost including
on-the-fly rotation vs pre-computed U.T @ target.

Usage:
    uv run python scripts/bench_lazy_eigen.py [--runs N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.eigen import eigendecompose_kinship, eigendecompose_kinship_lazy

KINSHIP_PATH = "tests/fixtures/mouse_hs1940/mouse_hs1940_kinship.cXX.txt"


def _make_targets(n: int, n_covariates: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Create W (intercept + covariates) and y for rotation benchmarking."""
    rng = np.random.default_rng(42)
    W = np.ones((n, 1 + n_covariates), dtype=np.float64)
    W[:, 1:] = rng.standard_normal((n, n_covariates))
    y = rng.standard_normal(n).astype(np.float64)
    return W, y


def bench_standard(K: np.ndarray, W: np.ndarray, y: np.ndarray) -> dict:
    """Benchmark standard eigendecomp + U.T @ targets."""
    K_copy = K.copy()

    t0 = time.perf_counter()
    eigenvalues, U = eigendecompose_kinship(K_copy, check_memory=False)
    t_eigen = time.perf_counter() - t0

    t0 = time.perf_counter()
    UtW = U.T @ W
    Uty = U.T @ y
    t_rotate = time.perf_counter() - t0

    mem_gb = (K_copy.nbytes + U.nbytes + eigenvalues.nbytes) / 1e9

    return {
        "eigenvalues": eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "t_eigen": t_eigen,
        "t_rotate": t_rotate,
        "t_total": t_eigen + t_rotate,
        "mem_gb": mem_gb,
    }


def bench_lazy(K: np.ndarray, W: np.ndarray, y: np.ndarray) -> dict:
    """Benchmark lazy eigendecomp + rotate() targets."""
    K_copy = K.copy()

    t0 = time.perf_counter()
    lazy = eigendecompose_kinship_lazy(K_copy, check_memory=False)
    t_eigen = time.perf_counter() - t0

    t0 = time.perf_counter()
    UtW = lazy.rotate(W)
    Uty = lazy.rotate(y)
    t_rotate = time.perf_counter() - t0

    mem_gb = lazy.memory_gb

    return {
        "eigenvalues": lazy.eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "t_eigen": t_eigen,
        "t_rotate": t_rotate,
        "t_total": t_eigen + t_rotate,
        "mem_gb": mem_gb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark lazy vs standard eigendecomp"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs (best-of-N)"
    )
    args = parser.parse_args()

    print(f"Loading kinship matrix from {KINSHIP_PATH}...")
    K = read_kinship_matrix(Path(KINSHIP_PATH))
    n = K.shape[0]
    print(f"Matrix size: {n} x {n} ({K.nbytes / 1e6:.1f} MB)\n")

    W, y = _make_targets(n)

    # Warmup (compile C extension paths)
    print("Warmup run...")
    bench_standard(K, W, y)
    bench_lazy(K, W, y)

    print(f"\nBenchmarking ({args.runs} runs, best-of)...\n")

    std_results = []
    lazy_results = []

    for i in range(args.runs):
        print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
        s = bench_standard(K, W, y)
        lz = bench_lazy(K, W, y)
        std_results.append(s)
        lazy_results.append(lz)
        print(f"std={s['t_total']:.3f}s  lazy={lz['t_total']:.3f}s")

    best_std = min(std_results, key=lambda r: r["t_total"])
    best_lazy = min(lazy_results, key=lambda r: r["t_total"])

    # Verify numerical equivalence
    eval_diff = np.max(np.abs(best_std["eigenvalues"] - best_lazy["eigenvalues"]))
    utw_diff = np.max(np.abs(np.abs(best_std["UtW"]) - np.abs(best_lazy["UtW"])))
    uty_diff = np.max(np.abs(np.abs(best_std["Uty"]) - np.abs(best_lazy["Uty"])))

    print("\n" + "=" * 65)
    print(f"  RESULTS — mouse_hs1940 ({n} samples)")
    print("=" * 65)
    print(f"{'':30s} {'Standard':>12s} {'Lazy':>12s} {'Ratio':>8s}")
    print("-" * 65)
    print(
        f"{'Eigendecomp (s)':30s} "
        f"{best_std['t_eigen']:12.3f} "
        f"{best_lazy['t_eigen']:12.3f} "
        f"{best_lazy['t_eigen'] / best_std['t_eigen']:7.2f}x"
    )
    print(
        f"{'Rotate W+y (s)':30s} "
        f"{best_std['t_rotate']:12.6f} "
        f"{best_lazy['t_rotate']:12.6f} "
        f"{best_lazy['t_rotate'] / max(best_std['t_rotate'], 1e-9):7.2f}x"
    )
    print(
        f"{'Total (s)':30s} "
        f"{best_std['t_total']:12.3f} "
        f"{best_lazy['t_total']:12.3f} "
        f"{best_lazy['t_total'] / best_std['t_total']:7.2f}x"
    )
    print(
        f"{'Memory (GB)':30s} "
        f"{best_std['mem_gb']:12.4f} "
        f"{best_lazy['mem_gb']:12.4f} "
        f"{best_lazy['mem_gb'] / best_std['mem_gb']:7.2f}x"
    )
    print("-" * 65)
    print(f"{'Eigenvalue max|diff|':30s} {eval_diff:.2e}")
    print(f"{'UtW max|diff| (abs)':30s} {utw_diff:.2e}")
    print(f"{'Uty max|diff| (abs)':30s} {uty_diff:.2e}")
    print("=" * 65)

    if best_lazy["t_total"] < best_std["t_total"]:
        pct = (1 - best_lazy["t_total"] / best_std["t_total"]) * 100
        print(f"\nLazy is {pct:.1f}% FASTER overall")
    else:
        pct = (best_lazy["t_total"] / best_std["t_total"] - 1) * 100
        print(
            f"\nLazy is {pct:.1f}% slower overall "
            f"(expected: skips dormtr, pays per-rotate)"
        )

    mem_save = (1 - best_lazy["mem_gb"] / best_std["mem_gb"]) * 100
    print(
        f"Lazy saves {mem_save:.0f}% memory "
        f"({best_std['mem_gb'] - best_lazy['mem_gb']:.4f} GB)"
    )


if __name__ == "__main__":
    main()
