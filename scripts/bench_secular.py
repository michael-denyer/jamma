#!/usr/bin/env python3
"""Benchmark: secular LOCO solver vs DSYEVD baseline at varying n.

Compares secular_eigendecompose_from_full (O(n^2 * r_eff) eigenvalue phase)
against loco_eigendecompose_from_full (O(n^3) direct DSYEVD) on synthetic data
with controlled effective rank.

At small n, DSYEVD often wins — the tridiagonal eigensolver's constant factor
is very low for moderate n. The secular advantage emerges at large n where
r_eff << n, because the secular solver avoids the full tridiagonal solve.
Expected crossover: n >> 5000 with r_eff/n < 0.10 (typical GWAS setting).

Usage:
    uv run python scripts/bench_secular.py
    uv run python scripts/bench_secular.py --sizes 500,1000,2000
    uv run python scripts/bench_secular.py --sizes 1000 --runs 1
    uv run python scripts/bench_secular.py --r-eff 100 --delta-threshold 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src/ to path if running from repo root without install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jamma.lmm.loco_eigen_update import (  # noqa: E402
    loco_eigendecompose_from_full,
    secular_eigendecompose_from_full,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _speedup(ref: float, fast: float) -> str:
    """Format speedup ratio."""
    ratio = ref / fast
    return f"{ratio:.2f}x"


def _orth_deviation(U: np.ndarray) -> float:
    """Compute max|U^T U - I|."""
    return float(np.max(np.abs(U.T @ U - np.eye(U.shape[0]))))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_benchmark_data(
    n: int,
    r_eff: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Generate synthetic kinship + chromosome data for benchmarking.

    Args:
        n: Number of samples.
        r_eff: Effective rank of the chromosome genotype matrix. Controls how
            many rank-1 updates the secular solver must perform. Lower r_eff
            relative to n gives more speedup.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (d_full, U_full, X_c, S_chr, p_full, p_chr) where:
            d_full: (n,) eigenvalues of K_full
            U_full: (n, n) eigenvectors of K_full
            X_c: (n, p_chr) chromosome genotype matrix (secular solver input)
            S_chr: (n, n) Gram matrix X_c @ X_c.T (DSYEVD baseline input)
            p_full: total SNP count
            p_chr: chromosome SNP count
    """
    rng = np.random.default_rng(seed)

    # Full kinship: n x (n*10) genotype matrix, normalised
    p_full = n * 10
    X_all = rng.standard_normal((n, p_full))
    X_all -= X_all.mean(axis=0)
    K_full = (X_all @ X_all.T) / p_full

    d_full, U_full = np.linalg.eigh(K_full)

    # Chromosome X_c: controlled effective rank via latent factor model.
    # X_c = latents @ weights produces a matrix whose column space is
    # exactly rank r_eff (no noise inflation).
    latents = rng.standard_normal((n, r_eff))
    p_chr = r_eff * 10
    weights = rng.standard_normal((r_eff, p_chr))
    X_c = latents @ weights
    X_c -= X_c.mean(axis=0)

    S_chr = X_c @ X_c.T

    return d_full, U_full, X_c, S_chr, p_full, p_chr


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_secular(
    d_full: np.ndarray,
    U_full: np.ndarray,
    X_c: np.ndarray,
    p_full: int,
    p_chr: int,
    runs: int,
    delta_threshold: int | None,
) -> tuple[float, float, np.ndarray]:
    """Time secular_eigendecompose_from_full over `runs` iterations.

    Args:
        d_full: (n,) eigenvalues of K_full.
        U_full: (n, n) eigenvectors of K_full.
        X_c: (n, p_chr) chromosome genotype matrix.
        p_full: Total SNP count.
        p_chr: Chromosome SNP count.
        runs: Number of timing iterations; report minimum.
        delta_threshold: Override n_threshold_for_delta. None uses function
            default (5000).

    Returns:
        Tuple of (min_elapsed, orth_deviation, U_loco).
    """
    kwargs = {}
    if delta_threshold is not None:
        kwargs["n_threshold_for_delta"] = delta_threshold

    best = float("inf")
    U_loco = None
    for _ in range(runs):
        t0 = time.perf_counter()
        _, U_loco = secular_eigendecompose_from_full(
            d_full, U_full, X_c, p_full, p_chr, **kwargs
        )
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)

    orth = _orth_deviation(U_loco)
    return best, orth, U_loco


def bench_dsyevd(
    d_full: np.ndarray,
    U_full: np.ndarray,
    S_chr: np.ndarray,
    p_full: int,
    p_chr: int,
    runs: int,
) -> tuple[float, float, np.ndarray]:
    """Time loco_eigendecompose_from_full (DSYEVD baseline) over `runs` iterations.

    Args:
        d_full: (n,) eigenvalues of K_full.
        U_full: (n, n) eigenvectors of K_full.
        S_chr: (n, n) chromosome Gram matrix X_c @ X_c.T.
        p_full: Total SNP count.
        p_chr: Chromosome SNP count.
        runs: Number of timing iterations; report minimum.

    Returns:
        Tuple of (min_elapsed, orth_deviation, U_loco).
    """
    best = float("inf")
    U_loco = None
    for _ in range(runs):
        t0 = time.perf_counter()
        _, U_loco = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)

    orth = _orth_deviation(U_loco)
    return best, orth, U_loco


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="500,1000,2000",
        help="Comma-separated n values (default: 500,1000,2000)",
    )
    parser.add_argument(
        "--r-eff",
        type=str,
        default="auto",
        help="Fixed r_eff or 'auto' (auto = n//10) (default: auto)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per size, report best (default: 3)",
    )
    parser.add_argument(
        "--delta-threshold",
        type=int,
        default=None,
        dest="delta_threshold",
        help=(
            "Override n_threshold_for_delta. 0 forces delta path; "
            "99999 forces Q path. Default: use function default (5000)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation (default: 42)",
    )
    args = parser.parse_args()

    # Parse sizes
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
    except ValueError:
        parser.error(f"--sizes must be comma-separated integers, got: {args.sizes!r}")

    # Parse r_eff mode
    r_eff_fixed: int | None = None
    if args.r_eff != "auto":
        try:
            r_eff_fixed = int(args.r_eff)
        except ValueError:
            parser.error(f"--r-eff must be 'auto' or an integer, got: {args.r_eff!r}")

    print("Secular vs DSYEVD Benchmark")
    print("=" * 27)
    print(f"Runs per size: {args.runs} (best of)")
    if args.delta_threshold is not None:
        print(f"Delta threshold override: {args.delta_threshold}")
    else:
        print("Delta threshold: function default (5000)")
    print()

    # Column widths for formatting
    col_n = 7
    col_r = 7
    col_path = 7
    col_t = 11
    col_s = 9
    col_o = 12

    header = (
        f"{'n':<{col_n}}"
        f"{'r_eff':<{col_r}}"
        f"{'path':<{col_path}}"
        f"{'secular(s)':<{col_t}}"
        f"{'dsyevd(s)':<{col_t}}"
        f"{'speedup':<{col_s}}"
        f"{'orth_sec':<{col_o}}"
        f"{'orth_dsyevd':<{col_o}}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        r_eff = r_eff_fixed if r_eff_fixed is not None else n // 10
        if r_eff < 1:
            r_eff = 1

        # Determine path label based on threshold
        threshold = args.delta_threshold if args.delta_threshold is not None else 5000
        path_label = "delta" if n > threshold else "Q"

        print(
            f"{'n=' + str(n):<{col_n}}"
            f"{str(r_eff):<{col_r}}"
            f"{path_label:<{col_path}}"
            "  (generating data)...",
            end="\r",
            flush=True,
        )

        d_full, U_full, X_c, S_chr, p_full, p_chr = _make_benchmark_data(
            n=n, r_eff=r_eff, seed=args.seed
        )

        sec_time, sec_orth, _ = bench_secular(
            d_full, U_full, X_c, p_full, p_chr, args.runs, args.delta_threshold
        )
        dsy_time, dsy_orth, _ = bench_dsyevd(
            d_full, U_full, S_chr, p_full, p_chr, args.runs
        )

        speedup_str = _speedup(dsy_time, sec_time)

        row = (
            f"{n:<{col_n}}"
            f"{r_eff:<{col_r}}"
            f"{path_label:<{col_path}}"
            f"{_fmt(sec_time):<{col_t}}"
            f"{_fmt(dsy_time):<{col_t}}"
            f"{speedup_str:<{col_s}}"
            f"{sec_orth:.2e}   "
            f"{dsy_orth:.2e}"
        )
        # Clear the progress line then print result
        print(" " * 80, end="\r")
        print(row)

    print()
    print(
        "Note: At n=83k with r_eff=300 (typical GWAS), secular solver is expected to"
        " outperform DSYEVD based on O(n^2*r_eff) eigenvalue + O(n^3) backward pass"
        " vs O(n^3) full tridiagonal eigensolver with larger constant. At moderate n"
        " (<5000), DSYEVD's highly optimised tridiagonal solver typically wins. The"
        " secular advantage is memory-driven at large n: the delta path avoids"
        " allocating Q = np.eye(n) (55 GB at n=83k), enabling LOCO runs that would"
        " otherwise OOM."
    )
    print()
    print(
        "Speedup interpretation: >1.0x means secular faster; <1.0x means DSYEVD"
        " faster. Orth = max|U^T U - I| (both should be near machine epsilon ~1e-15;"
        " secular may show slightly higher values at large n due to accumulated"
        " floating-point error in sequential rank-1 updates)."
    )


if __name__ == "__main__":
    main()
