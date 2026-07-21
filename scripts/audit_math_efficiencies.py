#!/usr/bin/env python3
"""Check exactness and scale models for mathematical efficiency candidates."""

from __future__ import annotations

import argparse

import numpy as np

from jamma.lmm.likelihood import calc_pab, compute_Uab, get_ab_index


def _block_projection(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    weighted_w = weights[:, None] * w
    a = w.T @ weighted_w
    b = w.T @ (weights * x)
    e = w.T @ (weights * y)
    p_xx = float(x @ (weights * x) - b @ np.linalg.solve(a, b))
    p_xy = float(x @ (weights * y) - b @ np.linalg.solve(a, e))
    p_yy_w = float(y @ (weights * y) - e @ np.linalg.solve(a, e))
    p_yy_wx = p_yy_w - p_xy * p_xy / p_xx
    design = np.column_stack((w, x))
    sign, logdet = np.linalg.slogdet(design.T @ (weights[:, None] * design))
    if sign <= 0:
        raise AssertionError("synthetic weighted Gram matrix is not positive definite")
    return np.array((p_xx, p_xy, p_yy_w, p_yy_wx, logdet))


def _packed_projection(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    n_cvt = w.shape[1]
    pab = calc_pab(n_cvt, weights, compute_Uab(w, y, x))
    idx_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
    idx_xy = get_ab_index(n_cvt + 1, n_cvt + 2, n_cvt)
    idx_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    diagonal = np.array(
        [
            pab[level, get_ab_index(level + 1, level + 1, n_cvt)]
            for level in range(n_cvt + 1)
        ]
    )
    return np.array(
        (
            pab[n_cvt, idx_xx],
            pab[n_cvt, idx_xy],
            pab[n_cvt, idx_yy],
            pab[n_cvt + 1, idx_yy],
            np.log(diagonal).sum(),
        )
    )


def verify_block_projection(rng: np.random.Generator, trials: int) -> float:
    worst = 0.0
    for n_cvt in (1, 2, 4, 8, 16):
        n_samples = max(128, 6 * n_cvt)
        for _ in range(trials):
            w = np.column_stack(
                (np.ones(n_samples), rng.standard_normal((n_samples, n_cvt - 1)))
            )
            x = rng.standard_normal(n_samples)
            y = rng.standard_normal(n_samples)
            eigenvalues = rng.uniform(0.0, 3.0, n_samples)
            lambda_value = 10.0 ** rng.uniform(-5.0, 5.0)
            weights = 1.0 / (1.0 + lambda_value * eigenvalues)
            packed = _packed_projection(w, x, y, weights)
            block = _block_projection(w, x, y, weights)
            scaled_error = np.abs(packed - block) / np.maximum(1.0, np.abs(packed))
            worst = max(worst, float(scaled_error.max()))
            np.testing.assert_allclose(block, packed, rtol=2e-11, atol=2e-11)
    return worst


def verify_low_rank_inverse(rng: np.random.Generator, trials: int) -> float:
    worst = 0.0
    for n_samples, n_markers in ((64, 12), (96, 40), (128, 80)):
        for _ in range(trials):
            genotypes = rng.standard_normal((n_samples, n_markers))
            kinship = genotypes @ genotypes.T / n_markers
            vectors = rng.standard_normal((n_samples, 4))
            lambda_value = 10.0 ** rng.uniform(-5.0, 5.0)
            expected = np.linalg.solve(
                np.eye(n_samples) + lambda_value * kinship, vectors
            )
            u, singular_values, _ = np.linalg.svd(genotypes, full_matrices=False)
            eigenvalues = singular_values * singular_values / n_markers
            correction = (1.0 / (1.0 + lambda_value * eigenvalues)) - 1.0
            actual = vectors + u @ (correction[:, None] * (u.T @ vectors))
            scaled_error = np.abs(expected - actual) / np.maximum(1.0, np.abs(expected))
            worst = max(worst, float(scaled_error.max()))
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    return worst


def _pab_entries(n_cvt: int) -> int:
    return (n_cvt + 1) * (n_cvt + 2) * (n_cvt + 3) // 6


def report_scale(
    n_samples: int, kinship_snps: int, n_grid: int, n_refine: int, threads: int
) -> None:
    print("packed Pab recursion")
    print("covariates entries evaluations entries_x_evaluations scratch_gb")
    for n_cvt in (1, 2, 4, 20, 50, 100):
        entries = _pab_entries(n_cvt)
        evaluations = n_grid + n_refine + 3
        n_index = (n_cvt + 2) * (n_cvt + 3) // 2
        scratch_gb = threads * n_samples * n_index * 8 / 1e9
        print(
            f"{n_cvt:10d} {entries:7d} {evaluations:11d} "
            f"{entries * evaluations:21d} {scratch_gb:10.2f}"
        )

    print("mode-4 coarse-grid reductions")
    print(f"current={2 * n_grid} shared={n_grid} theoretical_saving=50%")

    if kinship_snps < n_samples:
        ratio = kinship_snps / n_samples
        full_gb = n_samples * n_samples * 8 / 1e9
        thin_gb = n_samples * kinship_snps * 8 / 1e9
        print("low-rank spectral representation")
        print(
            f"rank_ratio={ratio:.3%} vectors_full_gb={full_gb:.2f} "
            f"vectors_thin_gb={thin_gb:.2f} decomposition_proxy={ratio * ratio:.3%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=125_632)
    parser.add_argument("--kinship-snps", type=int, default=91_586)
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--refine", type=int, default=20)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    rng = np.random.default_rng(20260721)
    block_error = verify_block_projection(rng, args.trials)
    low_rank_error = verify_low_rank_inverse(rng, args.trials)
    print(f"block_vs_packed=PASS worst_scaled_error={block_error:.3e}")
    print(f"low_rank_inverse=PASS worst_scaled_error={low_rank_error:.3e}")
    report_scale(args.samples, args.kinship_snps, args.grid, args.refine, args.threads)


if __name__ == "__main__":
    main()
