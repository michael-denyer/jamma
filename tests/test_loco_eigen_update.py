"""Tests for rotated-basis eigenvalue update for LOCO eigendecomposition.

Validates the correctness of loco_eigendecompose_from_full and measure_effective_rank
against DSYEVD reference eigenvalues, eigenvector orthogonality, Gram rotation
equivalence, and SVD-based effective rank measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jamma.lmm.loco_eigen_update import (
    loco_eigendecompose_from_full,
    measure_effective_rank,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
MOUSE_HS1940_BFILE = _FIXTURE_ROOT / "mouse_hs1940" / "mouse_hs1940"


def _mouse_hs1940_available() -> bool:
    return MOUSE_HS1940_BFILE.with_suffix(".bed").exists()


# ---------------------------------------------------------------------------
# Synthetic test helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n: int = 200, p: int = 500, p_c: int = 100, seed: int = 42):
    """Create a reproducible synthetic GWAS dataset.

    Returns:
        K_full: (n, n) full kinship matrix
        d_full: (n,) eigenvalues of K_full
        U_full: (n, n) eigenvectors of K_full
        S_chr:  (n, n) chromosome contribution X_c @ X_c.T
        K_loco: (n, n) expected LOCO kinship matrix
        p_full: total SNP count
        p_chr:  chromosome SNP count
    """
    rng = np.random.default_rng(seed=seed)
    # Simulate genotype matrix X: (n, p) centered
    X = rng.standard_normal((n, p))
    X -= X.mean(axis=0)

    # Full kinship and chromosome contribution
    K_full = (X @ X.T) / p
    X_c = X[:, :p_c]
    S_chr = X_c @ X_c.T
    K_loco = (X @ X.T - S_chr) / (p - p_c)

    d_full, U_full = np.linalg.eigh(K_full)
    return K_full, d_full, U_full, S_chr, K_loco, p, p_c


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_secular_vs_dsyevd():
    """Rotated-basis eigenvalues match DSYEVD eigenvalues for rank-k downdate."""
    _, d_full, U_full, S_chr, K_loco, p_full, p_chr = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    d_loco, _ = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)
    d_ref = np.linalg.eigh(K_loco)[0]

    np.testing.assert_allclose(
        np.sort(d_loco),
        np.sort(d_ref),
        rtol=1e-10,
        err_msg="Rotated-basis eigenvalues do not match DSYEVD reference",
    )


def test_eigenvector_orthogonality():
    """Eigenvectors from rotated-basis update satisfy U^T U = I within 1e-10."""
    _, d_full, U_full, S_chr, _, p_full, p_chr = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    _, U_loco = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)

    deviation = np.max(np.abs(U_loco.T @ U_loco - np.eye(U_loco.shape[0])))
    assert deviation < 1e-10, (
        f"Eigenvectors not orthogonal: max|U^T U - I| = {deviation:.2e}"
    )


def test_gram_vs_direct():
    """Gram rotation path (S_chr -> M_gram) and direct Z path produce identical M."""
    rng = np.random.default_rng(seed=7)
    n, p, p_c = 150, 400, 80
    X = rng.standard_normal((n, p))
    X -= X.mean(axis=0)
    X_c = X[:, :p_c]

    K_full = (X @ X.T) / p
    _, U_full = np.linalg.eigh(K_full)

    S_chr = X_c @ X_c.T
    alpha_c = p / (p - p_c)
    sigma = 1.0 / (p - p_c)

    # Path A: Gram rotation (S_chr -> M_gram)
    M_gram = U_full.T @ S_chr @ U_full
    M_via_gram = np.diag(alpha_c * np.linalg.eigh(K_full)[0]) - sigma * M_gram

    # Path B: Direct Z path
    Z = U_full.T @ X_c  # (n, p_c)
    Z_gram = Z @ Z.T  # (n, n) — same as M_gram
    d_full = np.linalg.eigh(K_full)[0]
    M_via_z = np.diag(alpha_c * d_full) - sigma * Z_gram

    np.testing.assert_allclose(
        M_via_gram,
        M_via_z,
        atol=1e-12,
        err_msg="Gram rotation path does not match direct Z path",
    )


def test_svd_compression():
    """SVD compression reduces r_eff below 0.5 * p_c for LD-correlated genotype data."""
    rng = np.random.default_rng(seed=99)
    n, p, p_c = 200, 500, 100

    X = rng.standard_normal((n, p))
    X -= X.mean(axis=0)
    K_full = (X @ X.T) / p
    _, U_full = np.linalg.eigh(K_full)

    # Create block-diagonal LD structure: group columns into blocks of 10,
    # replace each block with one latent factor plus noise
    X_c_raw = X[:, :p_c]
    block_size = 10
    n_blocks = p_c // block_size
    X_c_ld = np.zeros_like(X_c_raw)
    for b in range(n_blocks):
        start, end = b * block_size, (b + 1) * block_size
        latent = rng.standard_normal(n)
        # Strong within-block correlation (same latent factor + small noise)
        X_c_ld[:, start:end] = latent[:, None] + 0.05 * rng.standard_normal(
            (n, block_size)
        )
    X_c_ld -= X_c_ld.mean(axis=0)

    r_eff, singular_values = measure_effective_rank(U_full, X_c_ld)

    # Log compression ratio for diagnostic purposes
    compression_ratio = r_eff / p_c
    print(
        f"\n  test_svd_compression: p_c={p_c}, r_eff={r_eff}, "
        f"compression={1 - compression_ratio:.0%}"
    )

    assert r_eff < 0.5 * p_c, (
        f"r_eff={r_eff} is not below 0.5 * p_c={0.5 * p_c}; "
        f"LD compression ineffective (ratio={compression_ratio:.2f})"
    )
    assert len(singular_values) == min(n, p_c)


def test_scaled_eigenvalue_identity():
    """sum(d_loco) approximates trace(K_loco) — validates alpha_c scaling."""
    _, d_full, U_full, S_chr, K_loco, p_full, p_chr = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    d_loco, _ = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)

    trace_ref = np.trace(K_loco)
    trace_loco = np.sum(d_loco)

    np.testing.assert_allclose(
        trace_loco,
        trace_ref,
        rtol=1e-8,
        err_msg=(
            f"sum(d_loco)={trace_loco:.6f} does not match trace(K_loco)={trace_ref:.6f}"
        ),
    )


def test_degenerate_chromosome():
    """When p_chr = 0, d_loco equals alpha_c * d_full = 1.0 * d_full."""
    _, d_full, U_full, _, _, p_full, _ = _make_synthetic_dataset(
        n=100, p=300, p_c=0, seed=13
    )

    # p_chr = 0 means S_chr is the zero matrix and alpha_c = p_full / p_full = 1.0
    n = d_full.shape[0]
    S_chr_zero = np.zeros((n, n))

    d_loco, U_loco = loco_eigendecompose_from_full(
        d_full, U_full, S_chr_zero, p_full, p_chr=0
    )

    np.testing.assert_allclose(
        np.sort(d_loco),
        np.sort(d_full),
        rtol=1e-12,
        err_msg="Degenerate case (p_chr=0) must return d_full unchanged",
    )


@pytest.mark.tier1
@pytest.mark.slow
def test_mouse_hs1940_r_eff():
    """Measure effective rank per chromosome on mouse_hs1940.

    This is a diagnostic test that validates SVD compression is active
    (r_eff < p_c) for at least one chromosome. Logs compression ratio per
    chromosome to inform future secular equation optimization.
    """
    if not _mouse_hs1940_available():
        pytest.skip("mouse_hs1940 fixture not available")

    from bed_reader import open_bed

    from jamma.io.plink import get_chromosome_partitions
    from jamma.kinship.missing import impute_and_center
    from jamma.lmm.eigen import eigendecompose_kinship

    # Load full genotype matrix
    bed = open_bed(str(MOUSE_HS1940_BFILE.with_suffix(".bed")))
    X_full_raw = bed.read(dtype="float32").astype(np.float64)  # (n, p)
    X_full = impute_and_center(X_full_raw)

    n, p_full = X_full.shape

    # Build full kinship and eigendecompose
    K_full = (X_full @ X_full.T) / p_full
    d_full, U_full = eigendecompose_kinship(K_full, check_memory=False)

    # Get per-chromosome partitions
    partitions = get_chromosome_partitions(MOUSE_HS1940_BFILE)

    at_least_one_compressed = False
    results = []
    for chr_id, (start_idx, end_idx) in partitions.items():
        X_c = X_full[:, start_idx:end_idx]
        p_c = X_c.shape[1]

        r_eff, sv = measure_effective_rank(U_full, X_c)
        compression_ratio = r_eff / p_c if p_c > 0 else 1.0
        results.append((chr_id, p_c, r_eff, compression_ratio))

        if r_eff < p_c:
            at_least_one_compressed = True

    # Log results (visible with pytest -s or in CI logs)
    print("\n  mouse_hs1940 r_eff per chromosome:")
    print(f"  {'Chr':>4}  {'p_c':>6}  {'r_eff':>6}  {'ratio':>8}  {'compression':>12}")
    for chr_id, p_c, r_eff, ratio in results:
        print(
            f"  {chr_id!s:>4}  {p_c:>6}  {r_eff:>6}  {ratio:>8.2%}  {1 - ratio:>12.0%}"
        )

    assert at_least_one_compressed, (
        "No chromosome showed r_eff < p_c. "
        "SVD compression ineffective on mouse_hs1940 — secular equation approach "
        "may not be worth implementing."
    )
