"""Tests for rotated-basis eigenvalue update for LOCO eigendecomposition.

Validates the correctness of loco_eigendecompose_from_full and measure_effective_rank
against DSYEVD reference eigenvalues, eigenvector orthogonality, Gram rotation
equivalence, and SVD-based effective rank measurement.

Also validates secular_eigendecompose_from_full: the O(n^2 * r_eff) secular equation
solver that replaces np.linalg.eigh(M) in loco_eigendecompose_from_full.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jamma.lmm.loco_eigen_update import (
    _rank1_update_python,
    loco_eigendecompose_from_full,
    measure_effective_rank,
    secular_eigendecompose_from_full,
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
        X_c:    (n, p_c) centered chromosome genotype matrix
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
    return K_full, d_full, U_full, S_chr, K_loco, p, p_c, X_c


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_secular_vs_dsyevd():
    """Rotated-basis eigenvalues match DSYEVD eigenvalues for rank-k downdate.

    Both the loco_eigendecompose_from_full result and the DSYEVD reference
    have the GEMMA threshold (1e-10) applied before comparison.
    """
    _, d_full, U_full, S_chr, K_loco, p_full, p_chr, _ = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    d_loco, _ = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)
    d_ref = np.linalg.eigh(K_loco)[0]

    # Apply the same GEMMA threshold to the reference eigenvalues for comparison
    d_ref_thresh = d_ref.copy()
    d_ref_thresh[np.abs(d_ref_thresh) < 1e-10] = 0.0

    np.testing.assert_allclose(
        np.sort(d_loco),
        np.sort(d_ref_thresh),
        rtol=1e-10,
        err_msg="Rotated-basis eigenvalues do not match DSYEVD reference",
    )


def test_eigenvector_orthogonality():
    """Eigenvectors from rotated-basis update satisfy U^T U = I within 1e-10."""
    _, d_full, U_full, S_chr, _, p_full, p_chr, _ = _make_synthetic_dataset(
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
    """measure_effective_rank correctly reports r_eff for structured low-rank data.

    Constructs X_c as exactly r_true rank-1 latent factors (no noise), so SVD
    should return exactly r_true significant singular values regardless of
    rotation by U_full (orthogonal rotation preserves singular values).
    """
    rng = np.random.default_rng(seed=99)
    n, p, p_c = 200, 500, 100
    r_true = 12  # number of independent latent factors (= true rank of X_c)

    X = rng.standard_normal((n, p))
    X -= X.mean(axis=0)
    K_full = (X @ X.T) / p
    _, U_full = np.linalg.eigh(K_full)

    # Build exactly rank-r_true X_c: r_true latent factors, each copied to
    # ceil(p_c / r_true) columns. No noise — purely rank-r_true matrix.
    cols_per_factor = p_c // r_true  # 8 columns per factor, r_true=12 -> 96 cols
    latents = rng.standard_normal((n, r_true))
    X_c_ld = np.zeros((n, p_c))
    for j in range(r_true):
        start = j * cols_per_factor
        end = min(start + cols_per_factor, p_c)
        X_c_ld[:, start:end] = latents[:, j : j + 1]  # broadcast factor
    # Fill any remaining columns with the last factor
    last_end = r_true * cols_per_factor
    if last_end < p_c:
        X_c_ld[:, last_end:] = latents[:, -1:]
    X_c_ld -= X_c_ld.mean(axis=0)

    r_eff, singular_values = measure_effective_rank(U_full, X_c_ld)

    # Log compression ratio for diagnostic purposes
    compression_ratio = r_eff / p_c
    print(
        f"\n  test_svd_compression: p_c={p_c}, r_true={r_true}, r_eff={r_eff}, "
        f"compression={1 - compression_ratio:.0%}"
    )

    # r_eff should equal r_true (exactly r_true non-zero singular values)
    assert r_eff == r_true, (
        f"r_eff={r_eff} != r_true={r_true}; "
        f"measure_effective_rank returned wrong effective rank"
    )
    assert r_eff < 0.5 * p_c, (
        f"r_eff={r_eff} is not below 0.5 * p_c={0.5 * p_c}; "
        f"compression check failed (ratio={compression_ratio:.2f})"
    )
    assert len(singular_values) == min(n, p_c)


def test_scaled_eigenvalue_identity():
    """sum(d_loco) approximates trace(K_loco) — validates alpha_c scaling."""
    _, d_full, U_full, S_chr, K_loco, p_full, p_chr, _ = _make_synthetic_dataset(
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
    """When p_chr = 0, d_loco equals d_full after applying the eigenvalue threshold.

    Alpha_c = p_full / (p_full - 0) = 1.0, sigma * M_gram = 0 => M = diag(d_full).
    The implementation returns d_full with the standard GEMMA threshold applied.
    """
    _, d_full, U_full, _, _, p_full, _, _ = _make_synthetic_dataset(
        n=100, p=300, p_c=0, seed=13
    )

    # p_chr = 0 means S_chr is the zero matrix and alpha_c = p_full / p_full = 1.0
    n = d_full.shape[0]
    S_chr_zero = np.zeros((n, n))

    d_loco, U_loco = loco_eigendecompose_from_full(
        d_full, U_full, S_chr_zero, p_full, p_chr=0
    )

    # Apply same threshold to d_full for comparison (GEMMA-compatible behaviour)
    d_full_thresholded = d_full.copy()
    d_full_thresholded[np.abs(d_full_thresholded) < 1e-10] = 0.0

    np.testing.assert_allclose(
        np.sort(d_loco),
        np.sort(d_full_thresholded),
        rtol=1e-12,
        err_msg="Degenerate case (p_chr=0) must return thresholded d_full",
    )


# ---------------------------------------------------------------------------
# Secular equation solver tests (Plan 02)
# ---------------------------------------------------------------------------


def test_secular_solver_vs_gram_rotation():
    """secular_eigendecompose_from_full eigenvalues match loco_eigendecompose_from_full.

    Test 1: Secular solver vs O(n^3) rotated-basis update. Both should produce
    eigenvalues within rtol=1e-8 on synthetic n=200, p=500, p_c=100 data.
    The rotated-basis update (loco_eigendecompose_from_full) is the reference —
    it is validated against DSYEVD in test_secular_vs_dsyevd above.
    """
    _, d_full, U_full, S_chr, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    d_ref, _ = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)
    d_secular, _ = secular_eigendecompose_from_full(d_full, U_full, X_c, p_full, p_chr)

    np.testing.assert_allclose(
        np.sort(d_secular),
        np.sort(d_ref),
        rtol=1e-8,
        err_msg="secular_eigendecompose_from_full eigenvalues do not match "
        "loco_eigendecompose_from_full (O(n^3) reference)",
    )


def test_secular_solver_eigenvector_orthogonality():
    """Eigenvectors from secular solver satisfy max|U^T U - I| < 1e-8.

    Test 2: Eigenvectors must be orthonormal within numerical tolerance.
    """
    _, d_full, U_full, _, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    _, U_secular = secular_eigendecompose_from_full(d_full, U_full, X_c, p_full, p_chr)

    deviation = np.max(np.abs(U_secular.T @ U_secular - np.eye(U_secular.shape[0])))
    assert deviation < 1e-8, (
        f"Secular solver eigenvectors not orthogonal: max|U^T U - I| = {deviation:.2e}"
    )


def test_secular_solver_r_eff_compression():
    """Secular solver uses measure_effective_rank; r_eff < p_c for LD-structured data.

    Test 3: Low-rank X_c structure should give r_eff << p_c.
    Uses the same exactly-rank-r_true data as test_svd_compression.
    """
    rng = np.random.default_rng(seed=99)
    n, p, p_c = 200, 500, 100
    r_true = 12

    X = rng.standard_normal((n, p))
    X -= X.mean(axis=0)
    K_full = (X @ X.T) / p
    d_full, U_full = np.linalg.eigh(K_full)

    # Build exactly rank-r_true X_c
    cols_per_factor = p_c // r_true
    latents = rng.standard_normal((n, r_true))
    X_c_ld = np.zeros((n, p_c))
    for j in range(r_true):
        start = j * cols_per_factor
        end = min(start + cols_per_factor, p_c)
        X_c_ld[:, start:end] = latents[:, j : j + 1]
    last_end = r_true * cols_per_factor
    if last_end < p_c:
        X_c_ld[:, last_end:] = latents[:, -1:]
    X_c_ld -= X_c_ld.mean(axis=0)

    # Run secular solver — internally calls measure_effective_rank
    d_secular, U_secular = secular_eigendecompose_from_full(
        d_full, U_full, X_c_ld, p, p_c
    )

    # Verify r_eff was correctly determined: compare to S_chr reference
    S_chr_ld = X_c_ld @ X_c_ld.T
    d_ref, _ = loco_eigendecompose_from_full(d_full, U_full, S_chr_ld, p, p_c)
    np.testing.assert_allclose(
        np.sort(d_secular),
        np.sort(d_ref),
        rtol=1e-8,
        err_msg="Secular solver with LD-structured X_c does not match reference",
    )

    # Also verify r_eff compression: measure_effective_rank should find r_eff == r_true
    r_eff, _ = measure_effective_rank(U_full, X_c_ld)
    assert r_eff == r_true, (
        f"r_eff={r_eff} != r_true={r_true} for exactly-rank-{r_true} X_c"
    )
    assert r_eff < p_c, f"r_eff={r_eff} not less than p_c={p_c} (no compression)"


def test_secular_solver_trace_identity():
    """sum(d_secular) matches trace(K_loco) within rtol=1e-6.

    Test 4: Trace identity — sum of eigenvalues must equal trace of K_loco.
    """
    _, d_full, U_full, _, K_loco, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    d_secular, _ = secular_eigendecompose_from_full(d_full, U_full, X_c, p_full, p_chr)

    trace_ref = np.trace(K_loco)
    trace_secular = np.sum(d_secular)

    np.testing.assert_allclose(
        trace_secular,
        trace_ref,
        rtol=1e-6,
        err_msg=(
            f"sum(d_secular)={trace_secular:.6f} does not match "
            f"trace(K_loco)={trace_ref:.6f}"
        ),
    )


def test_secular_solver_python_fallback():
    """When C extension unavailable, Python fallback produces correct results.

    Test 5: Patches _SECULAR_ACCEL_AVAILABLE to False so
    secular_eigendecompose_from_full uses _rank1_update_python (pure Python,
    O(n^3) via eigh). Results must match the C extension path within rtol=1e-8.
    """
    import jamma.lmm.loco_eigen_update as loco_mod

    _, d_full, U_full, S_chr, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=50, p=200, p_c=40, seed=7
    )

    # Reference: C extension path (or standard if C not available)
    d_ref, U_ref = secular_eigendecompose_from_full(d_full, U_full, X_c, p_full, p_chr)

    # Fallback path: patch _SECULAR_ACCEL_AVAILABLE to False
    with patch.object(loco_mod, "_SECULAR_ACCEL_AVAILABLE", False):
        d_fallback, U_fallback = secular_eigendecompose_from_full(
            d_full, U_full, X_c, p_full, p_chr
        )

    np.testing.assert_allclose(
        np.sort(d_fallback),
        np.sort(d_ref),
        rtol=1e-8,
        err_msg="Python fallback eigenvalues do not match C extension path",
    )

    # Also verify fallback eigenvectors are orthogonal
    deviation = np.max(np.abs(U_fallback.T @ U_fallback - np.eye(U_fallback.shape[0])))
    assert deviation < 1e-8, (
        f"Python fallback eigenvectors not orthogonal: max|U^T U - I| = {deviation:.2e}"
    )


def test_secular_solver_degenerate_p_chr_zero():
    """When p_chr=0, secular solver returns d_full unchanged (same as gram rotation).

    Test 6: Degenerate case — X_c is empty (or zero columns). The solver must
    return d_full with threshold applied, matching loco_eigendecompose_from_full.
    """
    _, d_full, U_full, _, _, p_full, _, _ = _make_synthetic_dataset(
        n=100, p=300, p_c=0, seed=13
    )

    n = d_full.shape[0]
    X_c_zero = np.zeros((n, 0))  # zero columns

    d_secular, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c_zero, p_full, p_chr=0
    )

    # Apply same threshold to d_full for comparison (GEMMA-compatible behaviour)
    d_full_thresholded = d_full.copy()
    d_full_thresholded[np.abs(d_full_thresholded) < 1e-10] = 0.0

    np.testing.assert_allclose(
        np.sort(d_secular),
        np.sort(d_full_thresholded),
        rtol=1e-12,
        err_msg=(
            "Degenerate case (p_chr=0) secular solver must return thresholded d_full"
        ),
    )


def test_yield_x_c_k_full_correctness():
    """K_full from X_c accumulation equals K_full from standard S_full path.

    Test 7: K_full = sum(S_chr) / p_full = sum(X_c @ X_c.T) / p_full.
    Verifies the fundamental identity used by the yield_x_c path in loco.py.
    This test validates the formula independent of loco.py — a unit test of
    the mathematical identity, not the streaming implementation.
    """
    rng = np.random.default_rng(seed=55)
    n, p, n_chrs = 100, 400, 3
    p_per_chr = p // n_chrs

    X = rng.standard_normal((n, p))
    X -= X.mean(axis=0)

    # Standard path: K_full = (X @ X.T) / p
    K_full_std = (X @ X.T) / p

    # X_c accumulation path: sum(X_c_i @ X_c_i.T) / p
    K_full_xc = np.zeros((n, n))
    for i in range(n_chrs):
        start = i * p_per_chr
        end = start + p_per_chr if i < n_chrs - 1 else p
        X_c_i = X[:, start:end]
        K_full_xc += X_c_i @ X_c_i.T
    K_full_xc /= p

    np.testing.assert_allclose(
        K_full_xc,
        K_full_std,
        rtol=1e-12,
        atol=1e-14,
        err_msg="K_full from X_c accumulation does not match K_full from standard path",
    )


def test_rank1_update_python_correctness():
    """_rank1_update_python fallback matches np.linalg.eigh for D + rho*z*z^T.

    Validates the pure Python fallback used when C extension is unavailable.
    Tests both positive and negative rho (LOCO downdate case).
    """
    rng = np.random.default_rng(seed=42)
    n = 8
    d = np.sort(rng.uniform(0.1, 3.0, n))
    z = rng.standard_normal(n)

    for rho in [0.5, -0.3, 2.0, -1.5]:
        M = np.diag(d) + rho * np.outer(z, z)
        d_ref, V_ref = np.linalg.eigh(M)

        d_fallback, V_fallback = _rank1_update_python(d, rho, z)

        np.testing.assert_allclose(
            d_fallback,
            d_ref,
            rtol=1e-12,
            err_msg=f"_rank1_update_python eigenvalues wrong for rho={rho}",
        )

        # Eigenvectors: check V @ D @ V^T == M (allow sign flips per column)
        M_reconstructed = V_fallback @ np.diag(d_fallback) @ V_fallback.T
        np.testing.assert_allclose(
            M_reconstructed,
            M,
            atol=1e-10,
            err_msg=f"_rank1_update_python V @ D @ V^T != M for rho={rho}",
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
    for chr_id, snp_indices in partitions.items():
        X_c = X_full[:, snp_indices]
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
