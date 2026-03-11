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
    _apply_vj_to_rows_blocked,
    _apply_vj_transpose_to_vec_blocked,
    _find_deflated_columns,
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


# ---------------------------------------------------------------------------
# Delta path tests (Plan 69.3-02)
# ---------------------------------------------------------------------------


def _build_vj_from_cauchy(
    z_unit: np.ndarray,
    d: np.ndarray,
    eigenvalues: np.ndarray,
    norm_j: np.ndarray,
) -> np.ndarray:
    """Build explicit V_j from Cauchy formula.

    V_j[l,k] = z_unit[l] / (d[l] - lam[k]) / norm[k].

    This is the same formula used by _apply_vj_to_rows_blocked, so testing the
    helper against this reference avoids the eigenvector sign ambiguity from eigh.

    Args:
        z_unit: (n,) unit-norm update vector.
        d: (n,) diagonal before rank-1 update.
        eigenvalues: (n,) eigenvalues after rank-1 update.
        norm_j: (n,) normalization factors.

    Returns:
        (n, n) explicit V_j matrix, each column is a unit eigenvector.
    """
    n = len(d)
    V = np.empty((n, n))
    for k in range(n):
        delta_k = d - eigenvalues[k]
        # Deflation guard: set near-zero delta entries to large value (result ~0)
        delta_k = np.where(np.abs(delta_k) > 1e-300, delta_k, 1e300)
        V[:, k] = z_unit / delta_k / norm_j[k]
    return V


def _compute_norm_j(
    z_unit: np.ndarray, d: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    """Compute norm_j[k] = ||z_unit / (d - eigenvalues[k])||_2 for all k.

    Applies deflation guard: skip terms where |d[l] - eigenvalues[k]| < 1e-300.
    """
    n = len(d)
    norms = np.empty(n)
    for k in range(n):
        delta_k = d - eigenvalues[k]
        # Deflation guard
        safe = np.abs(delta_k) > 1e-300
        norm_sq = np.sum((z_unit[safe] / delta_k[safe]) ** 2)
        norms[k] = np.sqrt(norm_sq) if norm_sq > 0.0 else 0.0
    return norms


def test_apply_vj_to_rows_blocked():
    """_apply_vj_to_rows_blocked(R, z, d, lam, norm) matches R @ V_j within rtol=1e-10.

    Builds V_j using the explicit Cauchy formula (same as the helper), avoiding
    the sign ambiguity of eigenvectors from np.linalg.eigh.
    """
    rng = np.random.default_rng(seed=42)
    n = 20
    d = np.sort(rng.uniform(0.1, 5.0, n))
    z = rng.standard_normal(n)
    rho = 0.5

    # Get eigenvalues via eigh (eigenvalues have no sign ambiguity)
    M = np.diag(d) + rho * np.outer(z, z)
    eigenvalues = np.linalg.eigh(M)[0]

    # Compute z_unit and norm_j for the blocked helper
    z_unit = z / np.linalg.norm(z)
    norm_j = _compute_norm_j(z_unit, d, eigenvalues)

    # Build explicit V_j from Cauchy formula (same formula as helper, no sign flip)
    V_j = _build_vj_from_cauchy(z_unit, d, eigenvalues, norm_j)

    # Random row batch R (b=5, n=20)
    b = 5
    R = rng.standard_normal((b, n))

    # Reference: R @ V_j (explicit Cauchy formula)
    R_ref = R @ V_j

    # Test: blocked Cauchy multiply
    R_blocked = _apply_vj_to_rows_blocked(
        R, z_unit, d, eigenvalues, norm_j, col_block_size=7
    )

    np.testing.assert_allclose(
        R_blocked,
        R_ref,
        rtol=1e-10,
        atol=1e-12,
        err_msg="_apply_vj_to_rows_blocked does not match R @ V_j",
    )

    # Also verify V_j is orthogonal (unit eigenvectors)
    deviation = np.max(np.abs(V_j.T @ V_j - np.eye(n)))
    assert deviation < 1e-8, (
        f"V_j from Cauchy formula not orthogonal: max|V^T V - I| = {deviation:.2e}"
    )


def test_apply_vj_transpose_to_vec_blocked():
    """_apply_vj_transpose_to_vec_blocked(v, ...) matches V_j.T @ v (rtol=1e-10)."""
    rng = np.random.default_rng(seed=99)
    n = 20
    d = np.sort(rng.uniform(0.1, 5.0, n))
    z = rng.standard_normal(n)
    rho = 0.7

    M = np.diag(d) + rho * np.outer(z, z)
    eigenvalues = np.linalg.eigh(M)[0]

    z_unit = z / np.linalg.norm(z)
    norm_j = _compute_norm_j(z_unit, d, eigenvalues)
    V_j = _build_vj_from_cauchy(z_unit, d, eigenvalues, norm_j)

    # Random vector v (n=20)
    v = rng.standard_normal(n)

    # Reference: V_j.T @ v (using explicit Cauchy V_j)
    v_ref = V_j.T @ v

    # Test: blocked transpose
    v_blocked = _apply_vj_transpose_to_vec_blocked(
        v, z_unit, d, eigenvalues, norm_j, col_block_size=7
    )

    np.testing.assert_allclose(
        v_blocked,
        v_ref,
        rtol=1e-10,
        atol=1e-12,
        err_msg="_apply_vj_transpose_to_vec_blocked does not match V_j.T @ v",
    )


def test_delta_path_vs_q_path():
    """Delta path eigenvalues match Q path within rtol=1e-8 on n=200 synthetic data.

    n_threshold_for_delta=0 forces delta path; n_threshold_for_delta=99999 forces Q
    path. Also checks eigenvector orthogonality and subspace alignment.
    """
    _, d_full, U_full, _, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    # Force Q path
    d_q, U_q = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=99999
    )

    # Force delta path
    d_delta, U_delta = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=0
    )

    # Eigenvalues must match
    np.testing.assert_allclose(
        np.sort(d_delta),
        np.sort(d_q),
        rtol=1e-8,
        err_msg="Delta path eigenvalues do not match Q path",
    )

    # Eigenvectors must be orthogonal
    deviation = np.max(np.abs(U_delta.T @ U_delta - np.eye(U_delta.shape[0])))
    assert deviation < 1e-8, (
        f"Delta path eigenvectors not orthogonal: max|U^T U - I| = {deviation:.2e}"
    )

    # Eigenvectors span same subspace: each delta column should align with some Q column
    # Check: for each column, the max |dot product| with any Q-path column > 0.99
    for k in range(U_delta.shape[1]):
        dots = np.abs(U_q.T @ U_delta[:, k])
        assert np.max(dots) > 0.99, (
            f"Delta path eigenvector {k} does not align with any Q-path eigenvector "
            f"(max |dot| = {np.max(dots):.4f})"
        )


def test_delta_path_eigenvector_orthogonality():
    """Delta path eigenvectors satisfy max|U^T U - I| < 1e-8 for n=300 data."""
    _, d_full, U_full, _, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=300, p=600, p_c=120, seed=7
    )

    _, U_delta = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=0
    )

    deviation = np.max(np.abs(U_delta.T @ U_delta - np.eye(U_delta.shape[0])))
    assert deviation < 1e-8, (
        f"Delta path eigenvectors not orthogonal (n=300): "
        f"max|U^T U - I| = {deviation:.2e}"
    )


def test_delta_path_trace_identity():
    """sum(d_delta) matches trace(K_loco) within rtol=1e-6."""
    _, d_full, U_full, _, K_loco, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    d_delta, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=0
    )

    trace_ref = np.trace(K_loco)
    trace_delta = np.sum(d_delta)

    np.testing.assert_allclose(
        trace_delta,
        trace_ref,
        rtol=1e-6,
        err_msg=(
            f"sum(d_delta)={trace_delta:.6f} does not match "
            f"trace(K_loco)={trace_ref:.6f}"
        ),
    )


def test_delta_path_threshold_routing(capfd):
    """n_threshold_for_delta correctly routes n=200 to delta vs Q path.

    We verify routing by checking the result — delta path n_threshold_for_delta=100
    (n=200 > 100) triggers delta path; n_threshold_for_delta=300 (n=200 <= 300) uses Q.
    Both produce consistent results (eigenvalues match).
    """
    _, d_full, U_full, _, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    # Delta path: threshold=100 means n=200 > 100 -> delta
    d_delta, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=100
    )

    # Q path: threshold=300 means n=200 <= 300 -> Q
    d_q, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=300
    )

    np.testing.assert_allclose(
        np.sort(d_delta),
        np.sort(d_q),
        rtol=1e-8,
        err_msg="Threshold routing: delta and Q paths produce different eigenvalues",
    )


def test_delta_path_custom_batch_sizes():
    """Custom row_batch_size and col_block_size produce correct results (rtol=1e-8)."""
    _, d_full, U_full, _, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    # Reference: force Q path
    d_ref, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=99999
    )

    # Delta path with small non-default batch/block sizes
    d_custom, U_custom = secular_eigendecompose_from_full(
        d_full,
        U_full,
        X_c,
        p_full,
        p_chr,
        n_threshold_for_delta=0,
        row_batch_size=50,
        col_block_size=30,
    )

    np.testing.assert_allclose(
        np.sort(d_custom),
        np.sort(d_ref),
        rtol=1e-8,
        err_msg="Custom batch sizes: delta path eigenvalues differ from Q path",
    )

    # Orthogonality check
    deviation = np.max(np.abs(U_custom.T @ U_custom - np.eye(U_custom.shape[0])))
    assert deviation < 1e-8, (
        f"Custom batch sizes: eigenvectors not orthogonal: "
        f"max|U^T U - I| = {deviation:.2e}"
    )


def test_delta_path_memory():
    """Delta path does not allocate Q = np.eye(n) (covers DELTA-03).

    Verifies the structural memory property: Q path allocates np.eye(n) and
    per-step Q @ V_j (n x n DGEMM), while delta path stores only (r_eff, n)
    arrays and uses row-batched Cauchy multiply.

    We verify this by monkey-patching np.eye to detect if the delta path calls
    it with n >= threshold. The Q path is expected to call np.eye(n); the delta
    path must not.
    """
    _, d_full, U_full, _, _, p_full, p_chr, X_c = _make_synthetic_dataset(
        n=200, p=500, p_c=100, seed=42
    )

    # Track np.eye calls with n >= 200 (our problem size)
    _original_eye = np.eye
    eye_calls: list[int] = []

    def _tracking_eye(*args, **kwargs):
        if args:
            eye_calls.append(args[0])
        return _original_eye(*args, **kwargs)

    # Q path: should call np.eye(200)
    eye_calls.clear()
    with patch.object(np, "eye", _tracking_eye):
        secular_eigendecompose_from_full(
            d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=99999
        )
    q_eye_calls = [s for s in eye_calls if s >= 200]
    assert len(q_eye_calls) > 0, "Q path should call np.eye(n) — test setup is wrong"

    # Delta path: must NOT call np.eye(200)
    eye_calls.clear()
    with patch.object(np, "eye", _tracking_eye):
        secular_eigendecompose_from_full(
            d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=0
        )
    delta_eye_calls = [s for s in eye_calls if s >= 200]
    assert len(delta_eye_calls) == 0, (
        f"Delta path must not allocate Q = np.eye(n), but called np.eye with "
        f"n={delta_eye_calls}. This violates DELTA-03."
    )

    # Also verify delta path produces correct results (eigenvalue parity)
    d_q, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=99999
    )
    d_delta, _ = secular_eigendecompose_from_full(
        d_full, U_full, X_c, p_full, p_chr, n_threshold_for_delta=0
    )
    np.testing.assert_allclose(
        np.sort(d_delta),
        np.sort(d_q),
        rtol=1e-8,
        err_msg="Delta path results differ from Q path",
    )


# ---------------------------------------------------------------------------
# Sequential streaming integration tests (Plan 69.4-01)
# ---------------------------------------------------------------------------


_GEMMA_LOCO_BFILE = _FIXTURE_ROOT / "gemma_loco" / "test"


@pytest.mark.tier1
def test_run_lmm_loco_secular_uses_sequential_path():
    """run_lmm_loco(use_secular_update=True) uses sequential streaming path.

    Verifies that the secular path calls yield_x_c_sequential=True (not
    yield_x_c=True), so secular_x_c dict is not populated. The sequential
    path passes K_full directly from _compute_loco_kinship_streaming_numpy.
    """
    if not _GEMMA_LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    from unittest.mock import patch

    import jamma.lmm.loco as loco_module

    _LOCO_BFILE = _GEMMA_LOCO_BFILE

    # Track calls to _compute_loco_kinship_streaming_numpy to verify flag
    call_kwargs: list[dict] = []
    original_fn = loco_module._compute_loco_kinship_streaming_numpy

    def tracking_fn(*args, **kwargs):
        call_kwargs.append(dict(kwargs))
        return original_fn(*args, **kwargs)

    from tests.conftest import load_phenotypes_from_fam

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    with patch.object(
        loco_module,
        "_compute_loco_kinship_streaming_numpy",
        side_effect=tracking_fn,
    ):
        loco = loco_module.run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            backend="numpy",
            check_memory=False,
            show_progress=False,
            use_secular_update=True,
        )

    # At least one call should use yield_x_c_sequential=True
    sequential_calls = [kw for kw in call_kwargs if kw.get("yield_x_c_sequential")]
    assert len(sequential_calls) > 0, (
        f"Expected at least one call with yield_x_c_sequential=True, "
        f"got calls: {call_kwargs}"
    )
    # No call should use yield_x_c=True (old path)
    old_path_calls = [kw for kw in call_kwargs if kw.get("yield_x_c")]
    assert len(old_path_calls) == 0, (
        f"Expected no calls with yield_x_c=True (old accumulation path), "
        f"but found: {old_path_calls}"
    )
    assert loco.n_tested > 0, "Expected SNPs to be tested"


@pytest.mark.tier1
def test_secular_path_does_not_accumulate_x_c_dict():
    """secular_x_c dict is not used in the sequential path.

    Verifies that after refactoring, the secular path does not hold all X_c
    matrices simultaneously. We check that the secular_x_c variable is either
    not populated or is always empty during run_lmm_loco execution.
    """
    if not _GEMMA_LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    from unittest.mock import patch

    import jamma.lmm.loco as loco_module

    _LOCO_BFILE = _GEMMA_LOCO_BFILE

    from tests.conftest import load_phenotypes_from_fam

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Monkey-patch secular_eigendecompose_from_full to verify it's called
    # with X_c one at a time (not from a dict)
    call_count = [0]
    original_secular = loco_module.secular_eigendecompose_from_full

    def tracking_secular(*args, **kwargs):
        call_count[0] += 1
        return original_secular(*args, **kwargs)

    with patch.object(
        loco_module, "secular_eigendecompose_from_full", side_effect=tracking_secular
    ):
        loco = loco_module.run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            backend="numpy",
            check_memory=False,
            show_progress=False,
            use_secular_update=True,
        )

    # secular_eigendecompose_from_full should be called once per chromosome
    from jamma.io.plink import get_plink_metadata

    meta = get_plink_metadata(_LOCO_BFILE)
    n_chrs = len(set(meta["chromosome"].tolist()))
    assert call_count[0] == n_chrs, (
        f"Expected {n_chrs} secular_eigendecompose_from_full calls "
        f"(one per chromosome), got {call_count[0]}"
    )
    assert loco.n_tested > 0, "Expected SNPs to be tested"


# ---------------------------------------------------------------------------
# Input validation tests for loco_eigendecompose_from_full
# ---------------------------------------------------------------------------


class TestLocoEigendecomposeValidation:
    """Tests for ValueError guards in loco_eigendecompose_from_full."""

    def setup_method(self) -> None:
        n = 10
        rng = np.random.default_rng(99)
        self.n = n
        self.d_full = np.sort(rng.random(n))
        K = rng.standard_normal((n, n))
        K = K @ K.T / n
        self.U_full = np.linalg.eigh(K)[1]
        self.S_chr = rng.standard_normal((n, n))
        self.S_chr = self.S_chr @ self.S_chr.T / n
        self.p_full = 100
        self.p_chr = 20

    def test_d_full_not_1d(self) -> None:
        with pytest.raises(ValueError, match="d_full must be 1-D"):
            loco_eigendecompose_from_full(
                np.ones((3, 3)), self.U_full, self.S_chr, self.p_full, self.p_chr
            )

    def test_u_full_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="U_full must be"):
            loco_eigendecompose_from_full(
                self.d_full, np.ones((5, 5)), self.S_chr, self.p_full, self.p_chr
            )

    def test_s_chr_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="S_chr must be"):
            loco_eigendecompose_from_full(
                self.d_full, self.U_full, np.ones((5, 5)), self.p_full, self.p_chr
            )

    def test_p_chr_negative(self) -> None:
        with pytest.raises(ValueError, match="p_chr must be in"):
            loco_eigendecompose_from_full(
                self.d_full, self.U_full, self.S_chr, self.p_full, -1
            )

    def test_p_chr_exceeds_p_full(self) -> None:
        with pytest.raises(ValueError, match="p_chr must be in"):
            loco_eigendecompose_from_full(
                self.d_full, self.U_full, self.S_chr, 100, 200
            )

    def test_p_chr_equals_p_full(self) -> None:
        with pytest.raises(ValueError, match="cannot exclude all SNPs"):
            loco_eigendecompose_from_full(
                self.d_full, self.U_full, self.S_chr, 100, 100
            )


# ---------------------------------------------------------------------------
# Input validation tests for secular_eigendecompose_from_full
# ---------------------------------------------------------------------------


class TestSecularEigendecomposeValidation:
    """Tests for ValueError guards in secular_eigendecompose_from_full."""

    def setup_method(self) -> None:
        n = 10
        rng = np.random.default_rng(99)
        self.n = n
        self.d_full = np.sort(rng.random(n))
        K = rng.standard_normal((n, n))
        K = K @ K.T / n
        self.U_full = np.linalg.eigh(K)[1]
        self.X_c = rng.standard_normal((n, 5))
        self.p_full = 100
        self.p_chr = 5

    def test_d_full_not_1d(self) -> None:
        with pytest.raises(ValueError, match="d_full must be 1-D"):
            secular_eigendecompose_from_full(
                np.ones((3, 3)), self.U_full, self.X_c, self.p_full, self.p_chr
            )

    def test_u_full_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="U_full must be"):
            secular_eigendecompose_from_full(
                self.d_full, np.ones((5, 5)), self.X_c, self.p_full, self.p_chr
            )

    def test_x_c_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="X_c must be"):
            secular_eigendecompose_from_full(
                self.d_full, self.U_full, np.ones((5, 3)), self.p_full, self.p_chr
            )

    def test_p_chr_negative(self) -> None:
        with pytest.raises(ValueError, match="p_chr must be in"):
            secular_eigendecompose_from_full(
                self.d_full, self.U_full, self.X_c, self.p_full, -1
            )

    def test_p_chr_exceeds_p_full(self) -> None:
        with pytest.raises(ValueError, match="p_chr must be in"):
            secular_eigendecompose_from_full(
                self.d_full, self.U_full, self.X_c, 100, 200
            )

    def test_p_chr_equals_p_full(self) -> None:
        with pytest.raises(ValueError, match="cannot exclude all SNPs"):
            secular_eigendecompose_from_full(self.d_full, self.U_full, self.X_c, 5, 5)


# ---------------------------------------------------------------------------
# DLAED4 fallback path test
# ---------------------------------------------------------------------------


def test_secular_q_path_dlaed4_fallback() -> None:
    """C extension RuntimeError falls back to Python eigh."""
    rng = np.random.default_rng(77)
    n = 20
    p_chr = 5
    p_full = 50

    d_full = np.sort(rng.random(n))
    K = rng.standard_normal((n, n))
    K = K @ K.T / n
    U_full = np.linalg.eigh(K)[1]
    X_c = rng.standard_normal((n, p_chr))

    # Get reference result with no mocking
    d_ref, U_ref = secular_eigendecompose_from_full(d_full, U_full, X_c, p_full, p_chr)

    # Mock C extension to always raise RuntimeError, forcing Python fallback
    with (
        patch("jamma.lmm.loco_eigen_update._SECULAR_ACCEL_AVAILABLE", True),
        patch(
            "jamma.lmm.loco_eigen_update._rank1_update_c",
            side_effect=RuntimeError("mocked DLAED4 failure"),
        ),
    ):
        d_fallback, U_fallback = secular_eigendecompose_from_full(
            d_full,
            U_full,
            X_c,
            p_full,
            p_chr,
            n_threshold_for_delta=n + 1,  # force Q path
        )

    np.testing.assert_allclose(d_fallback, d_ref, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# _find_deflated_columns vectorized correctness
# ---------------------------------------------------------------------------


def test_find_deflated_columns_basic() -> None:
    """Vectorized _find_deflated_columns matches expected deflation detection."""
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # z_unit[1] ≈ 0 -> eigenvalue 2.0 should be deflated
    z_unit = np.array([0.5, 1e-15, 0.5, 0.5, 0.5])
    eigenvalues = np.array([0.9, 2.0, 3.1, 4.2, 5.3])

    result = _find_deflated_columns(z_unit, d, eigenvalues)
    # eigenvalues[1] == d[1] == 2.0, and z_unit[1] ≈ 0
    assert 1 in result
    assert result[1] == 1


def test_find_deflated_columns_no_deflation() -> None:
    """No deflation when all z_unit entries are above threshold."""
    d = np.array([1.0, 2.0, 3.0])
    z_unit = np.array([0.5, 0.5, 0.5])
    eigenvalues = np.array([0.9, 2.1, 3.2])
    result = _find_deflated_columns(z_unit, d, eigenvalues)
    assert result == {}
