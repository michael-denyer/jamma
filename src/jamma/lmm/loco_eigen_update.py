"""Rotated-basis eigenvalue update for LOCO eigendecomposition.

Given the full-kinship eigendecomposition (d_full, U_full), compute the LOCO
eigendecomposition (d_loco, U_loco) for a chromosome by eigendecomposing the
rotated matrix:

    M = alpha_c * diag(d_full) - sigma * U_full^T @ S_chr @ U_full

where:
    alpha_c = p_full / (p_full - p_chr)   (scaling factor, > 1)
    sigma   = 1.0 / (p_full - p_chr)      (downdate weight)
    S_chr   = X_c @ X_c.T                 (chromosome Gram matrix)

The eigenvalues of M equal the eigenvalues of K_loco_c, and the eigenvectors
of K_loco_c are U_full @ V where V are the eigenvectors of M.

This avoids constructing K_loco_c = (S_full - S_chr) / p_loco explicitly.
The low-rank structure of M_gram enables a secular equation solver (via LAPACK
DLAED4) computing all n eigenvalues at O(n^2) cost per rank-1 update
(O(n) per eigenvalue via DLAED4), versus O(n^3) for direct
eigendecomposition. Two eigenvector accumulation paths:

- **Q path**: sequential rank-1 updates with O(n^3) eigenvector multiply
  (DGEMM on (n, n) Q matrix) per step. Total: O(n^3 * r_eff) compute,
  O(n^2) memory. Wins over direct eigh when r_eff constant factor is
  favorable (secular DGEMM cheaper than full tridiagonal eigensolver).
- **Delta path**: forward pass stores eigenvalues + norms only (O(n) per step),
  then backward pass reconstructs U_loco via blocked Cauchy multiply.
  Compute: O(n^3 * r_eff) same as Q path. Memory: O(r_eff * n) stored
  intermediates (no Q matrix, no V_j matrices) — the key advantage at
  large n (eliminates 55 GB Q = np.eye(n) at n=83k).
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

# Small eigenvalue threshold inspired by GEMMA's EigenDecomp_Zeroed, applied to
# absolute values to also handle small negative eigenvalues from numerical noise
# in the rank-k downdate (GEMMA itself zeros non-positive eigenvalues only).
_DEFAULT_THRESHOLD: float = 1e-10

# Guard against division by zero at deflation poles in Cauchy formulas.
# When |d[l] - eigenvalue[k]| < this value, the denominator is inflated to
# 1e300, making the term negligibly small (the corresponding z[l] ≈ 0 at
# deflation points, so the contribution vanishes).
_DEFLATION_GUARD: float = 1e-300
_DEFLATION_FILL: float = 1.0 / _DEFLATION_GUARD  # reciprocal used as safe fill


# Threshold below which ||q_j|| is treated as degenerate (skip rank-1 update).
# Two orders above float64 machine epsilon (~2.2e-16) to catch only genuine
# null-space vectors while staying safely above the floating-point noise floor.
_DEGENERATE_NORM_THRESHOLD: float = 1e-14

# Maximum n for Python fallback rank-1 update (allocates n×n dense matrix).
# Above this, np.diag(d) + np.outer(z, z) is prohibitively large:
#   n=10k → 800 MB, n=50k → 20 GB, n=85k → 58 GB per call.
# The delta path calls this r_eff times, so total is r_eff × n² × 8 bytes.
_MAX_PYTHON_FALLBACK_N: int = 10_000

# Default threshold for |z_unit[l]| below which a column is considered
# deflated in rank-1 secular update (DLAED4 type-1 deflation).
_DEFLATION_TOL_Z: float = 1e-10


# ---------------------------------------------------------------------------
# The _secular_accel C extension was removed in v4.0 (PR #57). The LOCO
# secular path now always uses the Python fallback (_rank1_update_python).
# _SECULAR_ACCEL_AVAILABLE is retained as a constant so call sites in this
# module and loco.py continue to work without structural changes.
# ---------------------------------------------------------------------------

_SECULAR_ACCEL_AVAILABLE: bool = False


def _rank1_update_python(
    d: np.ndarray,
    rho: float,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 eigenvalue update via np.linalg.eigh.

    Computes eigenvalues and eigenvectors of D + rho * z @ z.T where D =
    diag(d). This is O(n^3) per call.

    Args:
        d: (n,) ascending diagonal elements.
        rho: Scalar multiplier (can be negative for LOCO downdate).
        z: (n,) rank-1 update vector (used as-is; caller is responsible
            for adjusting rho if z is rescaled).

    Returns:
        Tuple (eigenvalues, eigenvectors) where eigenvalues are ascending and
        eigenvectors are columns of the (n, n) matrix.

    Raises:
        MemoryError: If n > _MAX_PYTHON_FALLBACK_N (allocates n×n dense matrix).
    """
    n = len(d)
    if n > _MAX_PYTHON_FALLBACK_N:
        mem_gb = n * n * 8 / 1e9
        raise MemoryError(
            f"Python fallback rank-1 update requires {mem_gb:.0f} GB "
            f"(n={n:,} dense matrix) — not viable. "
            f"C extension required for n > {_MAX_PYTHON_FALLBACK_N:,}"
        )
    M = np.diag(d) + rho * np.outer(z, z)
    try:
        return np.linalg.eigh(M)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"rank1_update_python: eigh failed on {len(d)}-dim matrix "
            f"(rho={rho:.6e}): {e}"
        ) from e


def _rank1_eigenvalues_and_norms_python(
    d: np.ndarray,
    rho: float,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure Python fallback for eigenvalues+norms of D + rho * z @ z.T.

    Computes eigenvalues via np.linalg.eigh (O(n^3) fallback) then computes
    norm_j[k] = ||z_unit / (d - eigenvalues[k])||_2 for each k.

    Args:
        d: (n,) ascending diagonal elements.
        rho: Scalar multiplier (can be negative for LOCO downdate).
        z: (n,) rank-1 update vector. Normalized to unit norm internally;
            rho is adjusted by ||z||^2 accordingly.

    Returns:
        Tuple (eigenvalues, norms) both shape (n,) ascending.

    Raises:
        MemoryError: If n > _MAX_PYTHON_FALLBACK_N (allocates n×n dense matrix).
    """
    n = len(d)
    if n > _MAX_PYTHON_FALLBACK_N:
        mem_gb = n * n * 8 / 1e9
        raise MemoryError(
            f"Python fallback rank-1 update requires {mem_gb:.0f} GB "
            f"(n={n:,} dense matrix) — not viable. "
            f"C extension required for n > {_MAX_PYTHON_FALLBACK_N:,}"
        )
    z_norm = np.linalg.norm(z)
    z_unit = z / z_norm if z_norm > 0.0 else z.copy()
    rho_eff = rho * (z_norm**2)

    M = np.diag(d) + rho_eff * np.outer(z_unit, z_unit)
    try:
        eigenvalues, _ = np.linalg.eigh(M)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"rank1_eigenvalues_and_norms_python: eigh failed on {len(d)}-dim "
            f"matrix (rho={rho:.6e}): {e}"
        ) from e

    # Compute norms: norm_j[k] = ||z_unit / (d - eigenvalues[k])||_2
    # Deflation guard: skip terms where |delta| < _DEFLATION_GUARD
    n = len(d)
    norms = np.empty(n)
    for k in range(n):
        delta_k = d - eigenvalues[k]
        safe = np.abs(delta_k) > _DEFLATION_GUARD
        norm_sq = np.sum((z_unit[safe] / delta_k[safe]) ** 2)
        norms[k] = np.sqrt(norm_sq) if norm_sq > 0.0 else 0.0

    return eigenvalues, norms


def _cauchy_block(d: np.ndarray, lam_block: np.ndarray) -> np.ndarray:
    """Compute guarded Cauchy matrix C[l, k] = 1 / (d[l] - lam_block[k]).

    Applies deflation guard: when |d[l] - lam_block[k]| < _DEFLATION_GUARD,
    sets denominator to _DEFLATION_FILL (~1e300), making the entry ~0.
    This handles deflated eigenvalue poles where z_j[l] ~ 0 by construction.

    Args:
        d: (n,) diagonal values before rank-1 update.
        lam_block: (m,) block of eigenvalues.

    Returns:
        (n, m) guarded Cauchy block.
    """
    diffs = d[:, np.newaxis] - lam_block
    diffs = np.where(np.abs(diffs) > _DEFLATION_GUARD, diffs, _DEFLATION_FILL)
    return 1.0 / diffs


def _apply_vj_to_rows_blocked(
    R: np.ndarray,
    z_j: np.ndarray,
    d_j: np.ndarray,
    lambda_j: np.ndarray,
    norm_j: np.ndarray,
    deflated: dict[int, int],
    col_block_size: int = 1000,
) -> np.ndarray:
    """Apply implicit V_j to row batch R without materializing n x n V_j.

    Computes R @ V_j where:
        V_j[l, k] = z_j[l] / (d_j[l] - lambda_j[k]) / norm_j[k]

    For deflated eigenvalues (where z_j[l] ≈ 0 and d_j[l] = lambda_j[k]),
    the Cauchy term is 0/0. Without explicit deflation handling, the inline
    guard sets the denominator to 1e300, making z_j[l] / 1e300 ≈ 0. When
    ``deflated`` is non-empty, deflated columns are explicitly overridden with
    the correct e_l basis vector.

    Uses blocked Cauchy multiply: processes col_block_size columns at a time,
    avoiding materialization of the full n x n Cauchy matrix.

    Args:
        R: (b, n) batch of b rows of the current accumulation matrix.
        z_j: (n,) unit-norm update vector for step j (may have near-zero components
            for deflated indices).
        d_j: (n,) diagonal at the START of step j (before rank-1 update).
        lambda_j: (n,) eigenvalues AFTER rank-1 update at step j (ascending).
        norm_j: (n,) normalization factors:
            norm_j[k] = ||z_j / (d_j - lambda_j[k])||_2,
            self-consistent with the Cauchy formula deflation guard.
        deflated: Dict mapping deflated column k -> row index l.
            R_new[:, k] = R[:, l] for deflated columns. Pass ``{}`` when
            no columns are deflated.
        col_block_size: Number of output columns to compute per block (memory control).

    Returns:
        (b, n) result of R @ V_j.
    """
    b, n = R.shape
    # A[i, l] = R[i, l] * z_j[l] — elementwise broadcast
    A = R * z_j  # (b, n)
    R_new = np.empty_like(R)
    for k_start in range(0, n, col_block_size):
        k_end = min(k_start + col_block_size, n)
        lam_block = lambda_j[k_start:k_end]  # (m,)
        C_block = _cauchy_block(d_j, lam_block)
        # DGEMM: (b, n) @ (n, m) -> (b, m); scale by 1/norm_j
        R_new[:, k_start:k_end] = (A @ C_block) * (1.0 / norm_j[k_start:k_end])
    # Override deflated columns: eigenvector k is e_{row_l}
    if deflated:
        for k, row_l in deflated.items():
            if 0 <= k < n:
                R_new[:, k] = R[:, row_l]
    return R_new


def _apply_vj_transpose_to_vec_blocked(
    v: np.ndarray,
    z_j: np.ndarray,
    d_j: np.ndarray,
    lambda_j: np.ndarray,
    norm_j: np.ndarray,
    deflated: dict[int, int],
    col_block_size: int = 1000,
) -> np.ndarray:
    """Apply V_j.T to a single vector v using blocked Cauchy multiply.

    Computes (V_j.T @ v) where:
        (V_j.T @ v)[k] = (1/norm_j[k]) * sum_l (z_j[l] * v[l]) / (d_j[l] - lambda_j[k])

    Used in the delta-path forward pass to project u_z[:,j] into the current
    eigenbasis without materializing Q.

    Args:
        v: (n,) input vector.
        z_j: (n,) unit-norm update vector for step j.
        d_j: (n,) diagonal at the START of step j (before rank-1 update).
        lambda_j: (n,) eigenvalues AFTER rank-1 update at step j (ascending).
        norm_j: (n,) normalization factors.
        deflated: Dict mapping deflated column k -> row index l.
            result[k] = v[l] for deflated positions. Pass ``{}`` when
            no columns are deflated.
        col_block_size: Number of output elements to compute per block.

    Returns:
        (n,) result of V_j.T @ v.
    """
    c = z_j * v  # (n,) elementwise: c[l] = z_j[l] * v[l]
    n = len(v)
    result = np.empty(n)
    for k_start in range(0, n, col_block_size):
        k_end = min(k_start + col_block_size, n)
        lam_block = lambda_j[k_start:k_end]  # (m,)
        C_block = _cauchy_block(d_j, lam_block)
        result[k_start:k_end] = (c @ C_block) / norm_j[k_start:k_end]
    # Override deflated positions: (V_j.T @ v)[k] = v[row_l]
    if deflated:
        for k, row_l in deflated.items():
            if 0 <= k < n:
                result[k] = v[row_l]
    return result


def _find_deflated_columns(
    z_unit: np.ndarray,
    d: np.ndarray,
    eigenvalues: np.ndarray,
    tol_z: float = _DEFLATION_TOL_Z,
) -> dict[int, int]:
    """Identify deflated eigenvector columns in a rank-1 secular update.

    A column k is deflated when z_unit[l] ≈ 0 AND eigenvalues[k] = d[l] (i.e.,
    the k-th eigenvalue coincides with d[l], meaning the k-th eigenvector is e_l).
    This matches DLAED4's "Type 1" deflation: when |z[l]| is below a threshold,
    the corresponding eigenvalue stays at d[l] and eigenvector = e_l.

    Args:
        z_unit: (n,) unit-norm update vector.
        d: (n,) diagonal before rank-1 update.
        eigenvalues: (n,) eigenvalues after rank-1 update, ascending.
        tol_z: Threshold below which z_unit[l] is considered deflated.

    Returns:
        Dict mapping column index k to row index l: deflated eigenvector at
        column k is e_l. Empty dict if no deflation detected.
    """
    deflated: dict[int, int] = {}
    # Find deflated d-indices: where |z_unit[idx]| < tol_z
    deflated_mask = np.abs(z_unit) < tol_z
    deflated_indices = np.where(deflated_mask)[0]
    for idx in deflated_indices:
        # Find eigenvalue k closest to d[idx] using binary search (O(log n)
        # vs O(n) argmin per index). eigenvalues is ascending.
        val = d[idx]
        k = int(np.searchsorted(eigenvalues, val))
        # searchsorted gives insertion point; check neighbors for closest
        if k == len(eigenvalues):
            k = k - 1
        elif k > 0 and abs(eigenvalues[k - 1] - val) < abs(eigenvalues[k] - val):
            k = k - 1
        rel_tol = 1e-14 * max(abs(val), 1.0)
        if abs(eigenvalues[k] - val) < rel_tol:
            deflated[k] = int(idx)
    return deflated


def _compute_cauchy_norms(
    z_unit: np.ndarray,
    d: np.ndarray,
    eigenvalues: np.ndarray,
    deflated: dict[int, int] | None = None,
    col_block_size: int = 1000,
) -> np.ndarray:
    """Compute norm_j[k] = ||z_unit / (d - eigenvalues[k])||_2 for all k.

    These norms are consistent with the blocked Cauchy formula in
    _apply_vj_to_rows_blocked: using these norms guarantees that Cauchy
    eigenvector columns are unit-norm by construction.

    Deflation guard: terms where |d[l] - eigenvalues[k]| < 1e-300 are
    treated as zero (deflated), contributing zero to the norm.

    For deflated columns (k in deflated dict), the norm is set to 1.0 since
    the eigenvector is e_l (unit norm), and the Cauchy formula is not used.

    Uses blocked computation to avoid materializing the full (n, n) Cauchy
    matrix. Processes col_block_size columns at a time.

    Args:
        z_unit: (n,) unit-norm update vector.
        d: (n,) diagonal before rank-1 update.
        eigenvalues: (n,) eigenvalues after rank-1 update, ascending.
        deflated: Optional dict mapping deflated column k -> row index l.
            If provided, norm[k] is set to 1.0 for deflated columns.
        col_block_size: Number of columns to process per block. Default: 1000.

    Returns:
        (n,) norms, all positive.
    """
    n = len(d)
    norms = np.empty(n)
    for k_start in range(0, n, col_block_size):
        k_end = min(k_start + col_block_size, n)
        lam_block = eigenvalues[k_start:k_end]  # (m,)
        # C_block[l, k] = 1 / (d[l] - eigenvalues[k_start + k]), shape (n, m)
        C_block = _cauchy_block(d, lam_block)
        ratios = z_unit[:, np.newaxis] * C_block  # (n, m)
        norm_sq = np.sum(ratios**2, axis=0)  # (m,)
        norms[k_start:k_end] = np.sqrt(norm_sq)
    # Guard against zero norm (degenerate all-deflated case)
    zero_mask = norms <= 0
    if np.any(zero_mask):
        non_deflated_zeros = [
            int(i)
            for i in np.where(zero_mask)[0]
            if deflated is None or i not in deflated
        ]
        if non_deflated_zeros:
            logger.warning(
                f"_compute_cauchy_norms: {len(non_deflated_zeros)} non-deflated "
                f"columns have zero norm (indices: {non_deflated_zeros[:5]}). "
                f"This may indicate numerical instability."
            )
        norms[zero_mask] = 1.0
    # For deflated columns: eigenvector is e_l (unit norm), so norm = 1.0
    if deflated:
        for k in deflated:
            norms[k] = 1.0
    return norms


def _secular_eigendecompose_delta_path(
    d_full: np.ndarray,
    U_full: np.ndarray,
    u_z: np.ndarray,
    s: np.ndarray,
    alpha_c: float,
    sigma: float,
    n: int,
    r_eff: int,
    threshold: float,
    row_batch_size: int,
    col_block_size: int,
    t0: float,
    check_orthogonality: bool = False,
    reorth_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Delta-path secular eigendecomposition for large n (no Q = np.eye(n) allocation).

    Two-pass algorithm:
    - Forward pass: for each step j, compute q_j = V_{j-1}.T @ ... @ V_0.T @ u_z[:,j]
      from scratch using stored (z_k, d_k, lambda_k, norm_k_cauchy) data.
      Cauchy norms are self-consistent: norm_k[l] = ||z_k / (d_k - lambda_k[l])||_2,
      ensuring V_k_cauchy is unitary by construction (column k has unit norm).
      Eigenvalues computed via np.linalg.eigh on the rank-1 updated matrix.
      Stores (z_j, d_j, lambda_j, norm_j_cauchy) — total O(4 * r_eff * n) memory.
    - Backward pass: reconstruct U_loco row-batch by row-batch via blocked
      Cauchy matrix multiply using stored (z_j, d_j, lambda_j, norm_j_cauchy).
      With self-consistent Cauchy norms, V_j_cauchy columns are unit-norm by
      construction, giving a proper unitary reconstruction.

    Forward pass cost: O(r_eff^2 * n^2) Cauchy transforms
      + O(r_eff * n^2) eigenvalue updates.
    Backward pass cost: O(n^3 * r_eff) — blocked Cauchy multiply across all
      n rows (each step applies an implicit (n, n) matrix to all rows).
    Memory: O(4 * r_eff * n) stored intermediates + O(batch * n) row buffers
      (no Q matrix, no V_j matrices).

    Args:
        d_full: (n,) full eigenvalues, ascending.
        U_full: (n, n) full eigenvectors.
        u_z: (n, r_eff) left singular vectors of Z = U_full.T @ X_c.
        s: (r_eff,) singular values (descending).
        alpha_c: Scaling factor p_full / (p_full - p_chr).
        sigma: Downdate weight 1 / (p_full - p_chr).
        n: Problem dimension.
        r_eff: Effective rank.
        threshold: Eigenvalue threshold for zeroing near-zero values.
        row_batch_size: Number of rows to process per backward-pass batch.
        col_block_size: Number of output columns per Cauchy block.
        t0: Start time for elapsed logging.

    Returns:
        Tuple (d_loco, U_loco).
    """
    # Pre-allocate step storage for forward pass
    stored_z = np.empty((r_eff, n))
    stored_d = np.empty((r_eff, n))
    stored_lambda = np.empty((r_eff, n))
    stored_norm = np.empty((r_eff, n))
    # Track which steps are degenerate (norm_q < _DEGENERATE_NORM_THRESHOLD):
    # skip in backward pass.
    step_is_identity = np.zeros(r_eff, dtype=bool)
    # Track deflation maps: stored_deflated[j] maps deflated column k -> row l
    stored_deflated: list[dict[int, int]] = [{} for _ in range(r_eff)]

    d_current = alpha_c * d_full.copy()
    n_fallbacks = 0

    # Forward pass: for each step j, compute q_j = V_{j-1}^T @ ... @ V_0^T @ u_z[:,j]
    # by applying all stored V_k^T using the Cauchy formula with self-consistent norms
    # and explicit deflation handling.
    #
    # Self-consistent norms: norm_k[l] = ||z_k / (d_k - lambda_k[l])||_2 (same
    # denominators as the Cauchy formula), so V_k_cauchy[:,l] has unit norm.
    # Deflation-aware: when z_k[l] ≈ 0 and lambda_k[m] = d_k[l], V_k[:,m] = e_l.
    for j in range(r_eff):
        rho_j = -sigma * s[j] ** 2

        # Project u_z[:,j] through all previous V_k^T (k=0..j-1) using stored data
        q_implicit = u_z[:, j].copy()
        for k in range(j):
            if not step_is_identity[k]:
                q_implicit = _apply_vj_transpose_to_vec_blocked(
                    q_implicit,
                    stored_z[k],
                    stored_d[k],
                    stored_lambda[k],
                    stored_norm[k],
                    deflated=stored_deflated[k],
                    col_block_size=col_block_size,
                )

        norm_q = np.linalg.norm(q_implicit)
        if norm_q < _DEGENERATE_NORM_THRESHOLD:
            # Degenerate: this singular vector is zero in current basis — skip.
            # Mark as identity step: V_j = I, no eigenvalue change.
            step_is_identity[j] = True
            stored_z[j] = np.zeros(n)
            stored_d[j] = d_current.copy()
            stored_lambda[j] = d_current.copy()
            stored_norm[j] = np.ones(n)
            continue

        rho_j_eff = rho_j * norm_q**2
        q_j_unit = q_implicit / norm_q

        # Eigenvalues-only via rank-1 update (Python fallback).
        # The delta path computes its own Cauchy norms afterward, so we
        # discard the norms from the Python eigh.
        lambda_j, _ = _rank1_eigenvalues_and_norms_python(
            d_current, rho_j_eff, q_j_unit
        )

        # Guard against non-finite eigenvalues from DLAED4 or Python fallback.
        # A single NaN propagates through stored step data and corrupts the
        # backward Cauchy reconstruction silently.
        if not np.all(np.isfinite(lambda_j)):
            n_bad = int(np.sum(~np.isfinite(lambda_j)))
            raise RuntimeError(
                f"Secular delta-path step j={j}/{r_eff}: {n_bad} non-finite "
                f"eigenvalues detected (n={n}, rho_j_eff={rho_j_eff:.6e}). "
                f"This indicates numerical instability in the rank-1 update. "
                f"Re-run without --secular."
            )

        # Ensure strict ascending order (DLAED4 guarantees it, but verify FP safety)
        if not np.all(np.diff(lambda_j) >= 0):
            lambda_j = lambda_j[np.argsort(lambda_j)]

        # Detect deflated columns: where |q_j_unit[l]| ≈ 0 AND lambda[k] = d[l].
        # DLAED4 deflation type 1: q_unit[l] ≈ 0 -> eigenvalue stays at d[l],
        # eigenvector = e_l. The Cauchy formula breaks down at these poles.
        deflated_j = _find_deflated_columns(q_j_unit, d_current, lambda_j)

        # Compute Cauchy-consistent norms with deflation handling:
        # norm_j[k] = ||z_j / (d_j - lambda_j[k])||_2 (self-consistent with Cauchy),
        # but set to 1.0 for deflated columns (eigenvector is e_l, unit norm).
        norm_j = _compute_cauchy_norms(
            q_j_unit, d_current, lambda_j, deflated_j, col_block_size
        )

        # Store step-j data: d BEFORE update — using d AFTER would give wrong
        # Cauchy denominators (d_k - lambda_k[l]) in backward pass
        stored_z[j] = q_j_unit
        stored_d[j] = d_current.copy()  # d_j = diagonal BEFORE step j
        stored_lambda[j] = lambda_j
        stored_norm[j] = norm_j
        stored_deflated[j] = deflated_j

        d_current = lambda_j

    if n_fallbacks > 0:
        logger.warning(
            f"DLAED4 fell back to Python eigh on {n_fallbacks}/{r_eff} delta-path "
            f"steps (n={n}). Each fallback is O(n^3) instead of O(n^2). "
            f"Check input data if this is unexpected."
        )

    # Backward pass: reconstruct U_loco row-batch by row-batch using Cauchy formula.
    # For each row batch R from U_full, apply R = R @ V_j for j=0..r_eff-1.
    # V_j is implicit: V_j[l,k] = z_j[l] / (d_j[l] - lambda_j[k]) / norm_j[k].
    # Deflated columns use e_l convention: R_new[:,k] = R[:,l].
    #
    # Pre-allocate buffers to avoid per-step allocation (memory-critical for
    # large n: two (b, n) row buffers + one (b, n) A buffer).
    U_loco = np.empty_like(U_full)
    b_max = min(row_batch_size, n)
    R_buf0 = np.empty((b_max, n))
    R_buf1 = np.empty((b_max, n))
    A_buf = np.empty((b_max, n))

    for row_start in range(0, n, row_batch_size):
        row_end = min(row_start + row_batch_size, n)
        b = row_end - row_start
        R = R_buf0[:b]
        R[:] = U_full[row_start:row_end, :]
        cur = 0  # tracks which buffer R points to (0 or 1)

        for j in range(r_eff):
            if step_is_identity[j]:
                continue
            z_j = stored_z[j]
            d_j = stored_d[j]
            lam_j = stored_lambda[j]
            norm_j_arr = stored_norm[j]
            defl_j = stored_deflated[j]

            R_out = (R_buf1 if cur == 0 else R_buf0)[:b]
            A = A_buf[:b]
            np.multiply(R, z_j, out=A)  # A[i, l] = R[i, l] * z_j[l]

            for k_start in range(0, n, col_block_size):
                k_end = min(k_start + col_block_size, n)
                lam_block = lam_j[k_start:k_end]
                C_block = _cauchy_block(d_j, lam_block)
                R_out[:, k_start:k_end] = (A @ C_block) * (
                    1.0 / norm_j_arr[k_start:k_end]
                )
            # Override deflated columns: eigenvector k is e_{row_l}
            for k, row_l in defl_j.items():
                if 0 <= k < n:
                    R_out[:, k] = R[:, row_l]

            R = R_out
            cur = 1 - cur

        U_loco[row_start:row_end, :] = R

    # Post-hoc orthogonality check and optional QR re-orthogonalization (delta path)
    if check_orthogonality:
        U_loco = _check_and_reorthogonalize(U_loco, r_eff, "delta", reorth_threshold)

    # Zero near-zero eigenvalues
    _apply_threshold(d_current, threshold)

    elapsed = time.perf_counter() - t0
    logger.debug(
        f"secular_eigendecompose_from_full: n={n}, r_eff={r_eff}, "
        f"delta path, elapsed={elapsed:.3f}s"
    )

    return d_current, U_loco


def loco_eigendecompose_from_full(
    d_full: np.ndarray,
    U_full: np.ndarray,
    S_chr: np.ndarray,
    p_full: int,
    p_chr: int,
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute LOCO eigendecomposition from the full-kinship eigendecomposition.

    Eigendecomposes the rotated matrix M = alpha_c * diag(d_full) - sigma *
    U_full^T @ S_chr @ U_full to obtain (d_loco, U_loco) without constructing
    K_loco_c = (S_full - S_chr) / (p_full - p_chr) explicitly.

    Args:
        d_full: (n,) eigenvalues of K_full, ascending order.
        U_full: (n, n) eigenvectors of K_full (columns are eigenvectors).
        S_chr: (n, n) chromosome Gram matrix X_c @ X_c.T.
        p_full: Total number of SNPs used to build K_full.
        p_chr: Number of SNPs on the chromosome being excluded. May be 0.
        threshold: Eigenvalues with |value| < threshold are zeroed (inspired
            by GEMMA's EigenDecomp_Zeroed, extended to absolute values to
            handle numerical noise from the downdate). Default: 1e-10.

    Returns:
        Tuple (d_loco, U_loco) where:
        - d_loco: (n,) LOCO eigenvalues in ascending order.
        - U_loco: (n, n) LOCO eigenvectors (columns are eigenvectors).

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    n = d_full.shape[0]
    if d_full.ndim != 1:
        raise ValueError(f"d_full must be 1-D, got shape {d_full.shape}")
    if U_full.shape != (n, n):
        raise ValueError(f"U_full must be ({n}, {n}), got shape {U_full.shape}")
    if S_chr.shape != (n, n):
        raise ValueError(f"S_chr must be ({n}, {n}), got shape {S_chr.shape}")
    if p_chr < 0 or p_chr > p_full:
        raise ValueError(
            f"p_chr must be in [0, p_full], got p_chr={p_chr}, p_full={p_full}"
        )
    if p_chr == p_full:
        raise ValueError(
            f"p_chr == p_full ({p_full}): cannot exclude all SNPs. "
            f"The LOCO kinship for this chromosome has no remaining SNPs."
        )

    t0 = time.perf_counter()

    if p_chr == 0:
        # Degenerate case: alpha_c = 1.0, S_chr = 0 so M_gram = 0.
        # M = diag(d_full), eigenvectors are identity, U_loco = U_full.
        logger.debug(
            "loco_eigendecompose_from_full: p_chr=0, returning d_full unchanged"
        )
        d_loco = d_full.copy()
        U_loco = U_full.copy()
        _apply_threshold(d_loco, threshold)
        return d_loco, U_loco

    alpha_c = p_full / (p_full - p_chr)
    sigma = 1.0 / (p_full - p_chr)

    # Rotate chromosome Gram matrix into full eigen-basis: O(n^3) BLAS3
    # M = U_full^T @ S_chr @ U_full, shape (n, n), then transform in-place
    M = np.matmul(U_full.T, np.matmul(S_chr, U_full))

    # M = alpha_c * diag(d_full) - sigma * M  (in-place, avoids np.diag allocation)
    M *= -sigma
    M.flat[:: n + 1] += alpha_c * d_full

    # Eigendecompose M: O(n^3), uses numpy (LAPACK DSYEVD / ILP64-safe)
    try:
        d_loco, V = np.linalg.eigh(M)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"loco_eigendecompose_from_full: eigendecomposition of rotated "
            f"matrix failed (n={n}, p_chr={p_chr}, alpha_c={alpha_c:.6f}): {e}"
        ) from e

    # Map eigenvectors back to original basis: U_loco = U_full @ V, O(n^3)
    U_loco = np.matmul(U_full, V)

    # Zero near-zero eigenvalues from numerical noise (see _apply_threshold)
    _apply_threshold(d_loco, threshold)

    elapsed = time.perf_counter() - t0
    logger.debug(
        f"loco_eigendecompose_from_full: n={n}, p_chr={p_chr}, "
        f"alpha_c={alpha_c:.6f}, elapsed={elapsed:.3f}s"
    )

    return d_loco, U_loco


def secular_eigendecompose_from_full(
    d_full: np.ndarray,
    U_full: np.ndarray,
    X_c: np.ndarray,
    p_full: int,
    p_chr: int,
    threshold_ratio: float = 1e-8,
    threshold: float = _DEFAULT_THRESHOLD,
    n_threshold_for_delta: int = 5000,
    row_batch_size: int = 1000,
    col_block_size: int = 1000,
    check_orthogonality: bool = False,
    reorth_interval: int | None = None,
    reorth_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Secular equation solver for LOCO eigenvalue update.

    Complexity: O(n^3 * r_eff) compute for both Q and delta paths (DGEMM or
    blocked Cauchy multiply per step). Delta path saves O(n^2) memory by
    eliminating the Q matrix. When r_eff << n, wins over direct eigh via
    favorable constant factors (secular DGEMM vs full tridiagonal solver).

    Replaces the O(n^3) np.linalg.eigh(M) call in loco_eigendecompose_from_full
    with a secular equation solver that exploits the low-rank structure of the
    chromosome genotype matrix X_c.

    The rotated matrix is:
        M = alpha_c * diag(d_full) - sigma * Z Z^T
    where Z = U_full^T @ X_c has rank r_eff << n due to LD structure.
    The secular solver applies r_eff sequential rank-1 updates via DLAED4,
    each computing eigenvalues at O(n) per eigenvalue.

    Two eigenvector accumulation strategies:
    - Q path (n <= n_threshold_for_delta): tracked as n x n matrix Q updated
      at each step via O(n^3) DGEMM. Memory: O(n^2) — fine for small n.
    - Delta path (n > n_threshold_for_delta): forward pass stores
      (z_j, d_j, lambda_j, norm_j) per step; backward pass reconstructs
      U_loco row-batch by row-batch via blocked Cauchy multiply. Eliminates
      the Q = np.eye(n) allocation (55 GB at n=83k), reducing peak memory
      from ~110 GB to ~58 GB.

    Rank-1 updates use _rank1_update_python (np.linalg.eigh, O(n^3) per step).

    Args:
        d_full: (n,) eigenvalues of K_full, ascending order.
        U_full: (n, n) eigenvectors of K_full (columns are eigenvectors).
        X_c: (n, p_chr) centered chromosome genotype matrix.
        p_full: Total number of SNPs used to build K_full.
        p_chr: Number of SNPs on the chromosome being excluded. May be 0.
        threshold_ratio: Singular values below threshold_ratio * s_max are
            considered negligible (effective rank cutoff). Default: 1e-8.
        threshold: Eigenvalues with |value| < threshold are zeroed (inspired
            by GEMMA's EigenDecomp_Zeroed). Default: 1e-10.
        n_threshold_for_delta: Use delta path when n > this value; use Q path
            when n <= this value. Default: 5000.
        row_batch_size: Number of rows per backward-pass batch (delta path only).
            Default: 1000. Lower values reduce peak memory at cost of more loops.
        col_block_size: Number of output columns per Cauchy block (delta path only).
            Default: 1000. Lower values reduce peak memory at cost of more DGEMM calls.
        check_orthogonality: If True, compute max|U^T U - I| after eigenvector
            reconstruction and log via logger.debug. If deviation > reorth_threshold,
            applies QR re-orthogonalization and logs a warning. Default: False.
        reorth_interval: Q path only. If not None, apply QR re-orthogonalization to
            the accumulated Q matrix every reorth_interval steps. Reduces drift from
            floating-point accumulation across many steps. Default: None (disabled).
        reorth_threshold: Deviation threshold above which post-hoc QR is applied when
            check_orthogonality=True. Default: 1e-6.

    Returns:
        Tuple (d_loco, U_loco) where:
        - d_loco: (n,) LOCO eigenvalues in ascending order.
        - U_loco: (n, n) LOCO eigenvectors (columns are eigenvectors).

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    n = d_full.shape[0]
    if d_full.ndim != 1:
        raise ValueError(f"d_full must be 1-D, got shape {d_full.shape}")
    if U_full.shape != (n, n):
        raise ValueError(f"U_full must be ({n}, {n}), got shape {U_full.shape}")
    if X_c.ndim != 2 or X_c.shape[0] != n:
        raise ValueError(f"X_c must be (n, p_chr) with n={n}, got shape {X_c.shape}")
    if p_chr < 0 or p_chr > p_full:
        raise ValueError(
            f"p_chr must be in [0, p_full], got p_chr={p_chr}, p_full={p_full}"
        )
    if p_chr == p_full:
        raise ValueError(f"p_chr == p_full ({p_full}): cannot exclude all SNPs.")

    t0 = time.perf_counter()

    # Degenerate case: empty chromosome (p_chr=0 or zero columns)
    actual_p_chr = X_c.shape[1]
    if p_chr == 0 or actual_p_chr == 0:
        logger.debug(
            "secular_eigendecompose_from_full: p_chr=0, returning d_full unchanged"
        )
        d_loco = d_full.copy()
        U_loco = U_full.copy()
        _apply_threshold(d_loco, threshold)
        return d_loco, U_loco

    alpha_c = p_full / (p_full - p_chr)
    sigma = 1.0 / (p_full - p_chr)

    # Z = U_full^T @ X_c, shape (n, p_chr) — rotated chromosome genotypes
    # Thin SVD of Z to get r_eff singular vectors
    Z = np.matmul(U_full.T, X_c)
    try:
        u_z, s, _ = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"SVD of rotated chromosome genotype matrix Z (shape {Z.shape}) "
            f"failed in secular_eigendecompose_from_full: {e}"
        ) from e

    # Effective rank: singular values above threshold_ratio * s_max
    if len(s) == 0 or s[0] == 0.0:
        r_eff = 0
    else:
        r_eff = int(np.sum(s > threshold_ratio * s[0]))

    logger.debug(
        f"secular_eigendecompose_from_full: n={n}, p_chr={p_chr}, "
        f"r_eff={r_eff}, alpha_c={alpha_c:.6f}"
    )

    # If r_eff == 0, the chromosome has no significant signal — return scaled d_full
    if r_eff == 0:
        d_loco = alpha_c * d_full.copy()
        U_loco = U_full.copy()
        _apply_threshold(d_loco, threshold)
        return d_loco, U_loco

    u_z = u_z[:, :r_eff]
    s = s[:r_eff]

    # Route to delta path for large n (avoids Q = np.eye(n) allocation)
    if n > n_threshold_for_delta:
        logger.debug(
            f"secular_eigendecompose_from_full: n={n} > n_threshold_for_delta="
            f"{n_threshold_for_delta}, using delta path"
        )
        return _secular_eigendecompose_delta_path(
            d_full,
            U_full,
            u_z,
            s,
            alpha_c,
            sigma,
            n,
            r_eff,
            threshold,
            row_batch_size,
            col_block_size,
            t0,
            check_orthogonality=check_orthogonality,
            reorth_threshold=reorth_threshold,
        )

    # Sequential rank-1 updates:
    # Start: D_0 = alpha_c * d_full (ascending), Q_0 = I_n
    # For j = 0..r_eff-1:
    #   rho_j = -sigma * s[j]^2
    #   q_j = Q_{j-1}^T @ u_z[:, j]   (project singular vector into current basis)
    #   (d_new, V_j) = rank1_update(d_current, rho_j, q_j)
    #   Q_{j} = Q_{j-1} @ V_j
    # Final: d_loco = D_{r_eff}, U_loco = U_full @ Q_{r_eff}
    d_current = alpha_c * d_full.copy()
    Q = np.eye(n)  # accumulated eigenvector rotation matrix, starts as identity
    n_fallbacks = 0

    for j in range(r_eff):
        rho_j = -sigma * s[j] ** 2

        # Project j-th left singular vector into current basis
        # On first step: Q is identity, so q_j = u_z[:, j]
        # On subsequent steps: q_j = Q^T @ u_z[:, j] (project into rotated basis)
        q_j = Q.T @ u_z[:, j]

        # Normalize q_j; adjust rho_j to preserve the perturbation magnitude.
        # rank1_update requires unit-norm z; rho_j is adjusted accordingly.
        norm_q = np.linalg.norm(q_j)
        if norm_q < _DEGENERATE_NORM_THRESHOLD:
            # Degenerate: ||q_j|| has drifted below _DEGENERATE_NORM_THRESHOLD,
            # indicating severe floating-point erosion in the accumulated Q matrix.
            # Skip this near-zero rank-1 update (numerical noise, not meaningful).
            logger.debug(
                f"secular Q-path step j={j}/{r_eff}: degenerate "
                f"(norm_q={norm_q:.2e}), skipping rank-1 update"
            )
            continue
        rho_j_eff = rho_j * norm_q**2
        q_j_unit = q_j / norm_q

        # Rank-1 update: (D_{j+1}, V_j) for D_j + rho_j_eff * q_j * q_j^T
        d_new, V_j = _rank1_update_python(d_current, rho_j_eff, q_j_unit)

        # Ensure strict ascending order after each update (DLAED4 requires it).
        # The update should already return ascending eigenvalues, but floating point
        # accumulation across many steps can introduce tiny violations.
        if not np.all(np.diff(d_new) >= 0):
            sort_idx = np.argsort(d_new)
            d_new = d_new[sort_idx]
            V_j = V_j[:, sort_idx]

        d_current = d_new
        # Accumulate: Q_{j+1} = Q_j @ V_j
        Q = Q @ V_j

        # Periodic QR re-orthogonalization to limit drift across many steps
        if reorth_interval is not None and (j + 1) % reorth_interval == 0:
            Q, _ = np.linalg.qr(Q, mode="reduced")

    if n_fallbacks > 0:
        logger.warning(
            f"DLAED4 fell back to Python eigh on {n_fallbacks}/{r_eff} Q-path "
            f"steps (n={n}). Each fallback is O(n^3) instead of O(n^2). "
            f"Check input data if this is unexpected."
        )

    # Back-rotate: U_loco = U_full @ Q
    U_loco = np.matmul(U_full, Q)

    # Post-hoc orthogonality check and optional QR re-orthogonalization (Q path)
    if check_orthogonality:
        U_loco = _check_and_reorthogonalize(U_loco, r_eff, "Q", reorth_threshold)

    # Zero near-zero eigenvalues (same as loco_eigendecompose_from_full)
    _apply_threshold(d_current, threshold)

    elapsed = time.perf_counter() - t0
    logger.debug(
        f"secular_eigendecompose_from_full: n={n}, p_chr={p_chr}, r_eff={r_eff}, "
        f"elapsed={elapsed:.3f}s"
    )

    return d_current, U_loco


def measure_effective_rank(
    U_full: np.ndarray,
    X_c: np.ndarray,
    threshold_ratio: float = 1e-8,
) -> tuple[int, np.ndarray]:
    """Measure the effective rank of the rotated chromosome genotype matrix.

    Computes Z = U_full^T @ X_c and applies thin SVD to determine the number
    of significant singular values. This is the effective rank r_eff — the
    number of rank-1 updates that would be required in a future secular
    equation solver. Low r_eff indicates strong LD structure.

    Args:
        U_full: (n, n) eigenvectors of full kinship matrix.
        X_c: (n, p_c) centered chromosome genotype matrix.
        threshold_ratio: Singular values below threshold_ratio * s_max are
            considered negligible. Default: 1e-8.

    Returns:
        Tuple (r_eff, singular_values) where:
        - r_eff: Number of singular values above the threshold (int).
        - singular_values: (min(n, p_c),) all singular values in descending
          order.
    """
    # Rotate chromosome genotypes into full eigen-basis: Z shape (n, p_c)
    Z = np.matmul(U_full.T, X_c)

    # Thin SVD of Z: returns singular values only (no U_z or V_z needed)
    s = np.linalg.svd(Z, compute_uv=False)

    # Effective rank: number of singular values above threshold
    if len(s) == 0 or s[0] == 0.0:
        r_eff = 0
    else:
        r_eff = int(np.sum(s > threshold_ratio * s[0]))

    n_samples = U_full.shape[0]
    p_c = X_c.shape[1]
    compression_pct = 100.0 * (1 - r_eff / max(p_c, 1))
    s_max = float(s[0]) if len(s) > 0 else 0.0
    s_min = float(s[-1]) if len(s) > 0 else 0.0
    logger.debug(
        f"measure_effective_rank: n={n_samples}, p_c={p_c}, r_eff={r_eff}, "
        f"compression={compression_pct:.0f}%, s_max={s_max:.4f}, s_min={s_min:.4e}"
    )

    return r_eff, s


def _apply_threshold(d: np.ndarray, threshold: float) -> None:
    """Zero eigenvalues below threshold in-place (inspired by GEMMA EigenDecomp_Zeroed).

    Args:
        d: Eigenvalue array, modified in place.
        threshold: Eigenvalues with |value| < threshold are set to 0.
    """
    mask = np.abs(d) < threshold
    n_zeroed = int(np.sum(mask))
    if n_zeroed > 0:
        min_val = float(np.min(d[mask]))
        logger.debug(
            f"_apply_threshold: zeroed {n_zeroed}/{len(d)} eigenvalues "
            f"(threshold={threshold:.1e}, min_zeroed={min_val:.2e})"
        )
        if min_val < -threshold * 100:
            logger.warning(
                f"_apply_threshold: significantly negative eigenvalue "
                f"({min_val:.2e}) suggests numerical instability in the "
                f"rank-k downdate, not just noise. Re-run without --secular "
                f"to verify results match"
            )
        d[mask] = 0.0


def _check_and_reorthogonalize(
    U: np.ndarray,
    r_eff: int,
    path_name: str,
    reorth_threshold: float,
) -> np.ndarray:
    """Check eigenvector orthogonality and apply QR if drift exceeds threshold.

    Args:
        U: (n, n) eigenvector matrix.
        r_eff: Effective rank (for logging).
        path_name: "Q" or "delta" (for logging).
        reorth_threshold: max|U^T U - I| above which QR is applied.

    Returns:
        U, possibly QR-reorthogonalized.
    """
    n = U.shape[0]
    gram = U.T @ U
    deviation = float(np.max(np.abs(gram - np.eye(n))))
    logger.debug(
        f"secular_eigendecompose_from_full: orthogonality check "
        f"max|U^T U - I| = {deviation:.2e} (n={n}, r_eff={r_eff}, path={path_name})"
    )
    if deviation > reorth_threshold:
        logger.warning(
            f"secular_eigendecompose_from_full: eigenvector orthogonality "
            f"drift detected: max|U^T U - I| = {deviation:.2e}. "
            f"Applying QR re-orthogonalization."
        )
        U, _ = np.linalg.qr(U, mode="reduced")
    return U
