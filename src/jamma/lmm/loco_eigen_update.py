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

This avoids constructing K_loco_c = (S_full - S_chr) / p_loco explicitly,
but has the same O(n^3) cost as direct eigendecomposition (with additional
constant-factor overhead from three extra n x n matmuls: two for the Gram
rotation and one for eigenvector back-rotation). The practical benefit comes
in a future phase when the low-rank structure of M_gram enables a secular
equation solver at O(n^2 * r_eff) when r_eff << n.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

# Small eigenvalue threshold inspired by GEMMA's EigenDecomp_Zeroed, applied to
# absolute values to also handle small negative eigenvalues from numerical noise
# in the rank-k downdate (GEMMA itself zeros non-positive eigenvalues only).
_DEFAULT_THRESHOLD: float = 1e-10

# ---------------------------------------------------------------------------
# C extension for rank-1 eigenvalue update via LAPACK DLAED4
# ---------------------------------------------------------------------------

try:
    from jamma.lmm._secular_accel import rank1_eigenvalue_update as _rank1_update_c

    _SECULAR_ACCEL_AVAILABLE = True
except ImportError:
    _SECULAR_ACCEL_AVAILABLE = False


def _rank1_update_python(
    d: np.ndarray,
    rho: float,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure Python fallback for rank-1 eigenvalue update via np.linalg.eigh.

    Computes eigenvalues and eigenvectors of D + rho * z @ z.T where D =
    diag(d). This is O(n^3) and only used when the C extension (_secular_accel)
    is unavailable. The C extension uses LAPACK DLAED4 (O(n^2) per call).

    Args:
        d: (n,) ascending diagonal elements.
        rho: Scalar multiplier (can be negative for LOCO downdate).
        z: (n,) rank-1 update vector (normalised internally).

    Returns:
        Tuple (eigenvalues, eigenvectors) where eigenvalues are ascending and
        eigenvectors are columns of the (n, n) matrix.
    """
    M = np.diag(d) + rho * np.outer(z, z)
    return np.linalg.eigh(M)


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
) -> tuple[np.ndarray, np.ndarray]:
    """O(n^2 * r_eff) secular equation solver for LOCO eigenvalue update.

    Replaces the O(n^3) np.linalg.eigh(M) call in loco_eigendecompose_from_full
    with a secular equation solver that exploits the low-rank structure of the
    chromosome genotype matrix X_c.

    The rotated matrix is:
        M = alpha_c * diag(d_full) - sigma * Z Z^T
    where Z = U_full^T @ X_c has rank r_eff << n due to LD structure.
    The secular solver applies r_eff sequential rank-1 updates via DLAED4,
    each costing O(n^2), for a total of O(n^2 * r_eff).

    For full Q accumulation, eigenvectors are tracked as an n x n matrix Q
    updated at each step. This is fine for n <= ~5000 (see RESEARCH.md Pitfall 5
    for the 83k-scale optimization that is explicitly deferred).

    When the C extension (_secular_accel) is unavailable, falls back to
    _rank1_update_python (O(n^3) per step, same result).

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
    u_z, s, _ = np.linalg.svd(Z, full_matrices=False)

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

    # Choose rank-1 update function: C extension or Python fallback
    _rank1_fn = _rank1_update_c if _SECULAR_ACCEL_AVAILABLE else _rank1_update_python

    # Sequential rank-1 updates (RESEARCH.md "Sequential Rank-1 Update Strategy"):
    # Start: D_0 = alpha_c * d_full (ascending), Q_0 = I_n
    # For j = 0..r_eff-1:
    #   rho_j = -sigma * s[j]^2
    #   q_j = Q_{j-1}^T @ u_z[:, j]   (project singular vector into current basis)
    #   (d_new, V_j) = rank1_update(d_current, rho_j, q_j)
    #   Q_{j} = Q_{j-1} @ V_j
    # Final: d_loco = D_{r_eff}, U_loco = U_full @ Q_{r_eff}
    d_current = alpha_c * d_full.copy()
    Q = np.eye(n)  # accumulated eigenvector rotation matrix, starts as identity

    for j in range(r_eff):
        rho_j = -sigma * s[j] ** 2

        # Project j-th left singular vector into current basis
        # On first step: Q is identity, so q_j = u_z[:, j]
        # On subsequent steps: q_j = Q^T @ u_z[:, j] (project into rotated basis)
        q_j = Q.T @ u_z[:, j]

        # Normalize q_j; adjust rho_j to preserve the perturbation magnitude.
        # rank1_update requires unit-norm z; rho_j is adjusted accordingly.
        norm_q = np.linalg.norm(q_j)
        if norm_q < 1e-14:
            # Degenerate: this singular vector is orthogonal to all current basis
            # vectors. Skip this rank-1 update (eigenvalues unchanged for this step).
            continue
        rho_j_eff = rho_j * norm_q**2
        q_j_unit = q_j / norm_q

        # Rank-1 update: (D_{j+1}, V_j) for D_j + rho_j_eff * q_j * q_j^T
        # DLAED4 convergence failure falls back to Python eigh for this step.
        if _SECULAR_ACCEL_AVAILABLE:
            try:
                d_new, V_j = _rank1_update_c(d_current, rho_j_eff, q_j_unit)
            except RuntimeError:
                logger.debug(
                    f"secular_eigendecompose_from_full: DLAED4 convergence failure "
                    f"at step j={j}, falling back to Python eigh for this step"
                )
                d_new, V_j = _rank1_update_python(d_current, rho_j_eff, q_j_unit)
        else:
            d_new, V_j = _rank1_update_python(d_current, rho_j_eff, q_j_unit)

        # Ensure strict ascending order after each update (DLAED4 requires it).
        # The update should already return ascending eigenvalues, but floating point
        # accumulation across many steps can introduce tiny violations.
        sort_idx = np.argsort(d_new)
        if not np.all(sort_idx == np.arange(len(d_new))):
            d_new = d_new[sort_idx]
            V_j = V_j[:, sort_idx]

        d_current = d_new
        # Accumulate: Q_{j+1} = Q_j @ V_j
        Q = Q @ V_j

    # Back-rotate: U_loco = U_full @ Q
    U_loco = np.matmul(U_full, Q)

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
                f"rank-k downdate, not just noise"
            )
        d[mask] = 0.0
