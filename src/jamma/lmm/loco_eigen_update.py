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
constant-factor overhead from two extra n x n matmuls). The practical benefit
comes in a future phase when the rank-k structure of M_gram enables a secular
equation solver at O(n^2 * r_eff).
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

# Small eigenvalue threshold matching GEMMA's EigenDecomp_Zeroed behaviour
_DEFAULT_THRESHOLD: float = 1e-10


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
        threshold: Eigenvalues with |value| < threshold are zeroed, matching
            GEMMA's EigenDecomp_Zeroed behaviour. Default: 1e-10.

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
    # M_gram = U_full^T @ S_chr @ U_full, shape (n, n)
    M_gram = np.matmul(U_full.T, np.matmul(S_chr, U_full))

    # Construct rotated matrix M = alpha_c * diag(d_full) - sigma * M_gram
    M = np.diag(alpha_c * d_full) - sigma * M_gram

    # Eigendecompose M: O(n^3), uses numpy (LAPACK DSYEVD / ILP64-safe)
    d_loco, V = np.linalg.eigh(M)

    # Map eigenvectors back to original basis: U_loco = U_full @ V, O(n^3)
    U_loco = np.matmul(U_full, V)

    # Apply GEMMA threshold to small eigenvalues
    _apply_threshold(d_loco, threshold)

    elapsed = time.perf_counter() - t0
    logger.debug(
        f"loco_eigendecompose_from_full: n={n}, p_chr={p_chr}, "
        f"alpha_c={alpha_c:.6f}, elapsed={elapsed:.3f}s"
    )

    return d_loco, U_loco


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
    """Zero eigenvalues below threshold in-place (GEMMA EigenDecomp_Zeroed).

    Args:
        d: Eigenvalue array, modified in place.
        threshold: Eigenvalues with |value| < threshold are set to 0.
    """
    d[np.abs(d) < threshold] = 0.0
