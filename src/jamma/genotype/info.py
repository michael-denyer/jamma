"""Imputation INFO from quantised BGEN genotype probabilities.

INFO is GCTA's ``--info`` score, the IMPUTE2 information measure, computed
over the non-missing samples in the requested rows:

    e_j = 2*P11 + P12,  f_j = 4*P11 + P12,  theta = sum(e_j) / (2N)
    INFO = 1 - sum(f_j - e_j**2) / (2 N theta (1 - theta))

GCTA ``src/Geno.cpp`` ``Geno::getGenoDouble_bgen`` (lines 1319 to 1336 at
commit c6bbbee) accumulates the three sums as exact integers over the stored
B-bit numerators and divides once. This module does the same, in the same
floating-point operation order, so INFO equals GCTA's value bit for bit.
"""

from __future__ import annotations

import numpy as np

# GCTA sets INFO to 1 below this 2*theta*(1-theta).
_MONOMORPHIC_VARIANCE = 1e-50


def info_from_quantised(
    q11: np.ndarray,
    q12: np.ndarray,
    missing: np.ndarray,
    bit_depth: np.ndarray,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Return each variant's INFO over the non-missing samples of ``rows``.

    The sums are exact int64. A numerator is at most ``2**B - 1`` and a
    sample's P11 + P12 numerators sum to at most that too (the decoder
    rejects more), so ``e = 2*q11 + q12 <= 2*(2**16 - 1)`` at B <= 16 and
    ``e**2 <= 1.72e10``. The largest sum, E2, stays below the int64 limit
    of 9.22e18 for up to 5.3e8 samples. Nothing here checks N: a BGEN
    holding that many samples is past the int32 limits elsewhere first.

    Args:
        q11: ``(n_samples, k)`` uint16 P(11) numerators.
        q12: ``(n_samples, k)`` uint16 P(12) numerators.
        missing: ``(n_samples, k)`` bool, True for a missing sample.
        bit_depth: ``(k,)`` bit depth B of each variant, 1 to 16.
        rows: Sample row positions to include, or None for every row.

    Returns:
        float64 ``(k,)`` INFO. Not clamped, so it can be negative. A variant
        with no non-missing sample, or with 2*theta*(1-theta) < 1e-50, gets
        1.0, as GCTA gives a monomorphic variant.
    """
    if rows is not None:
        q11, q12, missing = q11[rows, :], q12[rows, :], missing[rows, :]
    present = ~missing
    n = np.count_nonzero(present, axis=0)
    e = 2 * q11.astype(np.int64) + q12
    e *= present
    dosage_sum = e.sum(axis=0)
    dosage2_sum = np.einsum("ij,ij->j", e, e)
    del e
    # GCTA's fij_sum accumulates 2*P11; its bracket adds it to the dosage sum.
    fij_sum = 2 * (q11 * present).sum(axis=0, dtype=np.int64)

    # Floating point from here follows Geno.cpp exactly, operation by operation.
    mask = ((1 << bit_depth.astype(np.int64)) - 1).astype(np.float64)
    valid_n = n.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        af = dosage_sum.astype(np.float64) / mask / (2 * n).astype(np.float64)
        std = 2.0 * af * (1.0 - af)
        dos2_fij_sum = (dosage_sum + fij_sum).astype(np.float64) / mask - (
            dosage2_sum.astype(np.float64) / (mask * mask)
        )
        info = 1.0 - dos2_fij_sum / (std * valid_n)
    info[(n == 0) | (std < _MONOMORPHIC_VARIANCE)] = 1.0
    return info
