"""AssocResult and the batch Wald, Score and LRT statistics that fill it.

The vectorised forms of GEMMA's CalcRLWald, CalcRLScore and the LRT
p-value, applied to a chunk of SNPs at once. The scalar ports they are
checked against live in ``tests/reference/stats.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jamma.lmm.pab import _P_YY_MIN, build_index_table
from jamma.lmm.special import betainc_batch, chi2_sf_batch
from jamma.lmm.uab import batch_compute_pab_numpy


@dataclass
class AssocResult:
    """Association test result for a single SNP.

    Matches GEMMA's output format. Fields present depend on test type:
    - Wald (-lmm 1): REML logl_H1, l_remle, p_wald
    - LRT (-lmm 2): l_mle, p_lrt (no beta/se in GEMMA output, but kept for consistency)
    - Score (-lmm 3): p_score only (no per-SNP logl_H1/l_remle)
    - All (-lmm 4): All fields; logl_H1 is the alternative-model MLE
    """

    chr: str
    rs: str
    ps: int  # base position
    n_miss: int  # missing count for this SNP
    allele1: str  # minor allele
    allele0: str  # major allele
    af: float  # allele frequency
    beta: float
    se: float
    logl_H1: float | None = None  # REML in mode 1, MLE in mode 4
    l_remle: float | None = None  # Not present for Score-only
    p_wald: float | None = None  # Only for Wald/-lmm 1
    p_score: float | None = None  # Only for Score/-lmm 3
    l_mle: float | None = None  # MLE lambda (for LRT/-lmm 2)
    p_lrt: float | None = None  # LRT p-value (for LRT/-lmm 2)


def _beta_se_from_pab(
    P_XX: np.ndarray,
    P_XY: np.ndarray,
    Px_YY: np.ndarray,
    df: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute beta, SE, and validity mask from Pab projections.

    Shared by Wald and Score tests. Handles degenerate SNPs (P_XX <= 0)
    by setting beta/SE to NaN and GEMMA-compatible safe_sqrt for tiny
    variance values.

    Args:
        P_XX: Projected genotype variance per SNP (n_snps,).
        P_XY: Projected genotype-phenotype covariance per SNP (n_snps,).
        Px_YY: Projected phenotype variance at n_cvt+1 level (n_snps,).
        df: Degrees of freedom (n_samples - n_cvt - 1).

    Returns:
        Tuple of (beta, se, is_valid) each shape (n_snps,).
    """
    is_valid = P_XX > 0
    safe_P_XX = np.where(is_valid, P_XX, 1.0)

    beta = np.where(is_valid, P_XY / safe_P_XX, np.nan)
    tau = df / Px_YY
    variance_beta = np.where(is_valid, 1.0 / (tau * safe_P_XX), np.nan)
    # safe_sqrt: for |v| < 0.001, use abs(v) to avoid sqrt of tiny negative
    # values from FP rounding (matches GEMMA lmm.cpp safe_sqrt behaviour)
    variance_safe = np.where(
        np.abs(variance_beta) < 0.001,
        np.abs(variance_beta),
        variance_beta,
    )
    # np.where evaluates sqrt on all elements including NaN/negative variance_safe
    # from invalid SNPs; those results are discarded by the is_valid mask
    with np.errstate(invalid="ignore"):
        se = np.where(is_valid, np.sqrt(variance_safe), np.nan)

    return beta, se, is_valid


def _f_to_pvalue(f_stat: np.ndarray, df: int, is_valid: np.ndarray) -> np.ndarray:
    """Convert F-statistics to p-values via regularized incomplete beta.

    Shared by Wald and Score test computations. Uses algebraically exact
    complement_z = f_safe / (df + f_safe) to avoid cancellation near z = 1.

    Args:
        f_stat: F-statistics per SNP (n_snps,).
        df: Degrees of freedom (n_samples - n_cvt - 1).
        is_valid: Boolean mask of valid (non-degenerate) SNPs.

    Returns:
        P-values (n_snps,), NaN for invalid SNPs.
    """
    f_safe = np.maximum(f_stat, 1e-10)
    denom = df + f_safe
    z = np.clip(df / denom, 0.0, 1.0)
    complement_z = f_safe / denom  # algebraically exact 1-z, avoids cancellation
    a_arr = np.full_like(z, df / 2.0)
    b_arr = np.full_like(z, 0.5)
    p_val = betainc_batch(a_arr, b_arr, z, complement_z)
    p_val = np.where(f_stat <= 0, 1.0, p_val)
    return np.where(is_valid, p_val, np.nan)


def batch_calc_wald_stats_from_pab_numpy(
    n_cvt: int,
    Pab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wald test statistics from pre-computed Pab batch.

    The REML optimizers always return the Pab batch they evaluated at the
    optimal lambda, so callers pass it straight through instead of
    reconstructing Hi_eval and Pab from the optimized lambdas.

    Args:
        n_cvt: Number of covariates.
        Pab_batch: Pre-computed Pab (n_snps, n_cvt+2, n_index) at optimal lambdas.
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_walds) each shape (n_snps,).
    """
    table = build_index_table(n_cvt)
    idx_xx = table.idx_xx
    idx_xy = table.idx_xy
    idx_yy = table.idx_yy
    df = n_samples - n_cvt - 1

    P_XX = Pab_batch[:, n_cvt, idx_xx]
    P_XY = Pab_batch[:, n_cvt, idx_xy]
    P_YY = Pab_batch[:, n_cvt, idx_yy]
    Px_YY = Pab_batch[:, n_cvt + 1, idx_yy]

    # Clamp Px_YY to avoid division by near-zero for degenerate SNPs.
    Px_YY = np.where((Px_YY >= 0.0) & (Px_YY < _P_YY_MIN), _P_YY_MIN, Px_YY)

    beta, se, is_valid = _beta_se_from_pab(P_XX, P_XY, Px_YY, df)

    tau = df / Px_YY
    f_stat = (P_YY - Px_YY) * tau
    p_wald = _f_to_pvalue(f_stat, df, is_valid)

    return beta, se, p_wald


def batch_calc_score_stats_numpy(
    n_cvt: int,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Score test statistics for a batch of SNPs.

    Compute batch Score test statistics. Uses fixed null-model
    Hi_eval shared across all SNPs (cheaper than Wald — no per-SNP optimization).

    Score F-statistic uses n_samples (not df) in numerator and P_yy*P_xx
    denominator (not Px_yy). Matches GEMMA CalcRLScore exactly.

    Args:
        n_cvt: Number of covariates.
        Hi_eval_null: Null-model 1/(lambda_null*eval+1) vector (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_scores) each shape (n_snps,).
    """
    table = build_index_table(n_cvt)
    idx_xx = table.idx_xx
    idx_xy = table.idx_xy
    idx_yy = table.idx_yy
    df = n_samples - n_cvt - 1

    # Batch Pab with shared null Hi_eval
    Pab_batch = batch_compute_pab_numpy(n_cvt, Hi_eval_null, Uab_batch)

    # Score test: extract at level n_cvt (covariates only, NOT genotype)
    P_yy = Pab_batch[:, n_cvt, idx_yy]
    P_yy = np.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)
    P_xx = Pab_batch[:, n_cvt, idx_xx]
    P_xy = Pab_batch[:, n_cvt, idx_xy]

    # Px_yy for beta/se computation
    Px_yy = Pab_batch[:, n_cvt + 1, idx_yy]
    Px_yy = np.where((Px_yy >= 0.0) & (Px_yy < _P_YY_MIN), _P_YY_MIN, Px_yy)

    beta, se, is_valid = _beta_se_from_pab(P_xx, P_xy, Px_yy, df)

    # Score F-statistic: F = n * P_xy^2 / (P_yy * P_xx)
    safe_P_xx = np.where(is_valid, P_xx, 1.0)
    f_stat = n_samples * (P_xy * P_xy) / (P_yy * safe_P_xx)

    # p_score via Cephes betainc
    p_score = _f_to_pvalue(f_stat, df, is_valid)

    return beta, se, p_score


def _batch_lrt_pvalues_numpy(
    logls_mle: np.ndarray,
    logl_H0: float,
) -> np.ndarray:
    """Compute LRT p-values for a batch of SNPs.

    Compute LRT p-values for a batch of SNPs.
    LRT statistic = 2 * (logl_H1 - logl_H0), chi2 with df=1.

    Uses special.chi2_sf_batch (erfc-based, stdlib-only).

    Args:
        logls_mle: Per-SNP MLE log-likelihoods under alternative (n_snps,).
        logl_H0: Null model MLE log-likelihood (scalar).

    Returns:
        LRT p-values (n_snps,).
    """
    lrt_stats = 2.0 * (logls_mle - logl_H0)
    lrt_stats = np.maximum(lrt_stats, 0.0)
    p_lrts = chi2_sf_batch(lrt_stats)
    # Propagate NaN from MLE optimization failures (degenerate SNPs)
    p_lrts = np.where(np.isnan(logls_mle), np.nan, p_lrts)
    return p_lrts
