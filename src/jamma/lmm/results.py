"""Result building functions for LMM association tests.

Constructs AssocResult objects from computed statistics for each
test mode (Wald, Score, LRT, All).
"""

import numpy as np

from jamma.lmm.stats import AssocResult


def _snp_metadata(snp_info: dict, af: float, n_miss: int) -> dict:
    """Extract common SNP metadata fields for AssocResult construction.

    Args:
        snp_info: SNP metadata dict with keys: chr, rs, pos/ps, a1/allele1, a0/allele0.
        af: Allele frequency of counted allele (BIM A1), can be > 0.5.
        n_miss: Missing genotype count.

    Returns:
        Dict of shared AssocResult fields.
    """
    return {
        "chr": snp_info["chr"],
        "rs": snp_info["rs"],
        "ps": snp_info.get("pos", snp_info.get("ps", 0)),
        "n_miss": n_miss,
        "allele1": snp_info.get("a1", snp_info.get("allele1", "")),
        "allele0": snp_info.get("a0", snp_info.get("allele0", "")),
        "af": af,
    }


def _build_results_wald(
    snp_indices: np.ndarray,
    snp_stats: list[tuple[float, int]],
    snp_info: list,
    best_lambdas_np: np.ndarray,
    best_logls_np: np.ndarray,
    betas_np: np.ndarray,
    ses_np: np.ndarray,
    p_walds_np: np.ndarray,
) -> list[AssocResult]:
    """Build AssocResult objects for Wald test mode.

    Args:
        snp_indices: Indices of SNPs that passed filtering.
        snp_stats: List of (af, n_miss) tuples for each filtered SNP.
        snp_info: Full SNP metadata list.
        best_lambdas_np: Optimal REML lambda values.
        best_logls_np: Log-likelihoods at optimal lambda.
        betas_np: Effect sizes.
        ses_np: Standard errors.
        p_walds_np: Wald test p-values.

    Returns:
        List of AssocResult objects.
    """
    results = []
    for j, snp_idx in enumerate(snp_indices):
        af, n_miss = snp_stats[j]
        meta = _snp_metadata(snp_info[snp_idx], af, n_miss)
        results.append(
            AssocResult(
                **meta,
                beta=float(betas_np[j]),
                se=float(ses_np[j]),
                logl_H1=float(best_logls_np[j]),
                l_remle=float(best_lambdas_np[j]),
                p_wald=float(p_walds_np[j]),
            )
        )
    return results


def _build_results_score(
    snp_indices: np.ndarray,
    snp_stats: list[tuple[float, int]],
    snp_info: list,
    betas_np: np.ndarray,
    ses_np: np.ndarray,
    p_scores_np: np.ndarray,
) -> list[AssocResult]:
    """Build AssocResult objects for Score test mode.

    Args:
        snp_indices: Indices of SNPs that passed filtering.
        snp_stats: List of (af, n_miss) tuples for each filtered SNP.
        snp_info: Full SNP metadata list.
        betas_np: Effect sizes (informational only).
        ses_np: Standard errors (informational only).
        p_scores_np: Score test p-values.

    Returns:
        List of AssocResult objects with p_score set.
    """
    results = []
    for j, snp_idx in enumerate(snp_indices):
        af, n_miss = snp_stats[j]
        meta = _snp_metadata(snp_info[snp_idx], af, n_miss)
        results.append(
            AssocResult(
                **meta,
                beta=float(betas_np[j]),
                se=float(ses_np[j]),
                p_score=float(p_scores_np[j]),
            )
        )
    return results


def _build_results_lrt(
    snp_indices: np.ndarray,
    snp_stats: list[tuple[float, int]],
    snp_info: list,
    lambdas_mle_np: np.ndarray,
    p_lrts_np: np.ndarray,
) -> list[AssocResult]:
    """Build AssocResult objects for LRT mode.

    LRT does not compute beta/se (matching GEMMA -lmm 2 output format).

    Args:
        snp_indices: Indices of SNPs that passed filtering.
        snp_stats: List of (af, n_miss) tuples for each filtered SNP.
        snp_info: Full SNP metadata list.
        lambdas_mle_np: MLE lambda values per SNP.
        p_lrts_np: LRT p-values.

    Returns:
        List of AssocResult objects with l_mle and p_lrt set.
    """
    results = []
    for j, snp_idx in enumerate(snp_indices):
        af, n_miss = snp_stats[j]
        meta = _snp_metadata(snp_info[snp_idx], af, n_miss)
        results.append(
            AssocResult(
                **meta,
                beta=float("nan"),
                se=float("nan"),
                l_mle=float(lambdas_mle_np[j]),
                p_lrt=float(p_lrts_np[j]),
            )
        )
    return results


def _build_results_all(
    snp_indices: np.ndarray,
    snp_stats: list[tuple[float, int]],
    snp_info: list,
    best_lambdas_np: np.ndarray,
    best_logls_np: np.ndarray,
    betas_np: np.ndarray,
    ses_np: np.ndarray,
    p_walds_np: np.ndarray,
    lambdas_mle_np: np.ndarray,
    p_lrts_np: np.ndarray,
    p_scores_np: np.ndarray,
) -> list[AssocResult]:
    """Build AssocResult objects for All-tests mode (Wald + LRT + Score).

    Uses Wald beta/se (not Score beta/se) and REML logl_H1 (not MLE),
    matching the NumPy runner output at __init__.py:331-347.

    Args:
        snp_indices: Indices of SNPs that passed filtering.
        snp_stats: List of (af, n_miss) tuples for each filtered SNP.
        snp_info: Full SNP metadata list.
        best_lambdas_np: Optimal REML lambda values (Wald).
        best_logls_np: REML log-likelihoods at optimal lambda.
        betas_np: Effect sizes from Wald test.
        ses_np: Standard errors from Wald test.
        p_walds_np: Wald test p-values.
        lambdas_mle_np: MLE lambda values per SNP (LRT).
        p_lrts_np: LRT p-values.
        p_scores_np: Score test p-values.

    Returns:
        List of AssocResult objects with all fields populated.
    """
    results = []
    for j, snp_idx in enumerate(snp_indices):
        af, n_miss = snp_stats[j]
        meta = _snp_metadata(snp_info[snp_idx], af, n_miss)
        results.append(
            AssocResult(
                **meta,
                beta=float(betas_np[j]),
                se=float(ses_np[j]),
                logl_H1=float(best_logls_np[j]),
                l_remle=float(best_lambdas_np[j]),
                l_mle=float(lambdas_mle_np[j]),
                p_wald=float(p_walds_np[j]),
                p_lrt=float(p_lrts_np[j]),
                p_score=float(p_scores_np[j]),
            )
        )
    return results
