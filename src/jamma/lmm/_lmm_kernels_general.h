/*
 * _lmm_kernels_general.h — the arbitrary-n_cvt numerical kernels.
 *
 * The Pab recursion, likelihood evaluations, statistic extraction and lambda
 * optimizers for the general (arbitrary covariate count) path. These are the
 * table-driven counterparts of the single-covariate kernels that remain in
 * _lmm_accel.c, and the two sets are disjoint: no general kernel calls an
 * ncvt1 kernel or the reverse, which is what makes this a translation-unit
 * boundary rather than an arbitrary cut.
 *
 * Callers are the general entry points in _lmm_accel.c — the batch Score and
 * LRT paths and the fused-Uab chunk functions. They keep the workspace structs
 * and all CPython marshalling; only the arithmetic moved.
 *
 * Pure double arithmetic over the Pab layout: no CPython, no NumPy, no
 * OpenMP, no workspace state. It needs only the table shape from
 * _lmm_types.h, so unlike _lmm_support.h it carries no import_array()
 * handling.
 */

#ifndef JAMMA_LMM_KERNELS_GENERAL_H
#define JAMMA_LMM_KERNELS_GENERAL_H

#include "_lmm_types.h"

#include <math.h>

/* -------------------------------------------------------------------------
 * calc_pab_general — Table-driven Pab recursion for arbitrary n_cvt.
 *
 * Row 0 from row0 array (dot product sums), rows 1..n_rows-1 from entries.
 * Output in pab[n_rows * n_index], row-major.
 * ------------------------------------------------------------------------- */
void calc_pab_general(
    const double *row0,
    const pab_table_t *t,
    double *pab
);

/* -------------------------------------------------------------------------
 * reml_finish_general — REML tail for general n_cvt.
 *
 * logdet_pab from logdet_diag entries, P_yy guard, return full REML formula
 * including logdet_h.
 * ------------------------------------------------------------------------- */
double reml_finish_general(
    const double *pab,
    const pab_table_t *t,
    double logdet_h,
    double logdet_iab,
    double reml_const
);

/* -------------------------------------------------------------------------
 * reml_logl_general_cached — REML using cached grid hi_eval + invariant sums.
 *
 * For cached grid points: invariant sums already computed, just compute
 * varying dot products, reconstruct row0, calc_pab, reml_finish.
 * ------------------------------------------------------------------------- */
double reml_logl_general_cached(
    const double *inv_sums_cached,
    const double *uab_var,
    const double *hi_eval,
    int n_samples,
    double logdet_h,
    double logdet_iab,
    double reml_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
);

/* -------------------------------------------------------------------------
 * reml_logl_general_fresh — Full REML evaluation for a specific lambda.
 *
 * Computes hi_eval + logdet_h + all dot products in single n_samples pass
 * (fused loop), then calc_pab + reml_finish.
 * Used during golden section refinement where lambda is SNP-specific.
 * ------------------------------------------------------------------------- */
double reml_logl_general_fresh(
    const double *uab_inv,
    const double *uab_var,
    const double *eigenvalues,
    int n_samples,
    double lambda,
    double logdet_iab,
    double reml_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
);


/* -------------------------------------------------------------------------
 * golden_section_lambda_general — Grid + golden section for general n_cvt.
 *
 * Mirrors the coarse-grid plus refine_lambda_ncvt1_split pair. Grid phase uses
 * precomputed hi_eval + invariant sums; refinement uses fresh evaluation.
 * At optimal lambda, computes full Pab and returns it + Wald stats.
 * ------------------------------------------------------------------------- */
double golden_section_lambda_general(
    const double *uab_inv,
    const double *uab_var,
    const double *eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const double *inv_sums_grid,    /* (n_grid, n_inv) */
    double log_l_min, double step,
    int n_grid, int n_refine,
    double logdet_iab,
    double reml_const,
    const pab_table_t *t,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
);


/* -------------------------------------------------------------------------
 * mle_logl_general — MLE log-likelihood for one SNP at one lambda (general n_cvt).
 *
 * MLE formula: -0.5 * n * log(P_yy_full) - 0.5 * logdet_h + mle_const
 * where P_yy_full is at level n_cvt+1 (fully projected).
 *
 * Uses full Uab row (n_samples * n_index) in AoS layout.
 * ------------------------------------------------------------------------- */
double mle_logl_general(
    const double *uab_snp,     /* (n_samples, n_index) row-major */
    const double *eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
);

/* -------------------------------------------------------------------------
 * mle_logl_general_cached — MLE using cached hi_eval for coarse grid search.
 * ------------------------------------------------------------------------- */
double mle_logl_general_cached(
    const double *uab_snp,
    const double *cached_hi_eval,
    double cached_logdet_h,
    int n_samples,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
);

/* -------------------------------------------------------------------------
 * golden_section_lambda_mle_general — Grid + golden section for MLE (general n_cvt).
 *
 * Mirrors golden_section_lambda_mle_ncvt1 but uses mle_logl_general.
 * Returns optimal lambda; writes logl to *logl_out.
 * ------------------------------------------------------------------------- */
double golden_section_lambda_mle_general(
    const double *uab_snp,
    const double *eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    double mle_const,
    const pab_table_t *t,
    double *logl_out,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
);

/* -------------------------------------------------------------------------
 * logdet_from_row0 — compute logdet(Iab) from identity dot products.
 *
 * Encapsulates the identity Pab prepass: calls calc_pab_general into the
 * caller-provided scratch buffer, then extracts diagonal entries for logdet.
 * Replaces the inline copies of the same pattern in fused general Wald and
 * fused general mode-4.
 *
 * row0:        n_index identity-weighted dot products
 * t:           pab_table_t with logdet_diag_rows/cols
 * n_cvt:       number of covariates
 * pab_scratch: caller-provided buffer of at least MAX_PAB_SIZE doubles
 *
 * Returns logdet value, or NAN if any diagonal <= 0.
 * ------------------------------------------------------------------------- */
static inline double logdet_from_row0(
    const double *row0,
    const pab_table_t *t,
    int n_cvt,
    double *pab_scratch)
{
    calc_pab_general(row0, t, pab_scratch);

    int ni = t->n_index;
    double logdet = 0.0;
    for (int d = 0; d < n_cvt + 1; d++) {
        double val = pab_scratch[t->logdet_diag_rows[d] * ni
                                 + t->logdet_diag_cols[d]];
        if (val <= 0.0) return (double)NAN;
        logdet += log(val);
    }
    return logdet;
}

#endif /* JAMMA_LMM_KERNELS_GENERAL_H */
