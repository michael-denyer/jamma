/*
 * _lmm_tests.h — Wald and score statistics from a populated Pab array.
 *
 * The C side of src/jamma/lmm/stats.py. Given a Pab array the likelihood
 * machinery has already filled in, these turn it into an effect size, a
 * standard error and an F statistic. They read the Pab layout and nothing
 * else: no CPython, no NumPy, no OpenMP, no workspace state.
 *
 * Every caller invokes them once per SNP, in the output block after the lambda
 * optimizer has converged, not inside the optimizer's inner loop. That is what
 * makes it safe for them to live out of line; the kernels the optimizer calls
 * ~70 times per SNP are a different problem and stay inline.
 *
 * Each returns 1 for a usable SNP and 0 for a degenerate one, and callers test
 * that return rather than isnan(beta): relying on NaN propagating through
 * comparisons is the more fragile check.
 */

#ifndef JAMMA_LMM_TESTS_H
#define JAMMA_LMM_TESTS_H

#include "_lmm_types.h"

/* ---------------------------------------------------------------------------
 * n_cvt = 1. Pab is the fixed 3x6 layout.
 * ------------------------------------------------------------------------- */

/* Wald: F = (P_yy - Px_yy) / Px_yy * df, beta and se from level 2. */
int wald_from_pab(
    const double pab[3][6],
    int df,
    double *beta_out, double *se_out, double *f_stat_out
);

/* Score: F = n_samples * P_xy^2 / (P_yy * P_xx), so it uses n_samples rather
 * than df and needs no per-SNP lambda. beta and se still come from level 2. */
int score_from_pab(
    const double pab[3][6],
    int n_samples,
    int df,
    double *beta_out, double *se_out, double *f_stat_out
);

/* ---------------------------------------------------------------------------
 * General n_cvt. Pab is flat and indexed through the table.
 * ------------------------------------------------------------------------- */

int score_from_pab_general(
    const double *pab,
    const pab_table_t *t,
    int n_samples,
    double *beta_out, double *se_out, double *f_stat_out
);

/* -------------------------------------------------------------------------
 * wald_from_pab_general — Extract Wald stats from general-n_cvt Pab.
 *
 * P_XX = Pab[n_cvt, idx_xx], P_XY = Pab[n_cvt, idx_xy],
 * P_YY = Pab[n_cvt, idx_yy] (pre-genotype-projection),
 * Px_YY = Pab[n_cvt+1, idx_yy] (fully projected).
 * Same Wald formula as existing wald_from_pab.
 * Returns 1 if valid, 0 if degenerate.
 * ------------------------------------------------------------------------- */
int wald_from_pab_general(
    const double *pab,
    const pab_table_t *t,
    double *beta_out, double *se_out, double *f_stat_out
);

#endif /* JAMMA_LMM_TESTS_H */
