/*
 * _lmm_stats.h — from a populated Pab array to a test statistic, and from a
 * test statistic to a p-value.
 *
 * The C side of src/jamma/lmm/stats.py and special.py. The first half reads
 * the Pab layout and turns it into an effect size, a standard error and an F
 * statistic; the second half is the regularized incomplete beta behind the
 * Wald F test and the chi-squared survival function behind the LRT. Pure
 * double arithmetic with no CPython, NumPy, OpenMP or workspace state, so
 * unlike _lmm_support.h this header needs none of the
 * PY_ARRAY_UNIQUE_SYMBOL / NO_IMPORT_ARRAY handling.
 *
 * Every caller invokes these once per SNP, in the output block after the
 * lambda optimizer has converged, not inside the optimizer's inner loop. None
 * of them sits in a hot loop, which is why they can live out of line; the
 * kernels the optimizer calls ~70 times per SNP are a different problem and
 * stay inline.
 *
 * The Pab functions each return 1 for a usable SNP and 0 for a degenerate
 * one, and callers test that return rather than isnan(beta): relying on NaN
 * propagating through comparisons is the more fragile check.
 */

#ifndef JAMMA_LMM_STATS_H
#define JAMMA_LMM_STATS_H

#include <math.h>

#include "_lmm_types.h"

/* ---------------------------------------------------------------------------
 * Regularized incomplete beta I_z(a, b), with the symmetry relation applied.
 * Matches special.py betainc() scalar interface.
 *
 * complement_z is the algebraically exact 1-z, kept separate for precision
 * near z=1. lbeta_ab is a precomputed lgamma term, hoisted by callers so the
 * per-SNP path does no lgamma work. Returns NaN if the continued fraction
 * fails to converge; warn_betainc_convergence() reports that to Python.
 * ------------------------------------------------------------------------- */
double betainc(double a, double b, double z, double complement_z,
               double lbeta_ab);

/* ---------------------------------------------------------------------------
 * F statistic to p-value via the regularized incomplete beta.
 * Matches _f_to_pvalue in likelihood_numpy.py.
 * Returns NaN when is_valid is false, which is how a degenerate SNP arrives.
 * ------------------------------------------------------------------------- */
double f_to_pvalue(double f_stat, int df, int is_valid, double a, double b,
                   double lbeta_ab);

/* ---------------------------------------------------------------------------
 * Chi-squared survival function for df=1: P(X > x) = erfc(sqrt(x/2)).
 * Matches special.py chi2_sf exactly.
 *
 * Inline in the header rather than compiled into _lmm_stats.c: it is four
 * branches over a libm call, so an out-of-line version would cost a call to
 * save nothing. Keeping it inline also makes the codegen in every caller
 * identical to what it was before this header existed.
 * ------------------------------------------------------------------------- */
static inline double chi2_sf_c(double x)
{
    if (isnan(x)) return x;          /* NaN propagation */
    if (x <= 0.0) return 1.0;
    if (!isfinite(x)) return 0.0;   /* +inf → 0 */
    return erfc(sqrt(x / 2.0));
}


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

#endif /* JAMMA_LMM_STATS_H */
