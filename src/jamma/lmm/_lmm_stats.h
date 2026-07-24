/*
 * _lmm_stats.h — test statistic to p-value conversion.
 *
 * The C side of src/jamma/lmm/special.py: the regularized incomplete beta
 * behind the Wald F test, and the chi-squared survival function behind the
 * LRT. Pure double arithmetic with no CPython, NumPy, OpenMP or workspace
 * state, so unlike _lmm_support.h this header needs none of the
 * PY_ARRAY_UNIQUE_SYMBOL / NO_IMPORT_ARRAY handling.
 *
 * Every caller invokes these once per SNP, after the lambda optimizer has
 * finished, to turn a finished statistic into a p-value. None of them sits in
 * an inner loop, which is why the continued fraction can live out of line.
 */

#ifndef JAMMA_LMM_STATS_H
#define JAMMA_LMM_STATS_H

#include <math.h>

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

#endif /* JAMMA_LMM_STATS_H */
