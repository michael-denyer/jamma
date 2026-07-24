/*
 * _lmm_stats.c — test statistic to p-value conversion.
 *
 * See _lmm_stats.h. The continued fraction is the only non-trivial piece
 * here, and it runs once per SNP rather than inside the lambda optimizer.
 */

#include "_lmm_stats.h"

/* Betainc continued fraction constants — matches special.py */
#define CF_TINY     1.0e-30
#define CF_STOP     1.0e-14
#define CF_MAX_ITER 200

/* -------------------------------------------------------------------------
 * betainc_cf
 *
 * Lentz continued fraction for regularized incomplete beta I_x(a, b).
 * Based on special.py _betainc_cf / codeplea incbeta (zlib license).
 * Differs: takes precomputed lbeta_ab to avoid per-call lgamma;
 * returns NaN (not exception) on non-convergence.
 * Caller guarantees x < (a+1)/(a+b+2) (symmetry threshold).
 * ------------------------------------------------------------------------- */
static double betainc_cf(double a, double b, double x, double lbeta_ab)
{
    double front = exp(log(x) * a + log(1.0 - x) * b - lbeta_ab) / a;

    double f = 1.0, c = 1.0, d = 0.0;

    for (int i = 0; i <= CF_MAX_ITER; i++) {
        int m = i / 2;
        double numerator;
        if (i == 0) {
            numerator = 1.0;
        } else if (i % 2 == 0) {
            double mf = (double)m;
            numerator = (mf * (b - mf) * x) /
                        ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        } else {
            double mf = (double)m;
            numerator = -((a + mf) * (a + b + mf) * x) /
                         ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        }

        d = 1.0 + numerator * d;
        if (fabs(d) < CF_TINY) d = CF_TINY;
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if (fabs(c) < CF_TINY) c = CF_TINY;

        double cd = c * d;
        f *= cd;

        if (fabs(1.0 - cd) < CF_STOP) {
            return front * (f - 1.0);
        }
    }
    return (double)NAN;  /* non-convergence */
}

/* -------------------------------------------------------------------------
 * betainc
 *
 * Regularized incomplete beta I_z(a, b) with symmetry relation.
 * Matches special.py betainc() scalar interface.
 *
 * complement_z is the algebraically exact 1-z, used for precision near z=1.
 * ------------------------------------------------------------------------- */
double betainc(
    double a,
    double b,
    double z,
    double complement_z,
    double lbeta_ab
)
{
    if (z <= 0.0) return 0.0;
    if (z >= 1.0) return 1.0;

    double threshold = (a + 1.0) / (a + b + 2.0);
    if (z <= threshold) {
        return betainc_cf(a, b, z, lbeta_ab);
    } else {
        return 1.0 - betainc_cf(b, a, complement_z, lbeta_ab);
    }
}

/* -------------------------------------------------------------------------
 * f_to_pvalue
 *
 * Convert F-statistic to p-value via regularized incomplete beta.
 * Matches _f_to_pvalue in likelihood_numpy.py.
 * Returns NaN if is_valid is false (degenerate SNP).
 * ------------------------------------------------------------------------- */
double f_to_pvalue(
    double f_stat,
    int df,
    int is_valid,
    double a,
    double b,
    double lbeta_ab
)
{
    if (!is_valid) return (double)NAN;
    if (f_stat <= 0.0) return 1.0;

    double f_safe = (f_stat > 1e-10) ? f_stat : 1e-10;
    double denom = (double)df + f_safe;
    double z = (double)df / denom;
    double complement_z = f_safe / denom;  /* algebraically exact 1-z */

    if (z < 0.0) z = 0.0;
    if (z > 1.0) z = 1.0;

    double p = betainc(a, b, z, complement_z, lbeta_ab);
    /* Clamp to [0, 1] — continued fraction FP accumulation can overshoot. */
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}
