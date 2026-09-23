/*
 * _lmm_stats.c — Pab to test statistic, test statistic to p-value.
 *
 * See _lmm_stats.h. Everything here runs once per SNP after the lambda
 * optimizer has finished; the continued fraction is the only non-trivial
 * piece.
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
 * Based on _betainc_cf in tests/reference/special.py / codeplea incbeta
 * (zlib license).
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
 * Matches the scalar betainc() oracle in tests/reference/special.py.
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
 * Matches _f_to_pvalue in stats.py.
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

/* Wald statistics. Returns 1 if the SNP is valid, 0 if degenerate
 * (Px_yy < 0 or P_xx <= 0), which sets beta = se = f_stat = NaN. */
int wald_stats(
    pab_terms_t p,
    int df,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    if (p.Px_yy < 0.0) {
        /* Schur complement went negative — degenerate SNP. Without this
         * guard, small negative Px_yy passes through variance_safe's fabs
         * branch and produces a fabricated positive SE with is_valid=1. */
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;
    }
    double Px_yy = replace_zero_p_yy(p.Px_yy);

    if (p.P_xx <= 0.0) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;
    }

    double beta = p.P_xy / p.P_xx;

    /* SE via JAMMA's corrected safe_sqrt (see GEMMA_DIVERGENCES.md section 1):
     * if |var| < 0.001, use fabs(var) instead of var to avoid sqrt of tiny
     * negative FP rounding artifacts. */
    double tau = (double)df / Px_yy;
    double variance_beta = 1.0 / (tau * p.P_xx);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    double se = sqrt(variance_safe);

    double f_stat = (p.P_yy - Px_yy) * tau;

    *beta_out   = beta;
    *se_out     = se;
    *f_stat_out = f_stat;

    /* Guard against non-finite results from pathological Px_yy / tau.
     * Without this, NaN f_stat passes is_valid=1 to f_to_pvalue, which
     * clamps NaN to 1e-10 and returns a bogus near-1 p-value. */
    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se))
        return 0;

    return 1;
}

/* Score statistics. Returns 1 if valid, 0 if degenerate (P_xx <= 0,
 * P_yy < 0, Px_yy < 0, or any output non-finite), which sets all three
 * outputs to NaN. */
int score_stats(
    pab_terms_t p,
    int n_samples,
    int df,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    if (p.P_xx <= 0.0 || p.P_yy < 0.0 || p.Px_yy < 0.0) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;
    }

    double P_yy = replace_zero_p_yy(p.P_yy);
    double Px_yy = replace_zero_p_yy(p.Px_yy);

    double beta = p.P_xy / p.P_xx;
    double tau = (double)df / Px_yy;
    double variance_beta = 1.0 / (tau * p.P_xx);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    double se = sqrt(variance_safe);

    double f_stat = (double)n_samples * (p.P_xy * p.P_xy) / (P_yy * p.P_xx);

    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se)) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;
    }

    *beta_out   = beta;
    *se_out     = se;
    *f_stat_out = f_stat;
    return 1;
}
