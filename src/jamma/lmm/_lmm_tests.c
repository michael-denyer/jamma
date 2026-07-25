/*
 * _lmm_tests.c — Wald and score statistics from a populated Pab array.
 *
 * See _lmm_tests.h. Every caller runs these once per SNP, after the lambda
 * optimizer has finished, which is why they can live out of line.
 */

#include "_lmm_tests.h"

#include <math.h>

/* Wald statistics from a populated pab array.
 * Shared by golden_section_lambda_ncvt1 and golden_section_lambda_ncvt1_split.
 *
 * Returns 1 if the SNP is valid (P_XX > 0), 0 if degenerate (P_XX <= 0).
 * Degenerate SNPs get beta = se = f_stat = NaN.
 *
 * The return value (not isnan(beta)) is used for validity checks — this is
 * more robust than relying on NaN propagation through comparisons. */
int wald_from_pab(
    const double pab[3][6],
    int df,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    double P_XX  = pab[1][3];
    double P_XY  = pab[1][4];
    double P_YY  = pab[1][5];
    double Px_YY = pab[2][5];

    if (Px_YY < 0.0) {
        /* Schur complement went negative — degenerate SNP. Without this
         * guard, small negative Px_YY passes through variance_safe's fabs
         * branch and produces a fabricated positive SE with is_valid=1. */
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;  /* degenerate */
    }
    if (Px_YY < P_YY_MIN) {
        Px_YY = P_YY_MIN;
    }

    if (P_XX <= 0.0) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;  /* degenerate */
    }

    double beta = P_XY / P_XX;

    /* SE via JAMMA's corrected safe_sqrt (see GEMMA_DIVERGENCES.md section 1):
     * if |var| < 0.001, use fabs(var) instead of var to avoid sqrt of tiny
     * negative FP rounding artifacts. */
    double tau = (double)df / Px_YY;
    double variance_beta = 1.0 / (tau * P_XX);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    double se = sqrt(variance_safe);

    double f_stat = (P_YY - Px_YY) * tau;

    *beta_out   = beta;
    *se_out     = se;
    *f_stat_out = f_stat;

    /* Guard against non-finite results from pathological Px_YY / tau.
     * Without this, NaN f_stat passes is_valid=1 to f_to_pvalue, which
     * clamps NaN to 1e-10 and returns a bogus near-1 p-value. */
    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se))
        return 0;

    return 1;  /* valid */
}

/* -------------------------------------------------------------------------
 * score_from_pab
 *
 * Score test statistics from a populated pab array (n_cvt=1).
 *
 * Reads from two pab levels:
 *   - Level 1 (pab[1]): P_yy, P_xx, P_xy for Score F-statistic
 *   - Level 2 (pab[2]): Px_yy for beta/SE computation (same as Wald)
 *
 * Key differences from Wald:
 *   - F = n_samples * P_xy^2 / (P_yy * P_xx) — uses n_samples, not df
 *   - No per-SNP lambda optimization (uses null-model Hi_eval)
 *
 * Returns 1 if valid, 0 if degenerate (P_xx <= 0 or P_yy < 0 or Px_yy < 0
 * or any output non-finite).
 * ------------------------------------------------------------------------- */
int score_from_pab(
    const double pab[3][6],
    int n_samples,
    int df,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    /* Score extracts at level n_cvt=1 (row 1), NOT n_cvt+1=2 */
    double P_yy = pab[1][5];
    double P_xx = pab[1][3];
    double P_xy = pab[1][4];
    /* Px_yy at level n_cvt+1=2 for beta/se computation */
    double Px_yy = pab[2][5];

    if (P_xx <= 0.0 || P_yy < 0.0 || Px_yy < 0.0) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;  /* degenerate */
    }

    /* Clamp P_yy for F-stat denominator */
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;
    /* Clamp Px_yy for beta/se */
    if (Px_yy < P_YY_MIN) Px_yy = P_YY_MIN;

    *beta_out = P_xy / P_xx;

    double tau = (double)df / Px_yy;
    double variance_beta = 1.0 / (tau * P_xx);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    *se_out = sqrt(variance_safe);

    /* Score F-statistic: uses n_samples (not df) in numerator */
    *f_stat_out = (double)n_samples * (P_xy * P_xy) / (P_yy * P_xx);

    if (!isfinite(*f_stat_out) || !isfinite(*beta_out) || !isfinite(*se_out)) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;
    }

    return 1;  /* valid */
}

/* -------------------------------------------------------------------------
 * score_from_pab_general — Score statistics from general-n_cvt Pab.
 *
 * Score differs from Wald:
 *   - F = n_samples * P_xy^2 / (P_yy * P_xx)  [not (P_yy - Px_yy) * tau]
 *   - Degenerate guard checks P_XX <= 0 || P_YY < 0 || Px_YY < 0
 *   - Px_yy at level n_cvt+1 used only for beta/se, not F-stat
 *
 * Returns 1 if valid, 0 if degenerate.
 * ------------------------------------------------------------------------- */
int score_from_pab_general(
    const double *pab,
    const pab_table_t *t,
    int n_samples,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    int ni = t->n_index;
    int df = t->df;
    int nc = t->n_cvt;

    /* Score: extract at level n_cvt (row nc), NOT n_cvt+1 */
    double P_XX  = pab[nc * ni + t->idx_xx];
    double P_XY  = pab[nc * ni + t->idx_xy];
    double P_YY  = pab[nc * ni + t->idx_yy];
    /* Px_yy at level n_cvt+1 for beta/se */
    double Px_YY = pab[(nc + 1) * ni + t->idx_yy];

    if (P_XX <= 0.0 || P_YY < 0.0 || Px_YY < 0.0) {
        *beta_out = *se_out = *f_stat_out = (double)NAN;
        return 0;
    }

    /* Clamp for numerical stability */
    if (P_YY < P_YY_MIN) P_YY = P_YY_MIN;
    if (Px_YY < P_YY_MIN) Px_YY = P_YY_MIN;

    double beta = P_XY / P_XX;
    double tau = (double)df / Px_YY;
    double variance_beta = 1.0 / (tau * P_XX);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    double se = sqrt(variance_safe);

    /* Score F-statistic: uses n_samples (not df) in numerator */
    double f_stat = (double)n_samples * (P_XY * P_XY) / (P_YY * P_XX);

    *beta_out   = beta;
    *se_out     = se;
    *f_stat_out = f_stat;

    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se)) {
        *beta_out = *se_out = *f_stat_out = (double)NAN;
        return 0;
    }

    return 1;
}
