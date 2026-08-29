/*
 * _lmm_kernels_ncvt1.c — see _lmm_kernels_ncvt1.h.
 *
 * Pure arithmetic. Nothing here touches CPython, so it needs none of
 * _lmm_support.h's import_array() handling.
 */

#include "_lmm_kernels_ncvt1.h"

/* wald_from_pab and score_from_pab live with the other statistic extractors
 * in _lmm_stats.c; the optimizers call them once per SNP on convergence. */
#include "_lmm_stats.h"

#include <math.h>
#include <string.h>


static inline double reml_finish(
    const double pab[3][6],
    double logdet_h,
    double logdet_iab,
    int df,
    double reml_const
)
{
    double logdet_pab = 0.0;
    if (pab[0][0] > 0.0) logdet_pab += log(pab[0][0]);
    if (pab[1][3] > 0.0) logdet_pab += log(pab[1][3]);
    double logdet_hiw = logdet_pab - logdet_iab;

    double P_yy = pab[2][5];
    if (P_yy < 0.0) {
        P_yy = (double)NAN;
    } else if (P_yy < P_YY_MIN) {
        P_yy = P_YY_MIN;
    }

    return reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy);
}

static inline void calc_pab_ncvt1_cached_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict cached_hi_eval,
    const grid_invariant_t *ginv,
    int n_samples,
    double pab[3][6]
)
{
    double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
    #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
    for (int i = 0; i < n_samples; i++) {
        double h = cached_hi_eval[i];
        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];
    }

    calc_pab_ncvt1_split(
        ginv->s_ww, s_wx, ginv->s_wy,
        s_xx, s_xy, ginv->s_yy, pab
    );
}


static inline double reml_finish_cached_split(
    const double pab[3][6],
    double cached_logdet_h,
    double logdet_iab,
    const grid_invariant_t *ginv,
    int df,
    double reml_const
)
{
    double logdet_pab = ginv->log_s_ww;
    if (pab[1][3] > 0.0) logdet_pab += log(pab[1][3]);
    double logdet_hiw = logdet_pab - logdet_iab;

    double P_yy = pab[2][5];
    if (P_yy < 0.0) {
        P_yy = (double)NAN;
    } else if (P_yy < P_YY_MIN) {
        P_yy = P_YY_MIN;
    }

    return reml_const - 0.5 * cached_logdet_h - 0.5 * logdet_hiw
           - 0.5 * df * log(P_yy);
}


/* -------------------------------------------------------------------------
 * reml_logl_ncvt1_split
 *
 * Refinement path: fused hi_eval computation + all 6 dot products in
 * a single pass over n_samples. Eliminates the separate calc_pab call.
 *
 * Used during golden section where lambda is SNP-specific.
 *
 * SoA layout: varying and invariant columns are contiguous (stride-1),
 * enabling SIMD vectorized loads instead of stride-3 gathers.
 *
 * pab_out is NULL during refinement iteration; the final evaluation passes
 * its own buffer to read the Pab this call computed for Wald extraction,
 * without a second n_samples pass.
 * ------------------------------------------------------------------------- */
double reml_logl_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    double lambda,
    double reml_const,
    double (*pab_out)[6]
)
{
    int df = n_samples - 2;

    /* Fused: compute hi_eval + logdet_h + all 6 dot products in single pass.
     * v = lambda * eval[i] + 1.0 is always > 1.0 (eigenvalues >= 0, lambda > 0),
     * so fabs() is unnecessary and would block SIMD vectorization of log().
     * SoA layout gives stride-1 access for all 6 columns — enables contiguous
     * SIMD loads (vmovupd) instead of stride-3 gather instructions. */
    double logdet_h = 0.0;
    double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

    #pragma omp simd reduction(+:logdet_h,s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        logdet_h += log(v);

        /* Varying (per-SNP, from DRAM) — stride-1 */
        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];

        /* Invariant (shared, from L2 cache) — stride-1 */
        s_ww += h * inv_ww[i];
        s_wy += h * inv_wy[i];
        s_yy += h * inv_yy[i];
    }

    /* Pab from sums */
    double pab[3][6];
    calc_pab_ncvt1_split(s_ww, s_wx, s_wy, s_xx, s_xy, s_yy, pab);

    if (pab_out) memcpy(pab_out, pab, sizeof(pab));

    return reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
}



/* -------------------------------------------------------------------------
 * refine_lambda_ncvt1_split
 *
 * Golden section refinement using a caller-selected split-Uab coarse bracket.
 *
 * SoA layout: var_wx/xx/xy and inv_ww/wy/yy are contiguous (stride-1).
 *
 * The final evaluation fuses REML logl + Wald stats in a single pass,
 * eliminating a redundant n_samples traversal per SNP.
 * ------------------------------------------------------------------------- */
double refine_lambda_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    const double *lambda_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    int best_idx,
    int df, double reml_const,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    const double phi = 0.6180339887498949;

    /* Every grid point produced NaN — fully degenerate SNP. */
    if (best_idx < 0) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return lambda_grid[0];
    }

    /* Bracket around best grid point */
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement (fused single-pass) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       logdet_iab, n_samples, exp(c),
                                       reml_const, NULL);
    double fd = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       logdet_iab, n_samples, exp(d),
                                       reml_const, NULL);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                        inv_ww, inv_wy, inv_yy, eigenvalues,
                                        logdet_iab, n_samples, exp(c),
                                        reml_const, NULL);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                        inv_ww, inv_wy, inv_yy, eigenvalues,
                                        logdet_iab, n_samples, exp(d),
                                        reml_const, NULL);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);

    /* Final evaluation: reml_logl_ncvt1_split fills pab as a side effect, so
     * the Wald extraction below reads the same Pab the logl was computed
     * from without a second n_samples pass. */
    double pab[3][6];
    *logl_out = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       logdet_iab, n_samples, lambda_opt,
                                       reml_const, pab);
    *is_valid_out = wald_from_pab(pab, df, beta_out, se_out, f_stat_out);

    return lambda_opt;
}


/* -------------------------------------------------------------------------
 * mle_finish
 *
 * MLE log-likelihood tail (simpler than REML — no logdet_hiw, no Iab).
 * logl = mle_const - 0.5 * logdet_h - 0.5 * n * log(P_yy)
 *
 * P_yy at level nc_total = n_cvt+1 = 2 (pab[2][5]) — same index as REML.
 * Uses n_samples (not df).
 * ------------------------------------------------------------------------- */
static inline double mle_finish(
    const double pab[3][6],
    double logdet_h,
    int n_samples,
    double mle_const
)
{
    double P_yy = pab[2][5];
    if (P_yy < 0.0) return (double)NAN;
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;
    return mle_const - 0.5 * logdet_h - 0.5 * n_samples * log(P_yy);
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_cached_split
 *
 * MLE log-likelihood from SoA split data using cached grid hi_eval.
 * Pattern-matches reml_logl_ncvt1_cached_split but:
 *   - No logdet_iab / logdet_hiw terms
 *   - Uses n_samples (not df)
 *   - Uses mle_const (not reml_const)
 * ------------------------------------------------------------------------- */
double mle_logl_ncvt1_cached_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    const grid_invariant_t *ginv,
    int n_samples,
    double mle_const
)
{
    double pab[3][6];
    calc_pab_ncvt1_cached_split(
        var_wx, var_xx, var_xy, cached_hi_eval, ginv, n_samples, pab
    );
    return mle_finish(pab, cached_logdet_h, n_samples, mle_const);
}


int coarse_grid_mle_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    int n_samples,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const grid_invariant_t *grid_inv,
    int n_grid,
    double mle_const
)
{
    double best_logl = REML_SENTINEL;
    int best_idx = -1;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_ncvt1_cached_split(
            var_wx, var_xx, var_xy,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g], &grid_inv[g], n_samples, mle_const
        );
        if (!isnan(logl) && logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }
    return best_idx;
}


void coarse_grid_mode4_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    int n_samples,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const grid_invariant_t *grid_inv,
    int n_grid,
    double logdet_iab,
    int df,
    double reml_const,
    double mle_const,
    int *best_reml_idx,
    int *best_mle_idx
)
{
    double best_reml = REML_SENTINEL;
    double best_mle = REML_SENTINEL;
    *best_reml_idx = -1;
    *best_mle_idx = -1;

    for (int g = 0; g < n_grid; g++) {
        const grid_invariant_t *ginv = &grid_inv[g];
        double pab[3][6];
        calc_pab_ncvt1_cached_split(
            var_wx, var_xx, var_xy,
            hi_eval_grid + (size_t)g * n_samples,
            ginv, n_samples, pab
        );

        double reml_logl = reml_finish_cached_split(
            pab, logdet_h_grid[g], logdet_iab, ginv, df, reml_const
        );
        double mle_logl = mle_finish(
            pab, logdet_h_grid[g], n_samples, mle_const
        );

        if (!isnan(reml_logl) && reml_logl > best_reml) {
            best_reml = reml_logl;
            *best_reml_idx = g;
        }
        if (!isnan(mle_logl) && mle_logl > best_mle) {
            best_mle = mle_logl;
            *best_mle_idx = g;
        }
    }
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_split
 *
 * MLE log-likelihood from SoA split data at an arbitrary lambda.
 * Used during golden section refinement. Computes hi_eval from scratch,
 * accumulates all 6 dot products (3 invariant + 3 varying), builds Pab.
 *
 * hi_eval is a caller-provided scratch buffer of size (n_samples,).
 * ------------------------------------------------------------------------- */
double mle_logl_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    double * restrict hi_eval
)
{
    double logdet_h = 0.0;
    double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

    #pragma omp simd reduction(+:logdet_h,s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        hi_eval[i] = h;
        logdet_h += log(v);

        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];

        s_ww += h * inv_ww[i];
        s_wy += h * inv_wy[i];
        s_yy += h * inv_yy[i];
    }

    double pab[3][6];
    calc_pab_ncvt1_split(s_ww, s_wx, s_wy, s_xx, s_xy, s_yy, pab);

    return mle_finish(pab, logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * refine_lambda_mle_ncvt1_split
 *
 * Golden section refinement for MLE using a caller-selected coarse bracket.
 *
 * Returns optimal MLE lambda; writes log-likelihood to *logl_out.
 * hi_eval is a caller-provided scratch buffer of size (n_samples,).
 * ------------------------------------------------------------------------- */
double refine_lambda_mle_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    int n_samples,
    const double *lambda_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    int best_idx,
    double mle_const,
    double * restrict hi_eval,
    double *logl_out
)
{
    const double phi = 0.6180339887498949;

    /* Fully degenerate SNP */
    if (best_idx < 0) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    /* Bracket around best grid point */
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                      inv_ww, inv_wy, inv_yy, eigenvalues,
                                      n_samples, exp(c), mle_const, hi_eval);
    double fd = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                      inv_ww, inv_wy, inv_yy, eigenvalues,
                                      n_samples, exp(d), mle_const, hi_eval);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       n_samples, exp(c), mle_const, hi_eval);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       n_samples, exp(d), mle_const, hi_eval);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                      inv_ww, inv_wy, inv_yy, eigenvalues,
                                      n_samples, lambda_opt, mle_const, hi_eval);

    return lambda_opt;
}


double golden_section_lambda_mle_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const grid_invariant_t *grid_inv,
    double log_l_min, double step,
    int n_grid, int n_refine,
    double mle_const,
    double * restrict hi_eval,
    double *logl_out
)
{
    int best_idx = coarse_grid_mle_ncvt1_split(
        var_wx, var_xx, var_xy, n_samples,
        hi_eval_grid, logdet_h_grid, grid_inv, n_grid, mle_const
    );
    return refine_lambda_mle_ncvt1_split(
        var_wx, var_xx, var_xy, inv_ww, inv_wy, inv_yy,
        eigenvalues, n_samples, lambda_grid, log_l_min, step,
        n_grid, n_refine, best_idx, mle_const, hi_eval, logl_out
    );
}

/* -------------------------------------------------------------------------
 * calc_pab_ncvt1_split
 *
 * Compute Pab from separated varying + invariant dot product sums.
 * The caller provides the 6 pre-accumulated sums (3 varying + 3 invariant).
 * ------------------------------------------------------------------------- */
void calc_pab_ncvt1_split(
    double s_ww, double s_wx, double s_wy,
    double s_xx, double s_xy, double s_yy,
    double pab[3][6]
)
{
    pab[0][0] = s_ww;
    pab[0][1] = s_wx;
    pab[0][2] = s_wy;
    pab[0][3] = s_xx;
    pab[0][4] = s_xy;
    pab[0][5] = s_yy;

    /* Row 1: project out W */
    double inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
    pab[1][3] = s_xx - s_wx * s_wx * inv_ww;
    pab[1][4] = s_xy - s_wx * s_wy * inv_ww;
    pab[1][5] = s_yy - s_wy * s_wy * inv_ww;

    /* Row 2: project out X */
    double ps_xx = pab[1][3];
    double inv_xx = (ps_xx != 0.0) ? 1.0 / ps_xx : 0.0;
    pab[2][5] = pab[1][5] - pab[1][4] * pab[1][4] * inv_xx;
}
