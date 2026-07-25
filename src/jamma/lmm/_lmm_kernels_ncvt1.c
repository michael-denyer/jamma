/*
 * _lmm_kernels_ncvt1.c — see _lmm_kernels_ncvt1.h.
 *
 * Pure arithmetic. Nothing here touches CPython, so it needs none of
 * _lmm_support.h's import_array() handling.
 */

#include "_lmm_kernels_ncvt1.h"

/* wald_from_pab and score_from_pab live with the other statistic extractors
 * in the tests unit; the optimizers call them once per SNP on convergence. */
#include "_lmm_tests.h"

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

/* -------------------------------------------------------------------------
 * calc_pab_ncvt1
 *
 * Compute all three Pab rows for n_cvt=1.
 * Single fused pass over n_samples accumulates all 6 dot products,
 * then computes the two projection rows from the accumulators.
 *
 * uab:      (n_samples, 6) C-contiguous, row-major
 * hi_eval:  (n_samples,) = 1 / (lambda * eigenvalues + 1)
 * n_samples: number of samples
 * pab:      output 3x6 array (caller-allocated)
 * ------------------------------------------------------------------------- */
void calc_pab_ncvt1(
    const double * restrict uab,
    const double * restrict hi_eval,
    int n_samples,
    double pab[3][6]
)
{
    /* Fused single pass: accumulate all 6 dot products simultaneously.
     * uab is row-major (n_samples, 6), so row i starts at uab[i*6].
     * Accessing 6 consecutive doubles per row is cache-friendly. */
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
    #pragma omp simd reduction(+:s0,s1,s2,s3,s4,s5)
    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval[i];
        const double *row = uab + i * 6;
        s0 += h * row[0];
        s1 += h * row[1];
        s2 += h * row[2];
        s3 += h * row[3];
        s4 += h * row[4];
        s5 += h * row[5];
    }
    pab[0][0] = s0;
    pab[0][1] = s1;
    pab[0][2] = s2;
    pab[0][3] = s3;
    pab[0][4] = s4;
    pab[0][5] = s5;

    /* Row 1: project out W (column index 0 = ww) */
    {
        double inv_ww = (s0 != 0.0) ? 1.0 / s0 : 0.0;
        pab[1][3] = s3 - s1 * s1 * inv_ww;  /* xx */
        pab[1][4] = s4 - s1 * s2 * inv_ww;  /* xy */
        pab[1][5] = s5 - s2 * s2 * inv_ww;  /* yy */
    }

    /* Row 2: project out X (column index 3 = xx at level 1) */
    {
        double ps_xx = pab[1][3];
        double inv_xx = (ps_xx != 0.0) ? 1.0 / ps_xx : 0.0;
        pab[2][5] = pab[1][5] - pab[1][4] * pab[1][4] * inv_xx;  /* yy */
    }
}

/* -------------------------------------------------------------------------
 * reml_logl_ncvt1
 *
 * REML log-likelihood for one SNP at one lambda (n_cvt=1).
 * Used during golden section refinement where lambda is SNP-specific.
 *
 * Returns REML log-likelihood (positive = better; optimizer maximises).
 * ------------------------------------------------------------------------- */
double reml_logl_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    double lambda,
    double reml_const,
    double * restrict hi_eval
)
{
    int df = n_samples - 2;

    double logdet_h = 0.0;
    #pragma omp simd reduction(+:logdet_h)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        hi_eval[i] = 1.0 / v;
        logdet_h += log(v);  /* v > 1.0: lambda > 0, eval >= 0 */
    }

    double pab[3][6];
    calc_pab_ncvt1(uab, hi_eval, n_samples, pab);

    return reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
}

/* -------------------------------------------------------------------------
 * reml_logl_ncvt1_cached
 *
 * REML log-likelihood using precomputed hi_eval and logdet_h from the
 * shared coarse grid cache. Avoids recomputing 1/(lambda*eval+1) and
 * log(v) for every SNP at every grid point.
 *
 * Returns REML log-likelihood.
 * ------------------------------------------------------------------------- */
double reml_logl_ncvt1_cached(
    const double * restrict uab,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    double logdet_iab,
    int n_samples,
    int df,
    double reml_const
)
{
    double pab[3][6];
    calc_pab_ncvt1(uab, cached_hi_eval, n_samples, pab);

    return reml_finish(pab, cached_logdet_h, logdet_iab, df, reml_const);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_ncvt1
 *
 * Grid search + golden section refinement in log-lambda space.
 * Returns optimal lambda; writes optimal logl to *logl_out.
 *
 * Coarse search uses precomputed hi_eval_grid and logdet_h_grid (shared
 * across all SNPs). Refinement uses per-eval reml_logl_ncvt1 since
 * refinement lambdas are SNP-specific.
 * ------------------------------------------------------------------------- */
double golden_section_lambda_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    int df, double reml_const,
    double * restrict hi_eval, double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    const double phi = 0.6180339887498949;  /* golden ratio - 1 */

    /* Stage 1: coarse grid search using cached hi_eval and logdet_h.
     * Degenerate grid points return NaN from reml_finish (P_yy < 0);
     * map NaN → REML_SENTINEL so the > comparison skips them. */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_ncvt1_cached(
            uab,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            logdet_iab,
            n_samples, df, reml_const
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Every grid point produced NaN — fully degenerate SNP.
     * Without this, refinement proceeds on a meaningless bracket from
     * best_idx=0 and can produce finite but nonsensical results. */
    if (best_logl == REML_SENTINEL) {
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

    /* Stage 2: golden section refinement (SNP-specific lambdas) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                 n_samples, exp(c), reml_const, hi_eval);
    double fd = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                 n_samples, exp(d), reml_const, hi_eval);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            /* Maximum is in [a, d] — keep left */
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                  n_samples, exp(c), reml_const, hi_eval);
        } else {
            /* Maximum is in [c, b] — keep right */
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                  n_samples, exp(d), reml_const, hi_eval);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                 n_samples, lambda_opt, reml_const, hi_eval);

    /* hi_eval is now populated with 1/(lambda_opt*eval+1) — reuse for Wald stats
     * without another n_samples pass to recompute hi_eval. */
    {
        double pab[3][6];
        calc_pab_ncvt1(uab, hi_eval, n_samples, pab);
        *is_valid_out = wald_from_pab(pab, df, beta_out, se_out, f_stat_out);
    }

    return lambda_opt;
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
 * reml_logl_ncvt1_cached_split
 *
 * Coarse grid REML log-likelihood using:
 *   - Precomputed hi_eval_grid and logdet_h_grid (per grid point)
 *   - Precomputed grid_invariant_t (invariant dot products per grid point)
 *   - Only 3 varying reductions from DRAM per SNP per grid eval
 *
 * This is the primary performance win: DRAM reads drop from 6 to 3
 * doubles per sample. Invariant sums come from L1-resident struct.
 *
 * SoA layout: var_wx/xx/xy are contiguous (stride-1) for SIMD.
 * ------------------------------------------------------------------------- */
double reml_logl_ncvt1_cached_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    double logdet_iab,
    const grid_invariant_t *ginv,
    int n_samples,
    int df,
    double reml_const
)
{
    double pab[3][6];
    calc_pab_ncvt1_cached_split(
        var_wx, var_xx, var_xy, cached_hi_eval, ginv, n_samples, pab
    );
    return reml_finish_cached_split(
        pab, cached_logdet_h, logdet_iab, ginv, df, reml_const
    );
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
    double reml_const
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

    return reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
}


int coarse_grid_reml_ncvt1_split(
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
    double reml_const
)
{
    double best_logl = REML_SENTINEL;
    int best_idx = -1;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_ncvt1_cached_split(
            var_wx, var_xx, var_xy,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g], logdet_iab, &grid_inv[g],
            n_samples, df, reml_const
        );
        if (!isnan(logl) && logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }
    return best_idx;
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
                                       reml_const);
    double fd = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       logdet_iab, n_samples, exp(d),
                                       reml_const);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                        inv_ww, inv_wy, inv_yy, eigenvalues,
                                        logdet_iab, n_samples, exp(c),
                                        reml_const);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                        inv_ww, inv_wy, inv_yy, eigenvalues,
                                        logdet_iab, n_samples, exp(d),
                                        reml_const);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);

    /* Final evaluation: fuse REML logl + Wald stats in single n_samples pass.
     * This eliminates the separate calc_rl_wald_ncvt1_split call that would
     * redundantly recompute the identical Pab sums. */
    {
        double logdet_h = 0.0;
        double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
        double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

        #pragma omp simd reduction(+:logdet_h,s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
        for (int i = 0; i < n_samples; i++) {
            double v = lambda_opt * eigenvalues[i] + 1.0;
            double h = 1.0 / v;
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

        *logl_out = reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
        *is_valid_out = wald_from_pab(pab, df, beta_out, se_out, f_stat_out);
    }

    return lambda_opt;
}


double golden_section_lambda_ncvt1_split(
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
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const grid_invariant_t *grid_inv,
    double log_l_min, double step,
    int n_grid, int n_refine,
    int df, double reml_const,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    int best_idx = coarse_grid_reml_ncvt1_split(
        var_wx, var_xx, var_xy, n_samples,
        hi_eval_grid, logdet_h_grid, grid_inv, n_grid,
        logdet_iab, df, reml_const
    );
    return refine_lambda_ncvt1_split(
        var_wx, var_xx, var_xy, inv_ww, inv_wy, inv_yy,
        eigenvalues, logdet_iab, n_samples, lambda_grid,
        log_l_min, step, n_grid, n_refine, best_idx,
        df, reml_const, logl_out, beta_out, se_out, f_stat_out,
        is_valid_out
    );
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
 * mle_logl_ncvt1
 *
 * MLE log-likelihood for one SNP at one lambda (n_cvt=1).
 * Used during golden section refinement.
 * Returns MLE log-likelihood.
 * ------------------------------------------------------------------------- */
double mle_logl_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    double * restrict hi_eval
)
{
    double logdet_h = 0.0;
    #pragma omp simd reduction(+:logdet_h)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        hi_eval[i] = 1.0 / v;
        logdet_h += log(v);
    }

    double pab[3][6];
    calc_pab_ncvt1(uab, hi_eval, n_samples, pab);

    return mle_finish(pab, logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_cached
 *
 * MLE log-likelihood using precomputed hi_eval and logdet_h from
 * the shared coarse grid cache.
 * ------------------------------------------------------------------------- */
double mle_logl_ncvt1_cached(
    const double * restrict uab,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    int n_samples,
    double mle_const
)
{
    double pab[3][6];
    calc_pab_ncvt1(uab, cached_hi_eval, n_samples, pab);

    return mle_finish(pab, cached_logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_mle_ncvt1
 *
 * Grid search + golden section refinement for MLE lambda (n_cvt=1).
 * Structurally identical to golden_section_lambda_ncvt1 but:
 *   - Uses mle_finish instead of reml_finish
 *   - No Iab parameter or logdet_iab
 *   - Uses n_samples instead of df in the likelihood
 *
 * Returns optimal lambda; writes logl to *logl_out.
 * ------------------------------------------------------------------------- */
double golden_section_lambda_mle_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    double mle_const,
    double * restrict hi_eval, double *logl_out
)
{
    const double phi = 0.6180339887498949;  /* golden ratio - 1 */

    /* Stage 1: coarse grid search using cached hi_eval and logdet_h */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_ncvt1_cached(
            uab,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            n_samples, mle_const
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Fully degenerate SNP — all grid evaluations returned NaN */
    if (best_logl == REML_SENTINEL) {
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
    double fc = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                exp(c), mle_const, hi_eval);
    double fd = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                exp(d), mle_const, hi_eval);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                 exp(c), mle_const, hi_eval);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                 exp(d), mle_const, hi_eval);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                lambda_opt, mle_const, hi_eval);

    return lambda_opt;
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
