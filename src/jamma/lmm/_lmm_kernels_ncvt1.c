/*
 * _lmm_kernels_ncvt1.c — see _lmm_kernels_ncvt1.h.
 *
 * Pure arithmetic. Nothing here touches CPython, so it needs none of
 * _lmm_support.h's import_array() handling.
 */

#include "_lmm_kernels_ncvt1.h"

#include "_lmm_stats.h"
/* logdet_h_lambda: the logdet(H) term every REML and MLE evaluation needs. */
#include "_lmm_logdet.h"
#include "_lmm_lambda_search.h"

#include <math.h>
#include <float.h>
#include <string.h>


static inline double reml_finish(
    const double pab[3][6],
    double logdet_h,
    double logdet_iab,
    int df,
    double reml_const
)
{
    double logdet_pab = logdet_diag_term(pab[0][0]) + logdet_diag_term(pab[1][3]);
    double logdet_hiw = logdet_pab - logdet_iab;

    double P_yy = replace_zero_p_yy(pab[2][5]);
    if (P_yy < 0.0) P_yy = (double)NAN;

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
    double logdet_pab = ginv->log_s_ww + logdet_diag_term(pab[1][3]);
    double logdet_hiw = logdet_pab - logdet_iab;

    double P_yy = replace_zero_p_yy(pab[2][5]);
    if (P_yy < 0.0) P_yy = (double)NAN;

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
static double reml_logl_ncvt1_split(
    const ncvt1_snp_t *snp,
    double lambda,
    double (*pab_out)[6]
)
{
    const double * restrict var_wx = snp->var_wx;
    const double * restrict var_xx = snp->var_xx;
    const double * restrict var_xy = snp->var_xy;
    const double * restrict inv_ww = snp->inv_ww;
    const double * restrict inv_wy = snp->inv_wy;
    const double * restrict inv_yy = snp->inv_yy;
    const double * restrict eigenvalues = snp->eigenvalues;
    int n_samples = snp->n_samples;
    int df = n_samples - 2;

    /* Fused: hi_eval + all 6 dot products in a single pass. logdet_h is a
     * second pass over eigenvalues alone (n doubles, L1-resident): the
     * per-element log() it used to add here dominated the loop, see
     * _lmm_logdet.h. SoA layout gives stride-1 access for all 6 columns,
     * so contiguous SIMD loads instead of stride-3 gathers. */
    double logdet_h = logdet_h_lambda(eigenvalues, n_samples, lambda);
    double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

    #pragma omp simd reduction(+:s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;

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

    return reml_finish(pab, logdet_h, snp->logdet_iab, df, snp->reml_const);
}

static double reml_objective_ncvt1(const void *ctx, double lambda)
{
    return reml_logl_ncvt1_split((const ncvt1_snp_t *)ctx, lambda, NULL);
}

/* Pab terms and their d/d log lambda that the REML and MLE scores share,
 * from compensated sums. trace is d log det(H) / d log lambda. */
typedef struct {
    double trace, s_ww, ds_ww, pxx, dpxx, pyy, dpyy;
} ncvt1_score_terms_t;

/* Returns 0 when a Schur-complement pivot is not positive. */
static int score_terms_ncvt1(const ncvt1_snp_t *snp, double lambda,
                             ncvt1_score_terms_t *out)
{
    const double * restrict var_wx = snp->var_wx;
    const double * restrict var_xx = snp->var_xx;
    const double * restrict var_xy = snp->var_xy;
    const double * restrict inv_ww = snp->inv_ww;
    const double * restrict inv_wy = snp->inv_wy;
    const double * restrict inv_yy = snp->inv_yy;
    const double * restrict eigenvalues = snp->eigenvalues;
    int n_samples = snp->n_samples;
    double s[6] = {0}, ds[6] = {0}, cs[6] = {0}, cds[6] = {0};
    double trace = 0.0, ctrace = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double d = eigenvalues[i];
        double h = 1.0 / (1.0 + lambda * d);
        double dh = -lambda * d * h * h;
        double trace_value = lambda * d * h - ctrace;
        double trace_next = trace + trace_value;
        ctrace = (trace_next - trace) - trace_value;
        trace = trace_next;
        const double values[6] = {
            inv_ww[i], var_wx[i], inv_wy[i],
            var_xx[i], var_xy[i], inv_yy[i]
        };
        for (int j = 0; j < 6; j++) {
            double value = h * values[j] - cs[j];
            double next = s[j] + value;
            cs[j] = (next - s[j]) - value;
            s[j] = next;
            value = dh * values[j] - cds[j];
            next = ds[j] + value;
            cds[j] = (next - ds[j]) - value;
            ds[j] = next;
        }
    }
    if (!(s[0] > 0.0)) return 0;
    double pxx = s[3] - s[1] * s[1] / s[0];
    double pxy = s[4] - s[1] * s[2] / s[0];
    double pyy1 = s[5] - s[2] * s[2] / s[0];
    double dpxx = ds[3] - 2.0 * s[1] * ds[1] / s[0]
                  + s[1] * s[1] * ds[0] / (s[0] * s[0]);
    double dpxy = ds[4] - (ds[1] * s[2] + s[1] * ds[2]) / s[0]
                  + s[1] * s[2] * ds[0] / (s[0] * s[0]);
    double dpyy1 = ds[5] - 2.0 * s[2] * ds[2] / s[0]
                   + s[2] * s[2] * ds[0] / (s[0] * s[0]);
    if (!(pxx > 0.0)) return 0;
    double pyy = pyy1 - pxy * pxy / pxx;
    double dpyy = dpyy1 - 2.0 * pxy * dpxy / pxx
                  + pxy * pxy * dpxx / (pxx * pxx);
    if (!(pyy > 0.0)) return 0;
    *out = (ncvt1_score_terms_t){
        .trace = trace, .s_ww = s[0], .ds_ww = ds[0],
        .pxx = pxx, .dpxx = dpxx, .pyy = pyy, .dpyy = dpyy,
    };
    return 1;
}

static double reml_score_loglambda_ncvt1(const void *ctx, double lambda)
{
    const ncvt1_snp_t *snp = (const ncvt1_snp_t *)ctx;
    ncvt1_score_terms_t t;
    if (!score_terms_ncvt1(snp, lambda, &t)) return NAN;
    return -0.5 * t.trace - 0.5 * t.ds_ww / t.s_ww - 0.5 * t.dpxx / t.pxx
           - 0.5 * (snp->n_samples - 2) * t.dpyy / t.pyy;
}

static double mle_score_loglambda_ncvt1(const void *ctx, double lambda)
{
    const ncvt1_snp_t *snp = (const ncvt1_snp_t *)ctx;
    ncvt1_score_terms_t t;
    if (!score_terms_ncvt1(snp, lambda, &t)) return NAN;
    return -0.5 * t.trace - 0.5 * snp->n_samples * t.dpyy / t.pyy;
}



/* -------------------------------------------------------------------------
 * refine_lambda_ncvt1_split
 *
 * REML lambda from the caller's coarse-grid index.
 * ------------------------------------------------------------------------- */
double refine_lambda_ncvt1_split(
    const ncvt1_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    int df,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    /* Every grid point produced NaN — fully degenerate SNP. */
    if (best_idx < 0) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return search->lambda_grid[0];
    }

    double lambda_opt = exp(golden_section_log_lambda(
        reml_objective_ncvt1, reml_score_loglambda_ncvt1, snp,
        search, best_idx));

    /* Final evaluation: reml_logl_ncvt1_split fills pab as a side effect, so
     * the Wald extraction below reads the same Pab the logl was computed
     * from without a second n_samples pass. */
    double pab[3][6];
    *logl_out = reml_logl_ncvt1_split(snp, lambda_opt, pab);
    *is_valid_out = wald_stats(pab_terms_ncvt1(pab), df,
                               beta_out, se_out, f_stat_out);

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
    double P_yy = replace_zero_p_yy(pab[2][5]);
    if (P_yy < 0.0) return (double)NAN;
    return mle_const - 0.5 * logdet_h - 0.5 * n_samples * log(P_yy);
}

void coarse_grid_ncvt1_split(
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
    if (best_reml_idx) *best_reml_idx = -1;
    if (best_mle_idx) *best_mle_idx = -1;

    for (int g = 0; g < n_grid; g++) {
        const grid_invariant_t *ginv = &grid_inv[g];
        double pab[3][6];
        calc_pab_ncvt1_cached_split(
            var_wx, var_xx, var_xy,
            hi_eval_grid + (size_t)g * n_samples,
            ginv, n_samples, pab
        );

        if (best_reml_idx) {
            double reml_logl = reml_finish_cached_split(
                pab, logdet_h_grid[g], logdet_iab, ginv, df, reml_const
            );
            if (!isnan(reml_logl) && reml_logl > best_reml) {
                best_reml = reml_logl;
                *best_reml_idx = g;
            }
        }
        if (best_mle_idx) {
            double mle_logl = mle_finish(
                pab, logdet_h_grid[g], n_samples, mle_const
            );
            if (!isnan(mle_logl) && mle_logl > best_mle) {
                best_mle = mle_logl;
                *best_mle_idx = g;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_split
 *
 * MLE log-likelihood from SoA split data at an arbitrary lambda.
 * Used during golden section refinement. Computes each Hi_eval term inline,
 * accumulates all 6 dot products (3 invariant + 3 varying), builds Pab.
 * ------------------------------------------------------------------------- */
static double mle_logl_ncvt1_split(const void *ctx, double lambda)
{
    const ncvt1_snp_t *snp = (const ncvt1_snp_t *)ctx;
    const double * restrict var_wx = snp->var_wx;
    const double * restrict var_xx = snp->var_xx;
    const double * restrict var_xy = snp->var_xy;
    const double * restrict inv_ww = snp->inv_ww;
    const double * restrict inv_wy = snp->inv_wy;
    const double * restrict inv_yy = snp->inv_yy;
    const double * restrict eigenvalues = snp->eigenvalues;
    int n_samples = snp->n_samples;
    double logdet_h = logdet_h_lambda(eigenvalues, n_samples, lambda);
    double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

    #pragma omp simd reduction(+:s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;

        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];

        s_ww += h * inv_ww[i];
        s_wy += h * inv_wy[i];
        s_yy += h * inv_yy[i];
    }

    double pab[3][6];
    calc_pab_ncvt1_split(s_ww, s_wx, s_wy, s_xx, s_xy, s_yy, pab);

    return mle_finish(pab, logdet_h, n_samples, snp->mle_const);
}

/* -------------------------------------------------------------------------
 * refine_lambda_mle_ncvt1_split
 *
 * MLE lambda from the caller's coarse-grid index. Returns the optimal
 * lambda; writes the log-likelihood to *logl_out.
 * ------------------------------------------------------------------------- */
double refine_lambda_mle_ncvt1_split(
    const ncvt1_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    double *logl_out
)
{
    /* Fully degenerate SNP */
    if (best_idx < 0) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    double lambda_opt = exp(golden_section_log_lambda(
        mle_logl_ncvt1_split, mle_score_loglambda_ncvt1, snp, search,
        best_idx));
    *logl_out = mle_logl_ncvt1_split(snp, lambda_opt);

    return lambda_opt;
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
