/*
 * _lmm_accel.c — C extension implementing per-SNP REML/MLE pipelines
 * for Wald, Score, and LRT tests (n_cvt=1 and general n_cvt).
 *
 * Exported functions: compute_lmm_batch_c, compute_lmm_batch_split_c,
 *                     create_workspace_split_c, compute_lmm_chunk_split_c,
 *                     create_workspace_general_c, compute_lmm_chunk_general_c,
 *                     compute_score_batch_c, compute_lrt_batch_c,
 *                     compute_score_batch_general_c, compute_lrt_batch_general_c,
 *                     create_workspace_mode4_split_c, compute_mode4_chunk_split_c,
 *                     compute_score_split_c, compute_lrt_split_c,
 *                     compute_score_fused_c, compute_lrt_fused_c
 *
 * Phase 116.1: -DJAMMA_SENTINEL_UB enables a heap-OOB sentinel function
 * (jamma_sentinel_oob) for sanitizer-workflow self-test. See
 * scripts/asan-suppressions.txt and .github/workflows/sanitizers.yml.
 * Never set in wheel builds — the macro is opt-in via apply_sanitizer_overrides
 * machinery and only the sanitizer workflow's separate sentinel-meta-test
 * job ever defines it.
 *
 * Translates the Python/NumPy golden-section REML/MLE optimizer + Wald/Score/LRT
 * test pipelines (likelihood_numpy.py) to C with optional OpenMP parallelism.
 *
 * Performance optimizations over the naive per-call approach (n_cvt=1 path;
 * the general n_cvt path uses table-driven recursion with cached invariant
 * dot products — see "GENERAL n_cvt support" section below):
 *   1. Fused Pab: single pass over n_samples accumulates all 6 dot products
 *   2. Thread-local hi_eval: one malloc per worker thread, reused across SNPs
 *   3. Pre-computed logdet_iab: lambda-independent log(iab) terms computed once
 *   4. Pre-built lambda grid: avoids exp() in every SNP's coarse search loop
 *   5. Hoisted constants: REML normalizer + betainc lgamma terms computed once
 *   6. C-side betainc: Lentz CF for F->p-value avoids Python round-trip
 *   7. Cached coarse-grid hi_eval: hi_eval[g][i] and logdet_h[g] precomputed
 *      once across all SNPs — eliminates n_snps * n_grid redundant hi_eval passes
 *   8. restrict + SIMD hints: helps compiler vectorize hot inner loops
 *      (#pragma omp simd is used without #ifdef _OPENMP guards — unknown
 *      pragmas are silently ignored per the C standard, so these are safe
 *      on non-OpenMP compilers and act purely as vectorization hints)
 *
 * Pab indexing (n_cvt=1, build_index_table(1)):
 *   n_index = 6
 *   col 0: ww = GetabIndex(1,1,1) = 0
 *   col 1: wx = GetabIndex(1,2,1) = 1
 *   col 2: wy = GetabIndex(1,3,1) = 2
 *   col 3: xx = GetabIndex(2,2,1) = 3
 *   col 4: xy = GetabIndex(2,3,1) = 4
 *   col 5: yy = GetabIndex(3,3,1) = 5
 *
 *   Row 0: dot products (all 6 columns)
 *   Row 1 (project W):
 *     Pab[1][3] = Pab[0][3] - Pab[0][1]*Pab[0][1] / Pab[0][0]  (xx)
 *     Pab[1][4] = Pab[0][4] - Pab[0][1]*Pab[0][2] / Pab[0][0]  (xy)
 *     Pab[1][5] = Pab[0][5] - Pab[0][2]*Pab[0][2] / Pab[0][0]  (yy)
 *   Row 2 (project X):
 *     Pab[2][5] = Pab[1][5] - Pab[1][4]*Pab[1][4] / Pab[1][3]  (yy)
 *
 *   logdet_diag_indices: [(0, 0), (1, 3)]
 *   idx_xx = 3, idx_xy = 4, idx_yy = 5
 *   nc_total = n_cvt + 1 = 2 (Pab row for Px_YY)
 *   df = n_samples - 2 (n_cvt=1)
 */

#include "_lmm_stats.h"
#include "_lmm_support.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Minimum P_yy guard — matches _P_YY_MIN in likelihood.py */
#define P_YY_MIN 1e-8

/* ABI version: bump when function signatures or array layout expectations change.
 * The Python side checks this at import time to detect stale .so files. */
#define ABI_VERSION 11  /* v11: Persistent Score/LRT workspaces (eliminate per-chunk malloc) */

/* REML sentinel: replaces NaN log-likelihood from degenerate P_yy.
 * reml_finish returns NaN when P_yy < 0; the golden section callers
 * map NaN -> REML_SENTINEL so the > comparison skips degenerate points
 * without needing an isnan() guard on every iteration.
 * Matches the Python path's np.where(isnan, -inf, logl). */
#define REML_SENTINEL (-INFINITY)


/* =========================================================================
 * Shared helpers — eliminate duplication across full/split paths
 * ========================================================================= */


/* REML log-likelihood tail: logdet_pab + P_yy guard + REML formula.
 * Shared by reml_logl_ncvt1, reml_logl_ncvt1_cached, reml_logl_ncvt1_split. */
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

/* Wald statistics from a populated pab array.
 * Shared by golden_section_lambda_ncvt1 and golden_section_lambda_ncvt1_split.
 *
 * Returns 1 if the SNP is valid (P_XX > 0), 0 if degenerate (P_XX <= 0).
 * Degenerate SNPs get beta = se = f_stat = NaN.
 *
 * The return value (not isnan(beta)) is used for validity checks — this is
 * more robust than relying on NaN propagation through comparisons. */
static inline int wald_from_pab(
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
static inline int score_from_pab(
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
static void calc_pab_ncvt1(
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
static double reml_logl_ncvt1(
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
static double reml_logl_ncvt1_cached(
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
 * compute_logdet_iab
 *
 * Pre-compute the lambda-independent log(iab) terms for one SNP.
 * logdet_diag_indices for n_cvt=1: [(0, 0), (1, 3)]
 *   iab layout: row r, col c -> iab[r*6 + c]
 *
 * This is called once per SNP before the optimization loop, avoiding
 * redundant log() calls across n_grid + n_refine + 3 logl evaluations per SNP.
 * ------------------------------------------------------------------------- */
static double compute_logdet_iab(const double *iab)
{
    double logdet = 0.0;
    /* (row=0, col=0) = ww */
    if (iab[0 * 6 + 0] > 0.0) logdet += log(iab[0 * 6 + 0]);
    /* (row=1, col=3) = xx at level 1 */
    if (iab[1 * 6 + 3] > 0.0) logdet += log(iab[1 * 6 + 3]);
    return logdet;
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
static double golden_section_lambda_ncvt1(
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

/* =========================================================================
 * SPLIT-Uab functions (SoA layout)
 *
 * These variants operate on separated varying/invariant Uab columns to
 * halve per-SNP DRAM traffic. The invariant columns (ww, wy, yy) are
 * identical across all SNPs and fit in L2 cache after the first SNP.
 *
 * SoA (Structure-of-Arrays) layout for SIMD:
 *   uab_var: (n_snps, 3, n_samples) — columns [wx, xx, xy] contiguous
 *   uab_inv: (3, n_samples)         — columns [ww, wy, yy] contiguous
 *
 * Each column is stride-1, enabling contiguous SIMD loads (vmovupd)
 * instead of stride-3 gather instructions (vgatherdpd).
 * ========================================================================= */

/* Pre-computed invariant dot products for one coarse grid point.
 * Memory: n_grid * sizeof(grid_invariant_t) ~ 50 * 32 = 1.6 KB (fits L1). */
typedef struct {
    double s_ww;       /* sum of hi * ww */
    double s_wy;       /* sum of hi * wy */
    double s_yy;       /* sum of hi * yy */
    double log_s_ww;   /* log(s_ww) if > 0, else 0 */
} grid_invariant_t;

/* -------------------------------------------------------------------------
 * calc_pab_ncvt1_split
 *
 * Compute Pab from separated varying + invariant dot product sums.
 * The caller provides the 6 pre-accumulated sums (3 varying + 3 invariant).
 * ------------------------------------------------------------------------- */
static void calc_pab_ncvt1_split(
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

/* Accumulate the three SNP-varying coarse-grid reductions and combine them
 * with the precomputed invariant reductions into the canonical Pab layout. */
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

/* Cached split REML tail. The invariant W determinant was precomputed with
 * the coarse-grid weights, so only the SNP-specific X term needs a log. */
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
static double reml_logl_ncvt1_cached_split(
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
static double reml_logl_ncvt1_split(
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

/* Return the best REML coarse-grid index, or -1 when every point is degenerate. */
static int coarse_grid_reml_ncvt1_split(
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
static double refine_lambda_ncvt1_split(
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

/* Full REML optimization for callers that do not share the coarse-grid pass. */
static double golden_section_lambda_ncvt1_split(
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

/* =========================================================================
 * Workspace API — persistent cross-chunk state for split-Uab pipeline
 *
 * Eliminates per-chunk malloc + grid precomputation overhead:
 *   - lambda_grid, hi_eval_grid, logdet_h_grid built once per run
 *   - grid_inv (invariant dot products) built once per run
 *   - iab_s_ww / iab_log_ww precomputed from invariant column sums
 *   - Python arrays kept alive via Py_INCREF until workspace freed
 * ========================================================================= */

typedef struct {
    int n_samples;
    int n_grid;
    int n_refine;
    int df;
    double l_min, l_max, log_l_min, step;
    double reml_const;
    double beta_a, beta_b, lbeta_ab;
    /* Per-grid precomputed data (owned by workspace) */
    double *lambda_grid;      /* (n_grid,) */
    double *hi_eval_grid;     /* (n_grid * n_samples) */
    double *logdet_h_grid;    /* (n_grid,) */
    grid_invariant_t *grid_inv;  /* (n_grid,) */
    /* Invariant Iab scalars (lambda-independent) */
    double iab_s_ww;    /* sum(inv_ww) */
    double iab_inv_ww;  /* 1/iab_s_ww (or 0) */
    double iab_log_ww;  /* log(iab_s_ww) (or 0) */
    /* Borrowed pointers — kept alive via Py_INCREF */
    const double *eigenvalues;
    const double *inv_ww;   /* uab_invariant_soa row 0 */
    const double *inv_wy;   /* uab_invariant_soa row 1 */
    const double *inv_yy;   /* uab_invariant_soa row 2 */
    PyObject *eigenvalues_ref;  /* keeps eigenvalues array alive */
    PyObject *uab_inv_ref;      /* keeps uab_invariant_soa array alive */
    /* Mode-4 fused fields (only populated when mode=4) */
    int mode;                   /* 0=Wald-only (default from calloc zero-init), 4=fused mode-4 */
    double *hi_eval_null;       /* (n_samples,) null-model Hi_eval, owned */
    double logl_H0;             /* null MLE log-likelihood */
    double mle_const;           /* 0.5 * n * (log(n) - log(2*pi) - 1) */
    double null_s_ww;           /* invariant dot product under null Hi_eval */
    double null_s_wy;
    double null_s_yy;
    double null_inv_ww;         /* 1/null_s_ww */
    /* Fused Uab fields -- w and Uty stored for on-the-fly wx/xx/xy computation */
    const double *w;          /* UtW[:,0] for n_cvt=1 -- (n_samples,) borrowed */
    const double *Uty;        /* rotated phenotype -- (n_samples,) borrowed */
    PyObject *w_ref;          /* keeps w array alive */
    PyObject *Uty_ref;        /* keeps Uty array alive */
} lmm_workspace_t;

/* PyCapsule destructor: free owned allocations, release Python array refs. */
static void lmm_workspace_destructor(PyObject *cap)
{
    lmm_workspace_t *ws =
        (lmm_workspace_t *)PyCapsule_GetPointer(cap, "lmm_workspace");
    if (!ws) return;
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->grid_inv);
    free(ws->hi_eval_null);
    Py_XDECREF(ws->eigenvalues_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->w_ref);
    Py_XDECREF(ws->Uty_ref);
    free(ws);
}

/* -------------------------------------------------------------------------
 * create_workspace_split_c
 *
 * Python signature:
 *   create_workspace_split_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       n_threads,        # int
 *   ) -> PyCapsule wrapping lmm_workspace_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "n_samples",
        "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

    /* Fill scalar fields */
    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->l_min     = l_min;
    ws->l_max     = l_max;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    ws->log_l_min   = log(l_min);
    double log_l_max = log(l_max);
    ws->step        = (log_l_max - ws->log_l_min) / (double)(n_grid - 1);
    ws->reml_const  = 0.5 * ws->df * (log((double)ws->df)
                       - log(2.0 * M_PI) - 1.0);

    /* Borrow pointers — arrays kept alive via Py_INCREF */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    /* Compute invariant Iab scalar: sum(inv_ww) */
    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_ws;
    }

    /* Build lambda grid + invariant dot products (same logic as batch_split_c) */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
            sw  += h * ws->inv_ww[i];
            swy += h * ws->inv_wy[i];
            sy  += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;

        ws->grid_inv[g].s_ww    = sw;
        ws->grid_inv[g].s_wy    = swy;
        ws->grid_inv[g].s_yy    = sy;
        ws->grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
    }

    /* Wrap in PyCapsule; destructor frees ws on GC */
    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    /* Release local refs — capsule now owns ws->*_ref via destructor */
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return capsule;

err_ws:
    /* Destructor not yet registered — free manually */
    if (ws) {
        Py_XDECREF(ws->eigenvalues_ref);
        Py_XDECREF(ws->uab_inv_ref);
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->grid_inv);
        free(ws);
    }
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * create_workspace_mode4_split_c
 *
 * Create a mode-4 workspace: extends the standard split workspace with
 * null-model Hi_eval (for Score), MLE constant, and null logl (for LRT).
 *
 * Python signature:
 *   create_workspace_mode4_split_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       n_threads,        # int
 *       hi_eval_null,     # (n_samples,) float64 — null-model Hi_eval
 *       logl_H0,          # float — null MLE log-likelihood
 *   ) -> PyCapsule wrapping lmm_workspace_t (mode=4)
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_mode4_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "n_samples",
        "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *hi_eval_null_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOiddiiiOd", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &hi_eval_null_obj, &logl_H0)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "hi_eval_null must be shape (n_samples,)");
        goto err_input;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    /* Validate Hi_eval_null for NaN/Inf and non-positive values */
    {
        const double *hi_null = (const double *)PyArray_DATA(hi_eval_null_arr);
        for (int i = 0; i < n_samples; i++) {
            char buf[64];
            if (!isfinite(hi_null[i])) {
                snprintf(buf, sizeof(buf), "%g", hi_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not finite. "
                    "Null model optimization may have failed.", i, buf);
                goto err_input;
            }
            if (hi_null[i] <= 0.0) {
                snprintf(buf, sizeof(buf), "%g", hi_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not positive. "
                    "Check kinship matrix conditioning.",
                    i, buf);
                goto err_input;
            }
        }
    }

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

    /* Fill scalar fields (same as create_workspace_split_c) */
    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->l_min     = l_min;
    ws->l_max     = l_max;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    ws->log_l_min   = log(l_min);
    double log_l_max = log(l_max);
    ws->step        = (log_l_max - ws->log_l_min) / (double)(n_grid - 1);
    ws->reml_const  = 0.5 * ws->df * (log((double)ws->df)
                       - log(2.0 * M_PI) - 1.0);

    /* Borrow pointers — arrays kept alive via Py_INCREF */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    /* Compute invariant Iab scalar: sum(inv_ww) */
    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_ws;
    }

    /* Build lambda grid + invariant dot products */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
            sw  += h * ws->inv_ww[i];
            swy += h * ws->inv_wy[i];
            sy  += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;

        ws->grid_inv[g].s_ww    = sw;
        ws->grid_inv[g].s_wy    = swy;
        ws->grid_inv[g].s_yy    = sy;
        ws->grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
    }

    /* --- Mode-4 specific fields --- */
    ws->mode = 4;
    ws->logl_H0 = logl_H0;
    ws->mle_const = 0.5 * (double)n_samples
                    * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

    /* Copy hi_eval_null into workspace-owned buffer */
    ws->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->hi_eval_null) {
        PyErr_NoMemory();
        goto err_ws;
    }
    {
        const double *src = (const double *)PyArray_DATA(hi_eval_null_arr);
        memcpy(ws->hi_eval_null, src, (size_t)n_samples * sizeof(double));
    }

    /* Precompute null-model invariant dot products under hi_eval_null */
    {
        double ns_ww = 0.0, ns_wy = 0.0, ns_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double h = ws->hi_eval_null[i];
            ns_ww += h * ws->inv_ww[i];
            ns_wy += h * ws->inv_wy[i];
            ns_yy += h * ws->inv_yy[i];
        }
        ws->null_s_ww   = ns_ww;
        ws->null_s_wy   = ns_wy;
        ws->null_s_yy   = ns_yy;
        ws->null_inv_ww  = (ns_ww != 0.0) ? 1.0 / ns_ww : 0.0;
    }

    /* Wrap in PyCapsule */
    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(hi_eval_null_arr);
    return capsule;

err_ws:
    if (ws) {
        Py_XDECREF(ws->eigenvalues_ref);
        Py_XDECREF(ws->uab_inv_ref);
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->grid_inv);
        free(ws->hi_eval_null);
        free(ws);
    }
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(hi_eval_null_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_split_c
 *
 * Per-chunk compute using a pre-built workspace. No allocation, no grid
 * rebuild, no Python Iab — logdet_iab computed entirely in C.
 *
 * Python signature:
 *   compute_lmm_chunk_split_c(
 *       workspace,     # PyCapsule from create_workspace_split_c
 *       uab_varying,   # (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_chunk_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "uab_varying", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *uab_var_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &uab_var_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_t *ws = (lmm_workspace_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace");
    if (!ws) return NULL;  /* PyCapsule_GetPointer sets ValueError */

    PyArrayObject *uab_var_arr = NULL;
    output_arrays_t out = {0};
    PyObject *result = NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) return NULL;

    int n_samples = ws->n_samples;

    /* Validate shape */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying must be shape (n_snps, 3, %d)", n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
    const double *inv_ww = ws->inv_ww;
    const double *inv_wy = ws->inv_wy;
    const double *inv_yy = ws->inv_yy;

    double *lambdas = (double *)PyArray_DATA(out.lambdas);
    double *logls   = (double *)PyArray_DATA(out.logls);
    double *betas   = (double *)PyArray_DATA(out.betas);
    double *ses     = (double *)PyArray_DATA(out.ses);
    double *pwalds  = (double *)PyArray_DATA(out.pwalds);

    int n_grid    = ws->n_grid;
    int n_refine  = ws->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Clamp n_threads to n_snps */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    Py_BEGIN_ALLOW_THREADS

    /* Static schedule: SNP cost is uniform (same n_grid, n_refine, n_samples).
     * No atomic work-stealing overhead vs dynamic scheduling. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        const double *snp_base = uab_var_data + (size_t)snp * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;

        /* Compute logdet_iab internally from raw Uab column sums.
         *
         * logdet_iab = log(iab[0][0]) + log(iab[1][3]) when positive.
         *
         *   iab[0][0] = sum(inv_ww) = ws->iab_s_ww  (invariant, precomputed)
         *   iab[1][3] = sum(var_xx) - (sum(var_wx))^2 / sum(inv_ww)
         *
         * The varying sums require O(n_samples) reductions — ~100 cycles for
         * 1400 samples, negligible vs the existing 50-grid-point REML loop.
         */
        double iab_s_wx = 0.0, iab_s_xx = 0.0;
        #pragma omp simd reduction(+:iab_s_wx,iab_s_xx)
        for (int i = 0; i < n_samples; i++) {
            iab_s_wx += vwx[i];
            iab_s_xx += vxx[i];
        }

        double iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * ws->iab_inv_ww;
        double logdet_iab = ws->iab_log_ww
                            + ((iab_p1_xx > 0.0) ? log(iab_p1_xx) : 0.0);

        /* Golden section lambda optimization (reuses workspace grids) */
        double logl_opt, beta, se, f_stat;
        int is_valid;
        double lambda_opt = golden_section_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->grid_inv, ws->log_l_min, ws->step, n_grid, n_refine,
            df, reml_const, &logl_opt, &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;

        pwalds[snp] = f_to_pvalue(
            f_stat, df, is_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(uab_var_arr);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(uab_var_arr);
    return NULL;
}

/* =========================================================================
 * GENERAL n_cvt support — table-driven Pab recursion for arbitrary covariates
 *
 * Adds new workspace type (lmm_workspace_general_t) and entry points
 * (create_workspace_general_c, compute_lmm_chunk_general_c) that accept
 * n_cvt as a parameter. The existing n_cvt=1 code path is unchanged.
 *
 * Key design: Python builds the recursion table (via build_pab_table_for_c)
 * and passes flat int32 arrays. C code just walks the table — no index
 * computation in C.
 *
 * Memory: Large per-SNP Pab buffers (pab_scratch, row0_scratch) are heap-
 * allocated per-thread in workspace structs or before parallel regions.
 * Only small MAX_N_INDEX arrays remain on the stack.
 * MAX_N_CVT=100 -> MAX_N_INDEX=5253 (~42KB per array).  Functions with
 * two such arrays peak at ~84KB, well within OpenMP thread stacks (2-4MB).
 * ========================================================================= */


/* -------------------------------------------------------------------------
 * calc_pab_general — Table-driven Pab recursion for arbitrary n_cvt.
 *
 * Row 0 from row0 array (dot product sums), rows 1..n_rows-1 from entries.
 * Output in pab[n_rows * n_index], row-major.
 * ------------------------------------------------------------------------- */
static void calc_pab_general(
    const double *row0,
    const pab_table_t *t,
    double *pab
)
{
    int ni = t->n_index;
    /* Copy row 0 */
    for (int i = 0; i < ni; i++) pab[i] = row0[i];

    /* Recursive projection: rows 1..n_rows-1 */
    for (int p = 1; p < t->n_rows; p++) {
        int offset = t->level_offsets[p];
        int count  = t->level_counts[p];
        for (int e = 0; e < count; e++) {
            const pab_entry_t *re = &t->entries[offset + e];
            double ps_ww = pab[(p - 1) * ni + re->index_ww];
            /* Match n_cvt=1 paths: zero projection when divisor is zero,
             * so Px_YY < 0 guard in wald_from_pab catches degeneracy. */
            double inv_ww = (ps_ww != 0.0) ? 1.0 / ps_ww : 0.0;
            pab[p * ni + re->index_ab] =
                pab[(p - 1) * ni + re->index_ab]
                - pab[(p - 1) * ni + re->index_aw]
                * pab[(p - 1) * ni + re->index_bw]
                * inv_ww;
        }
    }
}

/* -------------------------------------------------------------------------
 * logdet_from_row0 — compute logdet(Iab) from identity dot products.
 *
 * Encapsulates the identity Pab prepass: calls calc_pab_general into the
 * caller-provided scratch buffer, then extracts diagonal entries for logdet.
 * Replaces three inline copies of the same pattern in compute_lmm_chunk_general_c,
 * fused general Wald, and fused general mode-4.
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

/* -------------------------------------------------------------------------
 * reml_finish_general — REML tail for general n_cvt.
 *
 * logdet_pab from logdet_diag entries, P_yy guard, return full REML formula
 * including logdet_h.
 * ------------------------------------------------------------------------- */
static double reml_finish_general(
    const double *pab,
    const pab_table_t *t,
    double logdet_h,
    double logdet_iab,
    double reml_const
)
{
    int ni = t->n_index;
    int df = t->df;

    /* logdet_pab from diagonal entries.  A non-positive diagonal means the
     * projected matrix is not positive-definite — return NaN so the REML
     * sentinel mechanism correctly flags this as degenerate. */
    double logdet_pab = 0.0;
    for (int d = 0; d < t->n_cvt + 1; d++) {
        double val = pab[t->logdet_diag_rows[d] * ni + t->logdet_diag_cols[d]];
        if (val <= 0.0) return (double)NAN;
        logdet_pab += log(val);
    }
    double logdet_hiw = logdet_pab - logdet_iab;

    /* P_yy guard */
    int nc_total = t->n_cvt + 1;
    double P_yy = pab[nc_total * ni + t->idx_yy];
    if (P_yy < 0.0) {
        P_yy = (double)NAN;
    } else if (P_yy < P_YY_MIN) {
        P_yy = P_YY_MIN;
    }

    return reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy);
}

/* -------------------------------------------------------------------------
 * reml_logl_general_cached — REML using cached grid hi_eval + invariant sums.
 *
 * For cached grid points: invariant sums already computed, just compute
 * varying dot products, reconstruct row0, calc_pab, reml_finish.
 * ------------------------------------------------------------------------- */
static double reml_logl_general_cached(
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
)
{
    int ni = t->n_index;
    int n_var = t->n_var;

    /* Compute varying dot products (reuse tail of row0 as temp) */
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval[i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    /* Reconstruct row 0 */
    for (int i = 0; i < ni; i++) row0[i] = 0.0;
    for (int c = 0; c < t->n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums_cached[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];

    /* Full Pab via recursion */
    calc_pab_general(row0, t, pab_scratch);

    return reml_finish_general(pab_scratch, t, logdet_h, logdet_iab, reml_const);
}

/* -------------------------------------------------------------------------
 * reml_logl_general_fresh — Full REML evaluation for a specific lambda.
 *
 * Computes hi_eval + logdet_h + all dot products in single n_samples pass
 * (fused loop), then calc_pab + reml_finish.
 * Used during golden section refinement where lambda is SNP-specific.
 * ------------------------------------------------------------------------- */
static double reml_logl_general_fresh(
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
)
{
    int ni = t->n_index;
    int n_inv = t->n_inv;
    int n_var = t->n_var;

    double logdet_h = 0.0;
    double inv_sums[MAX_N_INDEX];
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_inv; c++) inv_sums[c] = 0.0;
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        logdet_h += log(v);
        for (int c = 0; c < n_inv; c++)
            inv_sums[c] += h * uab_inv[c * n_samples + i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    /* Reconstruct row 0 */
    for (int i = 0; i < ni; i++) row0[i] = 0.0;
    for (int c = 0; c < n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];

    /* Full Pab via recursion */
    calc_pab_general(row0, t, pab_scratch);

    return reml_finish_general(pab_scratch, t, logdet_h, logdet_iab, reml_const);
}

/* -------------------------------------------------------------------------
 * wald_from_pab_general — Extract Wald stats from general-n_cvt Pab.
 *
 * P_XX = Pab[n_cvt, idx_xx], P_XY = Pab[n_cvt, idx_xy],
 * P_YY = Pab[n_cvt, idx_yy] (pre-genotype-projection),
 * Px_YY = Pab[n_cvt+1, idx_yy] (fully projected).
 * Same Wald formula as existing wald_from_pab.
 * Returns 1 if valid, 0 if degenerate.
 * ------------------------------------------------------------------------- */
static int wald_from_pab_general(
    const double *pab,
    const pab_table_t *t,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    int ni = t->n_index;
    int df = t->df;
    int nc = t->n_cvt;

    double P_XX  = pab[nc * ni + t->idx_xx];
    double P_XY  = pab[nc * ni + t->idx_xy];
    double P_YY  = pab[nc * ni + t->idx_yy];
    double Px_YY = pab[(nc + 1) * ni + t->idx_yy];

    if (Px_YY < 0.0) {
        *beta_out = *se_out = *f_stat_out = (double)NAN;
        return 0;
    }
    if (Px_YY < P_YY_MIN) Px_YY = P_YY_MIN;

    if (P_XX <= 0.0) {
        *beta_out = *se_out = *f_stat_out = (double)NAN;
        return 0;
    }

    double beta = P_XY / P_XX;
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

    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se))
        return 0;

    return 1;
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_general — Grid + golden section for general n_cvt.
 *
 * Mirrors golden_section_lambda_ncvt1_split() structure. Grid phase uses
 * precomputed hi_eval + invariant sums; refinement uses fresh evaluation.
 * At optimal lambda, computes full Pab and returns it + Wald stats.
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_general(
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
)
{
    const double phi = 0.6180339887498949;
    int n_inv = t->n_inv;

    /* Stage 1: coarse grid search using cached invariant sums */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_general_cached(
            inv_sums_grid + (size_t)g * n_inv,
            uab_var,
            hi_eval_grid + (size_t)g * n_samples,
            n_samples,
            logdet_h_grid[g],
            logdet_iab,
            reml_const,
            t,
            row0, pab_scratch
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Fully degenerate SNP */
    if (best_logl == REML_SENTINEL) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return lambda_grid[0];
    }

    /* Bracket around best grid point */
    int idx_low  = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement (fresh evaluation) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, exp(c),
        logdet_iab, reml_const, t, row0, pab_scratch);
    double fd = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, exp(d),
        logdet_iab, reml_const, t, row0, pab_scratch);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_general_fresh(
                uab_inv, uab_var, eigenvalues, n_samples, exp(c),
                logdet_iab, reml_const, t, row0, pab_scratch);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_general_fresh(
                uab_inv, uab_var, eigenvalues, n_samples, exp(d),
                logdet_iab, reml_const, t, row0, pab_scratch);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);

    /* Final: compute REML logl + Pab at optimal lambda for Wald extraction */
    {
        int ni = t->n_index;
        int n_var = t->n_var;

        double logdet_h = 0.0;
        double inv_sums_final[MAX_N_INDEX];
        double var_sums_final[MAX_N_INDEX];
        for (int cc = 0; cc < n_inv; cc++) inv_sums_final[cc] = 0.0;
        for (int cc = 0; cc < n_var; cc++) var_sums_final[cc] = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double v = lambda_opt * eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            logdet_h += log(v);
            for (int cc = 0; cc < n_inv; cc++)
                inv_sums_final[cc] += h * uab_inv[cc * n_samples + i];
            for (int cc = 0; cc < n_var; cc++)
                var_sums_final[cc] += h * uab_var[cc * n_samples + i];
        }

        for (int i = 0; i < ni; i++) row0[i] = 0.0;
        for (int cc = 0; cc < n_inv; cc++)
            row0[t->invariant_indices[cc]] = inv_sums_final[cc];
        for (int cc = 0; cc < n_var; cc++)
            row0[t->varying_indices[cc]] = var_sums_final[cc];

        calc_pab_general(row0, t, pab_scratch);

        *logl_out = reml_finish_general(pab_scratch, t, logdet_h, logdet_iab, reml_const);
        *is_valid_out = wald_from_pab_general(
            pab_scratch, t, beta_out, se_out, f_stat_out);
    }

    return lambda_opt;
}

/* -------------------------------------------------------------------------
 * General workspace struct — persistent cross-chunk state for n_cvt >= 1
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Grid precomputed */
    double *lambda_grid;    /* (n_grid,) */
    double *hi_eval_grid;   /* (n_grid * n_samples) */
    double *logdet_h_grid;  /* (n_grid,) */
    double *inv_sums_grid;  /* (n_grid * n_inv) — precomputed invariant dot products */
    /* Fixed params */
    double *eigenvalues;    /* (n_samples,) — owned copy */
    double reml_const;
    int n_samples, n_grid, n_refine;
    /* Table (owned copy of indices) */
    pab_table_t table;
    /* Iab: invariant identity sums (precomputed, reused per-SNP) */
    double *inv_identity_sums;  /* (n_inv,) — sum of each invariant column at identity */
    /* F-distribution */
    double lbeta_ab;
    double beta_a, beta_b;
    /* Invariant SoA (reference, not owned — Python holds the array) */
    const double *uab_inv;
    PyObject *uab_inv_ref;      /* keeps uab_invariant_soa array alive */
    /* Fused Uab fields (NULL when non-fused general workspace) */
    double *utw_transposed;     /* (n_cvt * n_samples) column-major, owned */
    const double *UtW;          /* points to utw_transposed (column-major) */
    const double *Uty;          /* (n_samples,) borrowed */
    int n_cvt;                  /* stored for loop bounds */
    int *var_a_cols;            /* (n_var,) 0-based column indices. Owned. */
    int *var_b_cols;            /* (n_var,) 0-based column indices. Owned. */
    double *scratch_flat;       /* (actual_threads * n_var * n_samples) owned */
    int actual_threads;         /* for scratch deallocation sizing */
    /* Per-thread heap buffers for Pab recursion (replaces stack arrays) */
    double *pab_per_thread;     /* (actual_threads * pab_size) owned */
    double *row0_per_thread;    /* (actual_threads * n_index) owned */
    int pab_size;               /* n_rows * n_index for this workspace */
    PyObject *Uty_ref;          /* keeps Uty array alive */
    /* Mode-4 fused fields (NULL/0 when Wald-only) */
    int mode;                   /* 0=Wald-only, 4=mode-4 */
    double *hi_eval_null;       /* (n_samples,) owned */
    double logl_H0;
    double mle_const;
    double *null_inv_sums;      /* (n_inv,) precomputed null-model invariant sums. Owned. */
    /* Pre-allocated per-thread LRT buffer for mode-4 fused general.
     * (actual_threads * n_index * n_samples) doubles, row-major per SNP.
     * Avoids per-SNP malloc inside OpenMP loop. NULL when not mode-4. */
    double *uab_snp_flat;
} lmm_workspace_general_t;

/* PyCapsule destructor for general workspace */
static void lmm_workspace_general_destructor(PyObject *cap)
{
    lmm_workspace_general_t *ws =
        (lmm_workspace_general_t *)PyCapsule_GetPointer(cap, "lmm_workspace_general");
    if (!ws) return;
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->inv_sums_grid);
    free(ws->eigenvalues);
    free(ws->inv_identity_sums);
    free(ws->table.invariant_indices);
    free(ws->table.varying_indices);
    free(ws->table.logdet_diag_rows);
    free(ws->table.logdet_diag_cols);
    free(ws->table.level_offsets);
    free(ws->table.level_counts);
    free(ws->table.entries);
    Py_XDECREF(ws->uab_inv_ref);
    /* Fused general fields */
    free(ws->utw_transposed);
    free(ws->var_a_cols);
    free(ws->var_b_cols);
    free(ws->scratch_flat);
    free(ws->pab_per_thread);
    free(ws->row0_per_thread);
    Py_XDECREF(ws->Uty_ref);
    /* Mode-4 fused fields */
    free(ws->hi_eval_null);
    free(ws->null_inv_sums);
    free(ws->uab_snp_flat);
    free(ws);
}


/* -------------------------------------------------------------------------
 * create_workspace_general_c
 *
 * Python signature:
 *   create_workspace_general_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (n_inv, n_samples) float64 — SoA
 *       n_samples,        # int
 *       l_min, l_max,     # float
 *       n_grid, n_refine, n_threads,  # int
 *       n_cvt,            # int
 *       invariant_indices, varying_indices,    # (n_inv,) / (n_var,) int32
 *       logdet_diag_rows, logdet_diag_cols,    # (n_cvt+1,) int32
 *       level_offsets, level_counts,           # (n_rows,) int32
 *       entries,           # (n_entries * 4,) int32 stride-4 flat
 *       idx_xx, idx_xy, idx_yy,  # int
 *   ) -> PyCapsule wrapping lmm_workspace_general_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "n_samples",
        "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "n_cvt",
        "invariant_indices", "varying_indices",
        "logdet_diag_rows", "logdet_diag_cols",
        "level_offsets", "level_counts", "entries",
        "idx_xx", "idx_xy", "idx_yy",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj;
    PyObject *inv_idx_obj, *var_idx_obj;
    PyObject *diag_rows_obj, *diag_cols_obj;
    PyObject *offsets_obj, *counts_obj, *entries_obj;
    int n_samples, n_grid, n_refine, n_threads, n_cvt;
    int idx_xx, idx_xy, idx_yy;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOiddiiiiOOOOOOOiii", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &n_cvt,
            &inv_idx_obj, &var_idx_obj,
            &diag_rows_obj, &diag_cols_obj,
            &offsets_obj, &counts_obj, &entries_obj,
            &idx_xx, &idx_xy, &idx_yy)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError,
            "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }

    int n_index = (n_cvt + 3) * (n_cvt + 2) / 2;
    int n_rows  = n_cvt + 2;

    /* Parse invariant_indices to determine n_inv */
    PyArrayObject *inv_idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
        inv_idx_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!inv_idx_arr) return NULL;
    int n_inv = (int)PyArray_SIZE(inv_idx_arr);
    Py_DECREF(inv_idx_arr);

    PyArrayObject *var_idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
        var_idx_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!var_idx_arr) return NULL;
    int n_var = (int)PyArray_SIZE(var_idx_arr);
    Py_DECREF(var_idx_arr);

    if (n_inv + n_var != n_index) {
        PyErr_Format(PyExc_ValueError,
            "n_inv (%d) + n_var (%d) != n_index (%d)", n_inv, n_var, n_index);
        return NULL;
    }

    /* Parse entries to get total count */
    PyArrayObject *entries_arr = (PyArrayObject *)PyArray_FROM_OTF(
        entries_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!entries_arr) return NULL;
    int entries_len = (int)PyArray_SIZE(entries_arr);
    Py_DECREF(entries_arr);
    if (entries_len % 4 != 0) {
        PyErr_Format(PyExc_ValueError,
            "entries length (%d) not a multiple of 4", entries_len);
        return NULL;
    }
    int n_entries = entries_len / 4;

    /* Convert eigenvalues and uab_invariant */
    PyArrayObject *eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0) {
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }

    PyArrayObject *uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) { Py_DECREF(eigenvalues_arr); return NULL; }

    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != n_inv ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant must be shape (%d, %d)", n_inv, n_samples);
        Py_DECREF(eigenvalues_arr);
        Py_DECREF(uab_inv_arr);
        return NULL;
    }

    /* Allocate workspace */
    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)calloc(
        1, sizeof(lmm_workspace_general_t));
    if (!ws) {
        PyErr_NoMemory();
        Py_DECREF(eigenvalues_arr);
        Py_DECREF(uab_inv_arr);
        return NULL;
    }

    ws->n_samples = n_samples;
    ws->n_grid = n_grid;
    ws->n_refine = n_refine;

    /* Fill table */
    ws->table.n_cvt = n_cvt;
    ws->table.n_index = n_index;
    ws->table.n_rows = n_rows;
    ws->table.n_inv = n_inv;
    ws->table.n_var = n_var;
    ws->table.idx_xx = idx_xx;
    ws->table.idx_xy = idx_xy;
    ws->table.idx_yy = idx_yy;
    ws->table.df = n_samples - n_cvt - 1;
    ws->table.n_entries = n_entries;

    /* Parse index arrays into owned copies — check each immediately to
     * avoid calling Python/C API with a live exception set. */
    ws->table.invariant_indices = parse_int32_array(inv_idx_obj, n_inv, "invariant_indices");
    if (!ws->table.invariant_indices) goto err_ws;
    ws->table.varying_indices   = parse_int32_array(var_idx_obj, n_var, "varying_indices");
    if (!ws->table.varying_indices) goto err_ws;
    ws->table.logdet_diag_rows  = parse_int32_array(diag_rows_obj, n_cvt + 1, "logdet_diag_rows");
    if (!ws->table.logdet_diag_rows) goto err_ws;
    ws->table.logdet_diag_cols  = parse_int32_array(diag_cols_obj, n_cvt + 1, "logdet_diag_cols");
    if (!ws->table.logdet_diag_cols) goto err_ws;
    ws->table.level_offsets     = parse_int32_array(offsets_obj, n_rows, "level_offsets");
    if (!ws->table.level_offsets) goto err_ws;
    ws->table.level_counts      = parse_int32_array(counts_obj, n_rows, "level_counts");
    if (!ws->table.level_counts) goto err_ws;

    /* Parse entries (stride-4) into pab_entry_t array */
    {
        int *raw_entries = parse_int32_array(entries_obj, n_entries * 4, "entries");
        if (!raw_entries) goto err_ws;
        ws->table.entries = (pab_entry_t *)malloc(
            (size_t)n_entries * sizeof(pab_entry_t));
        if (!ws->table.entries) {
            free(raw_entries);
            PyErr_NoMemory();
            goto err_ws;
        }
        for (int i = 0; i < n_entries; i++) {
            ws->table.entries[i].index_ab = raw_entries[i * 4 + 0];
            ws->table.entries[i].index_aw = raw_entries[i * 4 + 1];
            ws->table.entries[i].index_bw = raw_entries[i * 4 + 2];
            ws->table.entries[i].index_ww = raw_entries[i * 4 + 3];
        }
        free(raw_entries);
    }

    /* Validate all table indices are within [0, n_index) to prevent
     * out-of-bounds access in the OpenMP parallel loop. A bug in
     * build_pab_table_for_c() or a stale @lru_cache entry could produce
     * invalid indices that corrupt stack buffers silently. */
    for (int i = 0; i < n_inv; i++) {
        if (ws->table.invariant_indices[i] < 0 ||
            ws->table.invariant_indices[i] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "invariant_indices[%d] = %d out of range [0, %d)",
                i, ws->table.invariant_indices[i], n_index);
            goto err_ws;
        }
    }
    for (int i = 0; i < n_var; i++) {
        if (ws->table.varying_indices[i] < 0 ||
            ws->table.varying_indices[i] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "varying_indices[%d] = %d out of range [0, %d)",
                i, ws->table.varying_indices[i], n_index);
            goto err_ws;
        }
    }
    for (int d = 0; d < n_cvt + 1; d++) {
        if (ws->table.logdet_diag_rows[d] < 0 ||
            ws->table.logdet_diag_rows[d] >= n_rows) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_rows[%d] = %d out of range [0, %d)",
                d, ws->table.logdet_diag_rows[d], n_rows);
            goto err_ws;
        }
        if (ws->table.logdet_diag_cols[d] < 0 ||
            ws->table.logdet_diag_cols[d] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_cols[%d] = %d out of range [0, %d)",
                d, ws->table.logdet_diag_cols[d], n_index);
            goto err_ws;
        }
    }
    /* Validate level_offsets + level_counts don't exceed n_entries */
    for (int p = 0; p < n_rows; p++) {
        if (ws->table.level_offsets[p] < 0 ||
            ws->table.level_counts[p] < 0 ||
            (int64_t)ws->table.level_offsets[p] + ws->table.level_counts[p] > n_entries) {
            PyErr_Format(PyExc_ValueError,
                "level_offsets[%d]=%d + level_counts[%d]=%d exceeds n_entries=%d",
                p, ws->table.level_offsets[p], p, ws->table.level_counts[p], n_entries);
            goto err_ws;
        }
    }
    if (idx_xx < 0 || idx_xx >= n_index ||
        idx_xy < 0 || idx_xy >= n_index ||
        idx_yy < 0 || idx_yy >= n_index) {
        PyErr_SetString(PyExc_ValueError, "idx_xx/xy/yy out of range [0, n_index)");
        goto err_ws;
    }
    for (int i = 0; i < n_entries; i++) {
        const pab_entry_t *e = &ws->table.entries[i];
        if (e->index_ab < 0 || e->index_ab >= n_index ||
            e->index_aw < 0 || e->index_aw >= n_index ||
            e->index_bw < 0 || e->index_bw >= n_index ||
            e->index_ww < 0 || e->index_ww >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "entries[%d] has index out of range [0, %d)", i, n_index);
            goto err_ws;
        }
    }

    /* Copy eigenvalues (owned) */
    ws->eigenvalues = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!ws->eigenvalues) { PyErr_NoMemory(); goto err_ws; }
    memcpy(ws->eigenvalues, PyArray_DATA(eigenvalues_arr),
           (size_t)n_samples * sizeof(double));

    /* Borrow invariant Uab pointer — keep alive via Py_INCREF */
    Py_INCREF(uab_inv_arr);
    ws->uab_inv_ref = (PyObject *)uab_inv_arr;
    ws->uab_inv = (const double *)PyArray_DATA(uab_inv_arr);

    /* Compute df, reml_const, beta params */
    int df = ws->table.df;
    ws->beta_a = (double)df / 2.0;
    ws->beta_b = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);
    ws->reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Build lambda grid */
    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

    ws->lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->inv_sums_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_inv * sizeof(double));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->inv_sums_grid) {
        PyErr_NoMemory();
        goto err_ws;
    }

    for (int g = 0; g < n_grid; g++)
        ws->lambda_grid[g] = exp(log_l_min + g * step);

    /* Precompute hi_eval_grid, logdet_h_grid, and invariant sums */
    for (int g = 0; g < n_grid; g++) {
        double lam = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;

        /* First pass: compute hi_eval + logdet_h */
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
        }
        ws->logdet_h_grid[g] = logdet;

        /* Compute invariant dot products for this grid point */
        double *inv_sums = ws->inv_sums_grid + (size_t)g * n_inv;
        for (int c = 0; c < n_inv; c++) {
            double s = 0.0;
            const double *col = ws->uab_inv + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += hi_row[i] * col[i];
            inv_sums[c] = s;
        }
    }

    /* Precompute invariant identity sums (sum of each invariant Uab column
     * at identity, i.e. lambda=0, hi=1). These are constant across SNPs
     * and reused in the per-SNP logdet_iab computation. The varying identity
     * sums are SNP-dependent (genotype cross-products), so logdet_iab must
     * be computed per-SNP in compute_lmm_chunk_general_c. */
    ws->inv_identity_sums = (double *)malloc((size_t)n_inv * sizeof(double));
    if (!ws->inv_identity_sums) { PyErr_NoMemory(); goto err_ws; }
    for (int c = 0; c < n_inv; c++) {
        double s = 0.0;
        const double *col = ws->uab_inv + (size_t)c * n_samples;
        for (int i = 0; i < n_samples; i++)
            s += col[i];
        ws->inv_identity_sums[c] = s;
    }
    /* Wrap in PyCapsule */
    PyObject *capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return capsule;

err_ws:
    if (ws) {
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->inv_sums_grid);
        free(ws->eigenvalues);
        free(ws->inv_identity_sums);
        free(ws->table.invariant_indices);
        free(ws->table.varying_indices);
        free(ws->table.logdet_diag_rows);
        free(ws->table.logdet_diag_cols);
        free(ws->table.level_offsets);
        free(ws->table.level_counts);
        free(ws->table.entries);
        Py_XDECREF(ws->uab_inv_ref);
        free(ws);
    }
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_general_c
 *
 * Per-chunk compute using a pre-built general workspace. OpenMP parallel
 * over SNPs. Each thread has its own stack-allocated Pab buffers.
 *
 * Python signature:
 *   compute_lmm_chunk_general_c(
 *       workspace,      # PyCapsule from create_workspace_general_c
 *       uab_varying,    # (n_snps, n_var, n_samples) float64 — SoA
 *       n_threads,      # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_chunk_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "uab_varying", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *uab_var_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &uab_var_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace_general");
    if (!ws) return NULL;

    PyArrayObject *uab_var_arr = NULL;
    output_arrays_t out = {0};
    PyObject *result = NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) return NULL;

    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_inv = ws->table.n_inv;

    /* Validate shape: (n_snps, n_var, n_samples) */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != n_var ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying must be shape (n_snps, %d, %d)", n_var, n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);

    double *lambdas = (double *)PyArray_DATA(out.lambdas);
    double *logls   = (double *)PyArray_DATA(out.logls);
    double *betas   = (double *)PyArray_DATA(out.betas);
    double *ses     = (double *)PyArray_DATA(out.ses);
    double *pwalds  = (double *)PyArray_DATA(out.pwalds);

    int n_grid = ws->n_grid;
    int n_refine = ws->n_refine;
    int df = ws->table.df;
    int n_index = ws->table.n_index;
    double reml_const = ws->reml_const;

    /* Compute log_l_min and step from lambda_grid */
    double log_l_min = log(ws->lambda_grid[0]);
    double step = (n_grid > 1)
        ? (log(ws->lambda_grid[n_grid - 1]) - log_l_min) / (double)(n_grid - 1)
        : 0.0;

    /* Clamp n_threads */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    /* Allocate per-thread heap buffers for Pab recursion */
    int pab_size = ws->table.n_rows * n_index;
    double *pab_heap = (double *)malloc(
        (size_t)actual_threads * (size_t)pab_size * sizeof(double));
    double *row0_heap = (double *)malloc(
        (size_t)actual_threads * (size_t)n_index * sizeof(double));
    if (!pab_heap || !row0_heap) {
        free(pab_heap); free(row0_heap);
        decref_output_arrays(&out);
        Py_DECREF(uab_var_arr);
        PyErr_NoMemory();
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    /* Static schedule: per-SNP cost is uniform (same table, n_grid,
     * n_refine, n_samples).  Matches n_cvt=1 split path rationale. */
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *my_pab = pab_heap + (size_t)tid * pab_size;
        double *my_row0 = row0_heap + (size_t)tid * n_index;

        const double *snp_var = uab_var_data +
            (size_t)snp * n_var * n_samples;

        /* Compute per-SNP logdet_iab at identity (lambda=0, hi=1).
         * Row 0: identity-weighted sums = simple column sums.
         * Invariant sums are precomputed in workspace; only varying
         * sums (genotype-dependent) need per-SNP computation.
         * Reuse per-thread heap buffer (consumed before my_row0 needed). */
        double *iab_row0 = my_row0;
        for (int i = 0; i < n_index; i++) iab_row0[i] = 0.0;

        /* Invariant identity sums from precomputed workspace */
        for (int c = 0; c < n_inv; c++)
            iab_row0[ws->table.invariant_indices[c]] = ws->inv_identity_sums[c];
        /* Varying identity sums from this SNP's uab_var */
        for (int c = 0; c < n_var; c++) {
            double s = 0.0;
            const double *col = snp_var + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += col[i];
            iab_row0[ws->table.varying_indices[c]] = s;
        }

        /* Compute logdet_iab via helper */
        double logdet_iab = logdet_from_row0(
            iab_row0, &ws->table, ws->table.n_cvt, my_pab);

        /* Golden section optimization */
        double logl_opt, beta, se, f_stat;
        int is_valid;
        double lambda_opt = golden_section_lambda_general(
            ws->uab_inv, snp_var, ws->eigenvalues,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->inv_sums_grid,
            log_l_min, step, n_grid, n_refine,
            logdet_iab, reml_const, &ws->table,
            &logl_opt, &beta, &se, &f_stat, &is_valid,
            my_row0, my_pab
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;
        pwalds[snp]  = f_to_pvalue(
            f_stat, df, is_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);
    }

    Py_END_ALLOW_THREADS
    free(pab_heap);
    free(row0_heap);

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(uab_var_arr);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(uab_var_arr);
    return NULL;
}


/* =========================================================================
 * Python entry points
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_lmm_batch_split_c
 *
 * Python-callable entry point for the split-Uab path.
 *
 * Python signature:
 *   compute_lmm_batch_split_c(
 *       eigenvalues,   # (n_samples,) float64
 *       uab_varying,   # (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]
 *       uab_invariant, # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       Iab_batch,     # (n_snps, 3, 6) float64
 *       n_samples,     # int
 *       l_min,         # float
 *       l_max,         # float
 *       n_grid,        # int
 *       n_refine,      # int
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_batch_split_c(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_varying", "uab_invariant", "Iab_batch",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_var_obj, *uab_inv_obj, *iab_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_var_obj, &uab_inv_obj, &iab_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    /* Locals for goto cleanup — must be declared before any goto target */
    PyArrayObject *eigenvalues_arr = NULL, *uab_var_arr = NULL;
    PyArrayObject *uab_inv_arr = NULL, *iab_arr = NULL;
    output_arrays_t out = {0};
    double *lambda_grid = NULL, *hi_eval_grid = NULL, *logdet_h_grid = NULL;
    grid_invariant_t *grid_inv = NULL;
    PyObject *result = NULL;

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) goto err_input;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    iab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        iab_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!iab_arr) goto err_input;

    /* Validate shapes — SoA layout: (n_snps, 3, n_samples) and (3, n_samples) */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_varying must be shape (n_snps, 3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(iab_arr) != 3 ||
        PyArray_DIM(iab_arr, 1) != 3 ||
        PyArray_DIM(iab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch must be shape (n_snps, 3, 6)");
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX",
            n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (PyArray_DIM(iab_arr, 0) != n_snps) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch.shape[0] must match uab_varying.shape[0]");
        goto err_input;
    }

    /* Allocate output arrays */
    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    /* Raw pointers — SoA layout */
    const double *eigenvalues_data = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_var_data     = (const double *)PyArray_DATA(uab_var_arr);
    const double *uab_inv_data     = (const double *)PyArray_DATA(uab_inv_arr);
    const double *iab_data         = (const double *)PyArray_DATA(iab_arr);

    /* SoA invariant column pointers: (3, n_samples) layout */
    const double *inv_ww = uab_inv_data;
    const double *inv_wy = uab_inv_data + (size_t)n_samples;
    const double *inv_yy = uab_inv_data + (size_t)2 * n_samples;

    /* Reject non-finite eigenvalues before entering compute loop */
    if (validate_eigenvalues(eigenvalues_data, n_samples) < 0)
        goto err_output;

    double *lambdas  = (double *)PyArray_DATA(out.lambdas);
    double *logls    = (double *)PyArray_DATA(out.logls);
    double *betas    = (double *)PyArray_DATA(out.betas);
    double *ses      = (double *)PyArray_DATA(out.ses);
    double *pwalds   = (double *)PyArray_DATA(out.pwalds);

    int df = n_samples - 2;
    double beta_a = (double)df / 2.0;
    double beta_b = 0.5;
    double lbeta_ab = lgamma(beta_a) + lgamma(beta_b) - lgamma(beta_a + beta_b);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);
    double reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Pre-build lambda grid */
    lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) { PyErr_NoMemory(); goto err_output; }
    for (int g = 0; g < n_grid; g++) {
        lambda_grid[g] = exp(log_l_min + g * step);
    }

    /* Precompute coarse-grid hi_eval, logdet_h, and invariant dot products */
    hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid_inv = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!hi_eval_grid || !logdet_h_grid || !grid_inv) {
        PyErr_NoMemory();
        goto err_output;
    }

    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi_row = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues_data[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);  /* v > 1.0: lambda > 0, eval >= 0 */

            sw  += h * inv_ww[i];
            swy += h * inv_wy[i];
            sy  += h * inv_yy[i];
        }
        logdet_h_grid[g] = logdet;

        grid_inv[g].s_ww = sw;
        grid_inv[g].s_wy = swy;
        grid_inv[g].s_yy = sy;
        grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
    }

    /* Thread setup — no per-thread hi_eval buffers needed for split path
     * since reml_logl_ncvt1_split fuses hi_eval into the dot product loop */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    Py_BEGIN_ALLOW_THREADS

    /* Static schedule: uniform SNP cost — no work-stealing overhead */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        /* SoA: (n_snps, 3, n_samples) — 3 contiguous columns per SNP */
        const double *snp_base = uab_var_data + (size_t)snp * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;
        const double *iab = iab_data + (size_t)snp * 3 * 6;

        double logdet_iab = compute_logdet_iab(iab);

        double lambda_opt, logl_opt, beta, se, f_stat;
        int is_valid;
        lambda_opt = golden_section_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            eigenvalues_data, logdet_iab,
            n_samples, lambda_grid, hi_eval_grid, logdet_h_grid,
            grid_inv, log_l_min, step, n_grid, n_refine,
            df, reml_const, &logl_opt,
            &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;

        betas[snp] = beta;
        ses[snp]   = se;

        pwalds[snp] = f_to_pvalue(f_stat, df, is_valid,
                                   beta_a, beta_b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_var_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    free(grid_inv);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    free(grid_inv);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_batch_c
 *
 * Python-callable entry point.
 *
 * Python signature:
 *   compute_lmm_batch_c(
 *       eigenvalues,   # (n_samples,) float64
 *       Uab_batch,     # (n_snps, n_samples, 6) float64
 *       Iab_batch,     # (n_snps, 3, 6) float64
 *       n_samples,     # int
 *       l_min,         # float
 *       l_max,         # float
 *       n_grid,        # int
 *       n_refine,      # int
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 *
 * p-values are computed C-side via betainc (Lentz CF) for full end-to-end
 * acceleration. The output key is 'pwalds' (not 'f_stats' as in v1).
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_batch_c(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "Uab_batch", "Iab_batch",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_obj, *iab_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_obj, &iab_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    /* Locals for goto cleanup — must be declared before any goto target */
    PyArrayObject *eigenvalues_arr = NULL, *uab_arr = NULL, *iab_arr = NULL;
    output_arrays_t out = {0};
    double *lambda_grid = NULL, *hi_eval_grid = NULL, *logdet_h_grid = NULL;
    double **thread_bufs = NULL;
    int actual_threads = 1;
    PyObject *result = NULL;

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) goto err_input;

    iab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        iab_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!iab_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples ||
        PyArray_DIM(uab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, 6)");
        goto err_input;
    }
    if (PyArray_NDIM(iab_arr) != 3 ||
        PyArray_DIM(iab_arr, 1) != 3 ||
        PyArray_DIM(iab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch must be shape (n_snps, 3, 6)");
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (PyArray_DIM(iab_arr, 0) != n_snps) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch.shape[0] must match Uab_batch.shape[0] (n_snps)");
        goto err_input;
    }

    /* Allocate output arrays */
    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    /* Raw pointers */
    const double *eigenvalues_data = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_data         = (const double *)PyArray_DATA(uab_arr);
    const double *iab_data         = (const double *)PyArray_DATA(iab_arr);

    /* Reject non-finite eigenvalues before entering compute loop */
    if (validate_eigenvalues(eigenvalues_data, n_samples) < 0)
        goto err_output;

    double *lambdas  = (double *)PyArray_DATA(out.lambdas);
    double *logls    = (double *)PyArray_DATA(out.logls);
    double *betas    = (double *)PyArray_DATA(out.betas);
    double *ses      = (double *)PyArray_DATA(out.ses);
    double *pwalds   = (double *)PyArray_DATA(out.pwalds);

    int df = n_samples - 2;
    double beta_a = (double)df / 2.0;
    double beta_b = 0.5;
    double lbeta_ab = lgamma(beta_a) + lgamma(beta_b) - lgamma(beta_a + beta_b);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);
    double reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Pre-build lambda grid */
    lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) { PyErr_NoMemory(); goto err_output; }
    for (int g = 0; g < n_grid; g++) {
        lambda_grid[g] = exp(log_l_min + g * step);
    }

    /* Precompute coarse-grid hi_eval and logdet_h.
     *
     * hi_eval_grid: (n_grid * n_samples) — hi_eval[g][i] = 1/(lambda_grid[g]*eval[i]+1)
     * logdet_h_grid: (n_grid)            — sum of log(lambda_grid[g]*eval[i]+1) per grid point
     *
     * These are identical across all SNPs (eigenvalues are shared) so we compute
     * them once here instead of n_snps * n_grid times inside the parallel loop.
     * Memory: n_grid * n_samples * 8 bytes (e.g. 50 * 50k = 20 MB). */
    hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));

    if (!hi_eval_grid || !logdet_h_grid) { PyErr_NoMemory(); goto err_output; }

    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi_row = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        #pragma omp simd reduction(+:logdet)
        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues_data[i] + 1.0;
            hi_row[i] = 1.0 / v;
            logdet += log(v);  /* v > 1.0: lambda > 0, eval >= 0 */
        }
        logdet_h_grid[g] = logdet;
    }

    /* Pre-allocate per-thread hi_eval buffers OUTSIDE the parallel region.
     * Each thread reuses one buffer across all its SNPs, eliminating the
     * malloc/free-per-SNP that caused heap lock contention with 48 threads. */
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    if (!thread_bufs) {
        PyErr_NoMemory();
        goto err_output;
    }

    Py_BEGIN_ALLOW_THREADS

    /* Static schedule: uniform SNP cost — no work-stealing overhead */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        /* Per-thread buffer — no malloc inside the hot loop */
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval = thread_bufs[tid];

        const double *uab = uab_data + (size_t)snp * n_samples * 6;
        const double *iab = iab_data + (size_t)snp * 3 * 6;

        double logdet_iab = compute_logdet_iab(iab);

        double lambda_opt, logl_opt;
        double beta, se, f_stat;
        int is_valid;
        lambda_opt = golden_section_lambda_ncvt1(
            uab, eigenvalues_data, logdet_iab, n_samples,
            lambda_grid, hi_eval_grid, logdet_h_grid,
            log_l_min, step, n_grid, n_refine,
            df, reml_const,
            hi_eval, &logl_opt,
            &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;

        pwalds[snp] = f_to_pvalue(f_stat, df, is_valid, beta_a, beta_b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0) {
        free_thread_scratch(thread_bufs, actual_threads);
        thread_bufs = NULL;
        goto err_output;
    }

    free_thread_scratch(thread_bufs, actual_threads);
    thread_bufs = NULL;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_arr);
    Py_DECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    free_thread_scratch(thread_bufs, actual_threads);
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_arr);
    Py_XDECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    return NULL;
}

/* =========================================================================
 * MLE (Maximum Likelihood Estimation) helpers for LRT
 *
 * Key differences from REML:
 *   - No Iab parameter or logdet_hiw computation
 *   - Uses n_samples instead of df = n_samples - n_cvt - 1
 *   - MLE constant: 0.5 * n * (log(n) - log(2*pi) - 1)
 * ========================================================================= */

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
static double mle_logl_ncvt1(
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
static double mle_logl_ncvt1_cached(
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
static double golden_section_lambda_mle_ncvt1(
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
static double mle_logl_ncvt1_cached_split(
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

/* Return the best MLE coarse-grid index, or -1 when every point is degenerate. */
static int coarse_grid_mle_ncvt1_split(
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

/* Find the REML and MLE coarse brackets together. Both likelihoods consume
 * one canonical Pab calculation per grid point and differ only in the tail. */
static void coarse_grid_mode4_ncvt1_split(
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
static double mle_logl_ncvt1_split(
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
static double refine_lambda_mle_ncvt1_split(
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

/* Full MLE optimization for callers that do not share the coarse-grid pass. */
static double golden_section_lambda_mle_ncvt1_split(
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

/* =========================================================================
 * LRT BATCH — compute_lrt_batch_c
 *
 * LRT test with per-SNP MLE optimization + chi2_sf p-value.
 * ========================================================================= */


/* -------------------------------------------------------------------------
 * compute_lrt_batch_c
 *
 * Batch LRT for n_cvt=1 with optional OpenMP.
 * Per-SNP MLE golden section optimization + chi2_sf for p-value.
 *
 * Args: eigenvalues (n_samples,), Uab_batch (n_snps, n_samples, 6),
 *       n_samples, l_min, l_max, n_grid, n_refine, logl_H0, n_threads
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_batch_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_batch_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTuple(args, "OOiddiidi",
            &eigenvalues_obj, &uab_batch_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    PyArrayObject *eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    PyArrayObject *uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_batch_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) { Py_DECREF(eigenvalues_arr); return NULL; }

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples ||
        PyArray_DIM(uab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, 6)");
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }

    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_batch = (const double *)PyArray_DATA(uab_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0) {
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }

    /* Allocate output arrays */
    lrt_output_t out;
    if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }

    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    /* Pre-compute MLE constant and grid */
    double n = (double)n_samples;
    double mle_const = 0.5 * n * (log(n) - log(2.0 * M_PI) - 1.0);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

    /* Build lambda grid */
    double *lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) {
        decref_lrt_output(&out);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }
    for (int g = 0; g < n_grid; g++)
        lambda_grid[g] = exp(log_l_min + g * step);

    /* Pre-compute hi_eval_grid and logdet_h_grid (shared across SNPs) */
    double *hi_eval_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_samples * sizeof(double));
    double *logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!hi_eval_grid || !logdet_h_grid) {
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        decref_lrt_output(&out);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }
    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues[i] + 1.0;
            hi[i] = 1.0 / v;
            logdet += log(v);
        }
        logdet_h_grid[g] = logdet;
    }

    /* Pre-allocate per-thread hi_eval buffers OUTSIDE the parallel region.
     * Matches the Wald batch pattern — fail hard with PyErr_NoMemory rather
     * than silently producing NaN from inside an OpenMP parallel region. */
    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    if (!thread_bufs) {
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        decref_lrt_output(&out);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];
        const double *uab = uab_batch + (size_t)s * n_samples * 6;

        double logl_H1;
        double lam_mle = golden_section_lambda_mle_ncvt1(
            uab, eigenvalues, n_samples,
            lambda_grid, hi_eval_grid, logdet_h_grid,
            log_l_min, step, n_grid, n_refine,
            mle_const, hi_eval_local, &logl_H1
        );
        out_lambdas_mle[s] = lam_mle;

        /* LRT stat = 2*(logl_H1 - logl_H0), clamp >= 0.
         * NaN logl_H1 (degenerate SNP): NaN-finite=NaN, NaN<0 is false
         * (IEEE 754), chi2_sf_c(NaN) returns NaN. Correct: degenerate
         * SNPs get NaN p-value. */
        double lrt_stat = 2.0 * (logl_H1 - logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[s] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    free_thread_scratch(thread_bufs, actual_threads);

    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    Py_DECREF(uab_arr);
    Py_DECREF(eigenvalues_arr);

    return build_lrt_result_dict(&out);
}

/* =========================================================================
 * SCORE BATCH — compute_score_batch_c
 *
 * Score test with fixed null-model Hi_eval shared across all SNPs.
 * No per-SNP lambda optimization — just Pab computation + F-test.
 * ========================================================================= */


/* -------------------------------------------------------------------------
 * compute_score_batch_c
 *
 * Batch Score test for n_cvt=1 with optional OpenMP.
 *
 * Args: eigenvalues (n_samples,), Uab_batch (n_snps, n_samples, 6),
 *       Hi_eval_null (n_samples,), n_samples, n_threads
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_batch_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_batch_obj, *hi_eval_null_obj;
    int n_samples, n_threads;
    PyArrayObject *eigenvalues_arr = NULL, *uab_arr = NULL, *hi_eval_null_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOii",
            &eigenvalues_obj, &uab_batch_obj, &hi_eval_null_obj,
            &n_samples, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input;

    uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_batch_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) goto err_input;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples ||
        PyArray_DIM(uab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, 6)");
        goto err_input;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Hi_eval_null must be shape (n_samples,)");
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input;
    }

    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_batch = (const double *)PyArray_DATA(uab_arr);
    const double *hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0)
        goto err_input;

    /* Validate Hi_eval_null for NaN/Inf and non-positive values */
    for (int i = 0; i < n_samples; i++) {
        char buf[64];
        if (!isfinite(hi_eval_null[i])) {
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not finite. "
                "Null model optimization may have failed.", i, buf);
            goto err_input;
        }
        if (hi_eval_null[i] <= 0.0) {
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not positive. "
                "Check kinship matrix conditioning.",
                i, buf);
            goto err_input;
        }
    }

    /* Allocate output arrays */
    score_output_t out;
    if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        goto err_input;
    }

    double *out_betas    = (double *)PyArray_DATA(out.betas);
    double *out_ses      = (double *)PyArray_DATA(out.ses);
    double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

    /* Pre-compute F-distribution constants (betainc) */
    int df = n_samples - 2;  /* n_cvt=1: df = n - n_cvt - 1 */
    double a = (double)df / 2.0;  /* same convention as Wald: a=df/2, b=0.5 */
    double b = 0.5;
    double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);

    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        const double *uab = uab_batch + (size_t)s * n_samples * 6;

        double pab[3][6];
        calc_pab_ncvt1(uab, hi_eval_null, n_samples, pab);

        double beta, se, f_stat;
        int is_valid = score_from_pab(pab, n_samples, df, &beta, &se, &f_stat);

        out_betas[s] = beta;
        out_ses[s] = se;
        out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
        decref_score_output(&out);
        Py_DECREF(hi_eval_null_arr);
        Py_DECREF(uab_arr);
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }

    Py_DECREF(hi_eval_null_arr);
    Py_DECREF(uab_arr);
    Py_DECREF(eigenvalues_arr);
    return build_score_result_dict(&out);

err_input:
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(uab_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * GENERAL n_cvt SCORE BATCH — compute_score_batch_general_c
 *
 * Score test for arbitrary n_cvt using table-driven Pab recursion.
 * Mirrors compute_score_batch_c (n_cvt=1) but uses calc_pab_general.
 * ========================================================================= */

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
static int score_from_pab_general(
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


/* -------------------------------------------------------------------------
 * compute_score_batch_general_c
 *
 * Batch Score test for arbitrary n_cvt.
 *
 * Args: eigenvalues (n_samples,), Uab_batch (n_snps, n_samples, n_index),
 *       Hi_eval_null (n_samples,), n_samples, n_cvt, pab_table_dict, n_threads
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_batch_general_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_batch_obj, *hi_eval_null_obj, *pab_table_dict;
    int n_samples, n_cvt, n_threads;
    PyArrayObject *eigenvalues_arr = NULL, *uab_arr = NULL, *hi_eval_null_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOiiOi",
            &eigenvalues_obj, &uab_batch_obj, &hi_eval_null_obj,
            &n_samples, &n_cvt,
            &pab_table_dict, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }
    if (!PyDict_Check(pab_table_dict)) {
        PyErr_SetString(PyExc_TypeError, "pab_table_dict must be a dict");
        return NULL;
    }

    /* Convert inputs */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_score_gen;

    uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_batch_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) goto err_input_score_gen;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input_score_gen;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        goto err_input_score_gen;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Hi_eval_null must be shape (n_samples,)");
        goto err_input_score_gen;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, n_index)");
        goto err_input_score_gen;
    }

    {
        npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
        if (n_snps_raw > INT_MAX || n_snps_raw == 0) {
            PyErr_SetString(PyExc_ValueError, "n_snps must be > 0 and <= INT_MAX");
            goto err_input_score_gen;
        }
        int n_snps = (int)n_snps_raw;
        int n_index_arr = (int)PyArray_DIM(uab_arr, 2);

        /* Parse pab_table from dict */
        pab_table_t table;
        if (parse_pab_table_from_dict(pab_table_dict, &table, n_samples) < 0)
            goto err_input_score_gen;

        if (table.n_index != n_index_arr) {
            PyErr_Format(PyExc_ValueError,
                "Uab_batch n_index=%d doesn't match pab_table n_index=%d",
                n_index_arr, table.n_index);
            free_pab_table(&table);
            goto err_input_score_gen;
        }

        const double *eigenvalues  = (const double *)PyArray_DATA(eigenvalues_arr);
        const double *uab_batch    = (const double *)PyArray_DATA(uab_arr);
        const double *hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);

        if (validate_eigenvalues(eigenvalues, n_samples) < 0) {
            free_pab_table(&table);
            goto err_input_score_gen;
        }

        /* Validate Hi_eval_null for NaN/Inf and non-positive values */
        for (int i = 0; i < n_samples; i++) {
            if (!isfinite(hi_eval_null[i])) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not finite.", i, buf);
                free_pab_table(&table);
                goto err_input_score_gen;
            }
            if (hi_eval_null[i] <= 0.0) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not positive. "
                    "Check kinship matrix conditioning.",
                    i, buf);
                free_pab_table(&table);
                goto err_input_score_gen;
            }
        }

        /* Allocate outputs */
        score_output_t out;
        if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
            free_pab_table(&table);
            PyErr_NoMemory();
            goto err_input_score_gen;
        }

        double *out_betas    = (double *)PyArray_DATA(out.betas);
        double *out_ses      = (double *)PyArray_DATA(out.ses);
        double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

        /* F-distribution constants (shared across all SNPs) */
        int df = n_samples - n_cvt - 1;
        double a = (double)df / 2.0;
        double b = 0.5;
        double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);

        int actual_threads = 1;
#ifdef _OPENMP
        if (n_threads > 0) {
            actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
        } else {
            actual_threads = omp_get_max_threads();
            if (actual_threads > n_snps) actual_threads = n_snps;
        }
        if (actual_threads < 1) actual_threads = 1;
#else
        (void)n_threads;
#endif

        /* Per-thread heap buffers for Pab recursion */
        int sc_n_index = table.n_index;
        int sc_pab_size = table.n_rows * sc_n_index;
        double *sc_pab_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)sc_pab_size * sizeof(double));
        double *sc_row0_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)sc_n_index * sizeof(double));
        if (!sc_pab_heap || !sc_row0_heap) {
            free(sc_pab_heap); free(sc_row0_heap);
            free_pab_table(&table);
            decref_score_output(&out);
            Py_DECREF(hi_eval_null_arr);
            Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }

        Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
        for (int s = 0; s < n_snps; s++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            double *my_pab = sc_pab_heap + (size_t)tid * sc_pab_size;
            double *my_row0 = sc_row0_heap + (size_t)tid * sc_n_index;

            const double *uab_snp = uab_batch + (size_t)s * n_samples * table.n_index;

            /* Compute row0: dot products of Hi_eval_null with each Uab column */
            for (int c = 0; c < table.n_index; c++) my_row0[c] = 0.0;
            for (int i = 0; i < n_samples; i++) {
                double h = hi_eval_null[i];
                for (int c = 0; c < table.n_index; c++)
                    my_row0[c] += h * uab_snp[i * table.n_index + c];
            }

            /* Full Pab via table-driven recursion */
            calc_pab_general(my_row0, &table, my_pab);

            double beta, se, f_stat;
            int is_valid = score_from_pab_general(my_pab, &table, n_samples,
                                                  &beta, &se, &f_stat);

            out_betas[s]    = beta;
            out_ses[s]      = se;
            out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b, lbeta_ab);
        }

        Py_END_ALLOW_THREADS
        free(sc_pab_heap);
        free(sc_row0_heap);

        free_pab_table(&table);

        if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
            decref_score_output(&out);
            Py_DECREF(hi_eval_null_arr);
            Py_DECREF(uab_arr);
            Py_DECREF(eigenvalues_arr);
            return NULL;
        }

        Py_DECREF(hi_eval_null_arr);
        Py_DECREF(uab_arr);
        Py_DECREF(eigenvalues_arr);
        return build_score_result_dict(&out);
    }

err_input_score_gen:
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(uab_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * GENERAL n_cvt LRT BATCH — compute_lrt_batch_general_c
 *
 * LRT test for arbitrary n_cvt using table-driven Pab recursion.
 * Mirrors compute_lrt_batch_c (n_cvt=1) but uses mle_logl_general.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * mle_logl_general — MLE log-likelihood for one SNP at one lambda (general n_cvt).
 *
 * MLE formula: -0.5 * n * log(P_yy_full) - 0.5 * logdet_h + mle_const
 * where P_yy_full is at level n_cvt+1 (fully projected).
 *
 * Uses full Uab row (n_samples * n_index) in AoS layout.
 * ------------------------------------------------------------------------- */
static double mle_logl_general(
    const double *uab_snp,     /* (n_samples, n_index) row-major */
    const double *eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    int ni = t->n_index;

    double logdet_h = 0.0;
    for (int c = 0; c < ni; c++) row0[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        logdet_h += log(v);
        for (int c = 0; c < ni; c++)
            row0[c] += h * uab_snp[i * ni + c];
    }

    calc_pab_general(row0, t, pab_scratch);

    /* P_yy_full at level n_cvt+1 (fully projected) */
    int nc = t->n_cvt;
    double P_yy = pab_scratch[(nc + 1) * ni + t->idx_yy];
    if (P_yy < 0.0) return (double)NAN;
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;

    return mle_const - 0.5 * logdet_h - 0.5 * (double)n_samples * log(P_yy);
}

/* -------------------------------------------------------------------------
 * mle_logl_general_cached — MLE using cached hi_eval for coarse grid search.
 * ------------------------------------------------------------------------- */
static double mle_logl_general_cached(
    const double *uab_snp,
    const double *cached_hi_eval,
    double cached_logdet_h,
    int n_samples,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    int ni = t->n_index;

    for (int c = 0; c < ni; c++) row0[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double h = cached_hi_eval[i];
        for (int c = 0; c < ni; c++)
            row0[c] += h * uab_snp[i * ni + c];
    }

    calc_pab_general(row0, t, pab_scratch);

    int nc = t->n_cvt;
    double P_yy = pab_scratch[(nc + 1) * ni + t->idx_yy];
    if (P_yy < 0.0) return (double)NAN;
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;

    return mle_const - 0.5 * cached_logdet_h - 0.5 * (double)n_samples * log(P_yy);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_mle_general — Grid + golden section for MLE (general n_cvt).
 *
 * Mirrors golden_section_lambda_mle_ncvt1 but uses mle_logl_general.
 * Returns optimal lambda; writes logl to *logl_out.
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_mle_general(
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
)
{
    const double phi = 0.6180339887498949;

    /* Stage 1: coarse grid search using cached hi_eval */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_general_cached(
            uab_snp,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            n_samples, mle_const, t,
            row0, pab_scratch
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    if (best_logl == REML_SENTINEL) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    /* Bracket */
    int idx_low  = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(c), mle_const, t,
                                  row0, pab_scratch);
    double fd = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(d), mle_const, t,
                                  row0, pab_scratch);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(c), mle_const, t,
                                   row0, pab_scratch);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(d), mle_const, t,
                                   row0, pab_scratch);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = mle_logl_general(uab_snp, eigenvalues, n_samples, lambda_opt, mle_const, t,
                                  row0, pab_scratch);

    return lambda_opt;
}

/* -------------------------------------------------------------------------
 * compute_lrt_batch_general_c
 *
 * Batch LRT for arbitrary n_cvt with optional OpenMP.
 *
 * Args: eigenvalues (n_samples,), Uab_batch (n_snps, n_samples, n_index),
 *       n_samples, n_cvt, pab_table_dict, l_min, l_max, n_grid, n_refine,
 *       logl_H0, n_threads
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_batch_general_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_batch_obj, *pab_table_dict;
    int n_samples, n_cvt, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;
    PyArrayObject *eigenvalues_arr = NULL, *uab_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOiiOddiidi",
            &eigenvalues_obj, &uab_batch_obj,
            &n_samples, &n_cvt,
            &pab_table_dict,
            &l_min, &l_max,
            &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }
    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }
    if (!PyDict_Check(pab_table_dict)) {
        PyErr_SetString(PyExc_TypeError, "pab_table_dict must be a dict");
        return NULL;
    }

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_lrt_gen;

    uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_batch_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) goto err_input_lrt_gen;

    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        goto err_input_lrt_gen;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, n_index)");
        goto err_input_lrt_gen;
    }

    {
        npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
        if (n_snps_raw > INT_MAX || n_snps_raw == 0) {
            PyErr_SetString(PyExc_ValueError, "n_snps must be > 0 and <= INT_MAX");
            goto err_input_lrt_gen;
        }
        int n_snps = (int)n_snps_raw;
        int n_index_arr = (int)PyArray_DIM(uab_arr, 2);

        /* Parse pab_table */
        pab_table_t table;
        if (parse_pab_table_from_dict(pab_table_dict, &table, n_samples) < 0)
            goto err_input_lrt_gen;

        if (table.n_index != n_index_arr) {
            PyErr_Format(PyExc_ValueError,
                "Uab_batch n_index=%d doesn't match pab_table n_index=%d",
                n_index_arr, table.n_index);
            free_pab_table(&table);
            goto err_input_lrt_gen;
        }

        const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
        const double *uab_batch   = (const double *)PyArray_DATA(uab_arr);

        if (validate_eigenvalues(eigenvalues, n_samples) < 0) {
            free_pab_table(&table);
            goto err_input_lrt_gen;
        }

        /* Allocate outputs */
        lrt_output_t out;
        if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
            free_pab_table(&table);
            PyErr_NoMemory();
            goto err_input_lrt_gen;
        }

        double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
        double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

        /* Pre-compute MLE constant and lambda grid */
        double n = (double)n_samples;
        double mle_const = 0.5 * n * (log(n) - log(2.0 * M_PI) - 1.0);

        double log_l_min = log(l_min);
        double log_l_max = log(l_max);
        double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

        double *lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
        if (!lambda_grid) {
            decref_lrt_output(&out);
            free_pab_table(&table);
            Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }
        for (int g = 0; g < n_grid; g++)
            lambda_grid[g] = exp(log_l_min + g * step);

        /* Pre-compute hi_eval_grid and logdet_h_grid */
        double *hi_eval_grid = (double *)malloc(
            (size_t)n_grid * (size_t)n_samples * sizeof(double));
        double *logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
        if (!hi_eval_grid || !logdet_h_grid) {
            free(lambda_grid); free(hi_eval_grid); free(logdet_h_grid);
            decref_lrt_output(&out);
            free_pab_table(&table);
            Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }
        for (int g = 0; g < n_grid; g++) {
            double lam = lambda_grid[g];
            double *hi = hi_eval_grid + (size_t)g * n_samples;
            double logdet = 0.0;
            for (int i = 0; i < n_samples; i++) {
                double v = lam * eigenvalues[i] + 1.0;
                hi[i] = 1.0 / v;
                logdet += log(v);
            }
            logdet_h_grid[g] = logdet;
        }

        int actual_threads = 1;
#ifdef _OPENMP
        if (n_threads > 0) {
            actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
        } else {
            actual_threads = omp_get_max_threads();
            if (actual_threads > n_snps) actual_threads = n_snps;
        }
        if (actual_threads < 1) actual_threads = 1;
#else
        (void)n_threads;
#endif

        /* Per-thread heap buffers for Pab recursion */
        int lrt_n_index = table.n_index;
        int lrt_pab_size = table.n_rows * lrt_n_index;
        double *lrt_pab_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)lrt_pab_size * sizeof(double));
        double *lrt_row0_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)lrt_n_index * sizeof(double));
        if (!lrt_pab_heap || !lrt_row0_heap) {
            free(lrt_pab_heap); free(lrt_row0_heap);
            free(lambda_grid); free(hi_eval_grid); free(logdet_h_grid);
            free_pab_table(&table);
            decref_lrt_output(&out);
            Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }

        Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
        for (int s = 0; s < n_snps; s++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            double *my_pab = lrt_pab_heap + (size_t)tid * lrt_pab_size;
            double *my_row0 = lrt_row0_heap + (size_t)tid * lrt_n_index;

            const double *uab_snp = uab_batch + (size_t)s * n_samples * table.n_index;

            double logl_H1;
            double lam_mle = golden_section_lambda_mle_general(
                uab_snp, eigenvalues, n_samples,
                lambda_grid, hi_eval_grid, logdet_h_grid,
                log_l_min, step, n_grid, n_refine,
                mle_const, &table, &logl_H1,
                my_row0, my_pab
            );
            out_lambdas_mle[s] = lam_mle;

            double lrt_stat = 2.0 * (logl_H1 - logl_H0);
            if (lrt_stat < 0.0) lrt_stat = 0.0;
            out_p_lrts[s] = chi2_sf_c(lrt_stat);
        }

        Py_END_ALLOW_THREADS
        free(lrt_pab_heap);
        free(lrt_row0_heap);

        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        free_pab_table(&table);

        Py_DECREF(uab_arr);
        Py_DECREF(eigenvalues_arr);
        return build_lrt_result_dict(&out);
    }

err_input_lrt_gen:
    Py_XDECREF(uab_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * FUSED MODE-4 — compute_mode4_chunk_split_c
 *
 * Single OpenMP loop computes Score + Wald + LRT from SoA split data.
 * Eliminates Uab reconstruction and redundant Pab computation.
 * ========================================================================= */


/* -------------------------------------------------------------------------
 * compute_mode4_chunk_split_c
 *
 * Fused per-chunk mode-4 compute: Score + Wald + LRT in a single OpenMP
 * parallel loop from SoA split data. Requires a mode-4 workspace.
 *
 * Python signature:
 *   compute_mode4_chunk_split_c(
 *       workspace,     # PyCapsule from create_workspace_mode4_split_c
 *       uab_varying,   # (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds, p_scores, lambdas_mle, p_lrts}
 * ------------------------------------------------------------------------- */
static PyObject *compute_mode4_chunk_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "uab_varying", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *uab_var_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &uab_var_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_t *ws = (lmm_workspace_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace");
    if (!ws) return NULL;

    /* Validate workspace mode */
    if (ws->mode != 4) {
        PyErr_Format(PyExc_ValueError,
            "compute_mode4_chunk_split_c requires a mode-4 workspace "
            "(got mode=%d). Use create_workspace_mode4_split_c.", ws->mode);
        return NULL;
    }

    PyArrayObject *uab_var_arr = NULL;
    mode4_output_t out = {0};
    PyObject *result = NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) return NULL;

    int n_samples = ws->n_samples;

    /* Validate shape */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying must be shape (n_snps, 3, %d)", n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_mode4_output(&out, (npy_intp)n_snps) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        goto err_input;
    }

    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
    const double *inv_ww = ws->inv_ww;
    const double *inv_wy = ws->inv_wy;
    const double *inv_yy = ws->inv_yy;

    double *out_lambdas     = (double *)PyArray_DATA(out.lambdas);
    double *out_logls       = (double *)PyArray_DATA(out.logls);
    double *out_betas       = (double *)PyArray_DATA(out.betas);
    double *out_ses         = (double *)PyArray_DATA(out.ses);
    double *out_pwalds      = (double *)PyArray_DATA(out.pwalds);
    double *out_p_scores    = (double *)PyArray_DATA(out.p_scores);
    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    int n_grid    = ws->n_grid;
    int n_refine  = ws->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Clamp n_threads to n_snps */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    /* Per-thread scratch buffers for MLE golden section refinement */
    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    if (!thread_bufs) {
        decref_mode4_output(&out);
        PyErr_NoMemory();
        goto err_input;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];

        const double *snp_base = uab_var_data + (size_t)snp * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;

        /* ---- (a) Score: null-model Pab (single pass, no optimization) ---- */
        {
            double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
            #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
            for (int i = 0; i < n_samples; i++) {
                double h = ws->hi_eval_null[i];
                s_wx += h * vwx[i];
                s_xx += h * vxx[i];
                s_xy += h * vxy[i];
            }

            double pab_null[3][6];
            calc_pab_ncvt1_split(ws->null_s_ww, s_wx, ws->null_s_wy,
                                  s_xx, s_xy, ws->null_s_yy, pab_null);

            double score_beta, score_se, score_f;
            int score_valid = score_from_pab(pab_null, n_samples, df,
                                              &score_beta, &score_se, &score_f);

            out_p_scores[snp] = f_to_pvalue(
                score_f, df, score_valid,
                ws->beta_a, ws->beta_b, ws->lbeta_ab);
        }

        /* ---- (b) logdet_iab (same as compute_lmm_chunk_split_c) ---- */
        double iab_s_wx = 0.0, iab_s_xx = 0.0;
        #pragma omp simd reduction(+:iab_s_wx,iab_s_xx)
        for (int i = 0; i < n_samples; i++) {
            iab_s_wx += vwx[i];
            iab_s_xx += vxx[i];
        }

        double iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * ws->iab_inv_ww;
        double logdet_iab = ws->iab_log_ww
                            + ((iab_p1_xx > 0.0) ? log(iab_p1_xx) : 0.0);

        int best_reml_idx, best_mle_idx;
        coarse_grid_mode4_ncvt1_split(
            vwx, vxx, vxy, n_samples,
            ws->hi_eval_grid, ws->logdet_h_grid, ws->grid_inv, n_grid,
            logdet_iab, df, reml_const, ws->mle_const,
            &best_reml_idx, &best_mle_idx
        );

        /* ---- (c) Wald: REML refinement from the shared coarse grid ---- */
        double logl_reml, wald_beta, wald_se, wald_f;
        int wald_valid;
        double lambda_reml = refine_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, ws->lambda_grid, ws->log_l_min, ws->step,
            n_grid, n_refine, best_reml_idx,
            df, reml_const, &logl_reml, &wald_beta, &wald_se, &wald_f,
            &wald_valid
        );

        out_lambdas[snp] = lambda_reml;
        out_logls[snp]   = logl_reml;
        out_betas[snp]   = wald_beta;
        out_ses[snp]     = wald_se;
        out_pwalds[snp]  = f_to_pvalue(
            wald_f, df, wald_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);

        /* ---- (d) LRT: MLE optimization ---- */
        double logl_H1;
        double lambda_mle = refine_lambda_mle_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, n_samples, ws->lambda_grid,
            ws->log_l_min, ws->step, n_grid, n_refine,
            best_mle_idx, ws->mle_const, hi_eval_local, &logl_H1
        );

        out_lambdas_mle[snp] = lambda_mle;

        /* LRT stat = 2*(logl_H1 - logl_H0), clamp >= 0 */
        double lrt_stat = 2.0 * (logl_H1 - ws->logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[snp] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    /* Free per-thread scratch buffers before any Python calls that might
     * raise (warn_betainc_convergence can raise if warnings are errors).
     * Buffers are only used inside the GIL-released compute loop above. */
    free_thread_scratch(thread_bufs, actual_threads);
    thread_bufs = NULL;

    if (warn_betainc_convergence(out_betas, out_pwalds, n_snps) < 0)
        goto err_output;

    result = build_mode4_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(uab_var_arr);
    return result;

err_output:
    decref_mode4_output(&out);
err_input:
    Py_XDECREF(uab_var_arr);
    return NULL;
}

/* =========================================================================
 * FUSED Uab — workspace holds w/Uty, chunk accepts UtG_T directly
 *
 * Eliminates the (n_snps, 3, n_samples) uab_varying_soa intermediate
 * allocation by computing wx/xx/xy products on-the-fly from UtG_T columns
 * in thread-local scratch buffers. Same FP operations in the same order
 * as the SoA path — results are bitwise-identical.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * create_workspace_fused_c
 *
 * Identical to create_workspace_split_c but with 2 additional parameters:
 *   w   (ndarray, shape (n_samples,)) — UtW[:,0]
 *   Uty (ndarray, shape (n_samples,)) — rotated phenotype
 *
 * Python signature:
 *   create_workspace_fused_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       w,                # (n_samples,) float64 — UtW[:,0]
 *       Uty,              # (n_samples,) float64 — rotated phenotype
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       n_threads,        # int
 *   ) -> PyCapsule wrapping lmm_workspace_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_fused_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "w", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *w_obj, *Uty_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &w_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *w_arr = NULL, *Uty_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) goto err_input;

    Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(w_arr) != 1 ||
        PyArray_DIM(w_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "w must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(Uty_arr) != 1 ||
        PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Uty must be shape (n_samples,)");
        goto err_input;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

    /* Fill scalar fields (same as create_workspace_split_c) */
    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->l_min     = l_min;
    ws->l_max     = l_max;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    ws->log_l_min   = log(l_min);
    double log_l_max = log(l_max);
    ws->step        = (log_l_max - ws->log_l_min) / (double)(n_grid - 1);
    ws->reml_const  = 0.5 * ws->df * (log((double)ws->df)
                       - log(2.0 * M_PI) - 1.0);

    /* Borrow pointers — arrays kept alive via Py_INCREF */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    /* Store w and Uty for fused on-the-fly Uab computation */
    Py_INCREF(w_arr);
    Py_INCREF(Uty_arr);
    ws->w = (const double *)PyArray_DATA(w_arr);
    ws->Uty = (const double *)PyArray_DATA(Uty_arr);
    ws->w_ref = (PyObject *)w_arr;
    ws->Uty_ref = (PyObject *)Uty_arr;

    /* Compute invariant Iab scalar: sum(inv_ww) */
    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_ws;
    }

    /* Build lambda grid + invariant dot products */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
            sw  += h * ws->inv_ww[i];
            swy += h * ws->inv_wy[i];
            sy  += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;

        ws->grid_inv[g].s_ww    = sw;
        ws->grid_inv[g].s_wy    = swy;
        ws->grid_inv[g].s_yy    = sy;
        ws->grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
    }

    /* Wrap in PyCapsule */
    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    /* Release local refs — capsule now owns ws->*_ref via destructor */
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    return capsule;

err_ws:
    if (ws) {
        Py_XDECREF(ws->eigenvalues_ref);
        Py_XDECREF(ws->uab_inv_ref);
        Py_XDECREF(ws->w_ref);
        Py_XDECREF(ws->Uty_ref);
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->grid_inv);
        free(ws);
    }
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(w_arr);
    Py_XDECREF(Uty_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_fused_c
 *
 * Fused per-chunk Wald compute: accepts UtG_T (n_snps, n_samples) and
 * computes wx/xx/xy on-the-fly from w/Uty stored in workspace.
 * Same FP operations as compute_lmm_chunk_split_c — bitwise-identical.
 *
 * Python signature:
 *   compute_lmm_chunk_fused_c(
 *       workspace,   # PyCapsule from create_workspace_fused_c
 *       utg_t,       # (n_snps, n_samples) float64 — UtG.T
 *       n_threads,   # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_chunk_fused_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "utg_t", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *utg_t_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &utg_t_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_t *ws = (lmm_workspace_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace");
    if (!ws) return NULL;

    /* Validate workspace has w/Uty (fused workspace) */
    if (!ws->w || !ws->Uty) {
        PyErr_SetString(PyExc_ValueError,
            "compute_lmm_chunk_fused_c requires a fused workspace "
            "(w/Uty pointers are NULL). Use create_workspace_fused_c.");
        return NULL;
    }

    PyArrayObject *utg_t_arr = NULL;
    output_arrays_t out = {0};
    PyObject *result = NULL;

    utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) return NULL;

    int n_samples = ws->n_samples;

    /* Validate shape: must be 2D (n_snps, n_samples) */
    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d), got (%d, %d)",
            n_samples,
            (int)(PyArray_NDIM(utg_t_arr) >= 1 ? PyArray_DIM(utg_t_arr, 0) : -1),
            (int)(PyArray_NDIM(utg_t_arr) >= 2 ? PyArray_DIM(utg_t_arr, 1) : -1));
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);
    const double *inv_ww = ws->inv_ww;
    const double *inv_wy = ws->inv_wy;
    const double *inv_yy = ws->inv_yy;
    const double *w_ptr = ws->w;
    const double *Uty_ptr = ws->Uty;

    double *lambdas = (double *)PyArray_DATA(out.lambdas);
    double *logls   = (double *)PyArray_DATA(out.logls);
    double *betas   = (double *)PyArray_DATA(out.betas);
    double *ses     = (double *)PyArray_DATA(out.ses);
    double *pwalds  = (double *)PyArray_DATA(out.pwalds);

    int n_grid    = ws->n_grid;
    int n_refine  = ws->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Clamp n_threads to n_snps */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    /* Per-thread scratch buffers for on-the-fly wx/xx/xy computation */
    double **scratch_wx = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **scratch_xx = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **scratch_xy = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    if (!scratch_wx || !scratch_xx || !scratch_xy) {
        free_thread_scratch(scratch_wx, actual_threads);
        free_thread_scratch(scratch_xx, actual_threads);
        free_thread_scratch(scratch_xy, actual_threads);
        decref_output_arrays(&out);
        PyErr_NoMemory();
        goto err_input;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *vwx = scratch_wx[tid];
        double *vxx = scratch_xx[tid];
        double *vxy = scratch_xy[tid];

        const double *x = utg_t_data + (size_t)snp * n_samples;

        /* Compute wx/xx/xy on-the-fly — same operations as SoA path */
        for (int i = 0; i < n_samples; i++) {
            vwx[i] = w_ptr[i] * x[i];
            vxx[i] = x[i] * x[i];
            vxy[i] = x[i] * Uty_ptr[i];
        }

        /* From here, identical to compute_lmm_chunk_split_c per-SNP body */
        double iab_s_wx = 0.0, iab_s_xx = 0.0;
        #pragma omp simd reduction(+:iab_s_wx,iab_s_xx)
        for (int i = 0; i < n_samples; i++) {
            iab_s_wx += vwx[i];
            iab_s_xx += vxx[i];
        }

        double iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * ws->iab_inv_ww;
        double logdet_iab = ws->iab_log_ww
                            + ((iab_p1_xx > 0.0) ? log(iab_p1_xx) : 0.0);

        double logl_opt, beta, se, f_stat;
        int is_valid;
        double lambda_opt = golden_section_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->grid_inv, ws->log_l_min, ws->step, n_grid, n_refine,
            df, reml_const, &logl_opt, &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;

        pwalds[snp] = f_to_pvalue(
            f_stat, df, is_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    /* Free scratch buffers */
    free_thread_scratch(scratch_wx, actual_threads);
    free_thread_scratch(scratch_xx, actual_threads);
    free_thread_scratch(scratch_xy, actual_threads);

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(utg_t_arr);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(utg_t_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * create_workspace_mode4_fused_c
 *
 * Mode-4 fused workspace: extends standard mode-4 workspace with w/Uty
 * for on-the-fly Uab computation from UtG_T.
 *
 * Python signature:
 *   create_workspace_mode4_fused_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       w,                # (n_samples,) float64 — UtW[:,0]
 *       Uty,              # (n_samples,) float64 — rotated phenotype
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       n_threads,        # int
 *       hi_eval_null,     # (n_samples,) float64 — null-model Hi_eval
 *       logl_H0,          # float — null MLE log-likelihood
 *   ) -> PyCapsule wrapping lmm_workspace_t (mode=4)
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_mode4_fused_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "w", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *w_obj, *Uty_obj;
    PyObject *hi_eval_null_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiiiOd", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &w_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &hi_eval_null_obj, &logl_H0)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *w_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) goto err_input;

    Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) goto err_input;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(w_arr) != 1 ||
        PyArray_DIM(w_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "w must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(Uty_arr) != 1 ||
        PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Uty must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "hi_eval_null must be shape (n_samples,)");
        goto err_input;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    /* Validate Hi_eval_null for NaN/Inf and non-positive values */
    {
        const double *hi_null = (const double *)PyArray_DATA(hi_eval_null_arr);
        for (int i = 0; i < n_samples; i++) {
            char buf[64];
            if (!isfinite(hi_null[i])) {
                snprintf(buf, sizeof(buf), "%g", hi_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not finite. "
                    "Null model optimization may have failed.", i, buf);
                goto err_input;
            }
            if (hi_null[i] <= 0.0) {
                snprintf(buf, sizeof(buf), "%g", hi_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not positive. "
                    "Check kinship matrix conditioning.",
                    i, buf);
                goto err_input;
            }
        }
    }

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

    /* Fill scalar fields */
    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->l_min     = l_min;
    ws->l_max     = l_max;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    ws->log_l_min   = log(l_min);
    double log_l_max_m4 = log(l_max);
    ws->step        = (log_l_max_m4 - ws->log_l_min) / (double)(n_grid - 1);
    ws->reml_const  = 0.5 * ws->df * (log((double)ws->df)
                       - log(2.0 * M_PI) - 1.0);

    /* Borrow pointers */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    /* Store w and Uty for fused on-the-fly Uab computation */
    Py_INCREF(w_arr);
    Py_INCREF(Uty_arr);
    ws->w = (const double *)PyArray_DATA(w_arr);
    ws->Uty = (const double *)PyArray_DATA(Uty_arr);
    ws->w_ref = (PyObject *)w_arr;
    ws->Uty_ref = (PyObject *)Uty_arr;

    /* Compute invariant Iab scalar */
    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_ws;
    }

    /* Build lambda grid + invariant dot products */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
            sw  += h * ws->inv_ww[i];
            swy += h * ws->inv_wy[i];
            sy  += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;

        ws->grid_inv[g].s_ww    = sw;
        ws->grid_inv[g].s_wy    = swy;
        ws->grid_inv[g].s_yy    = sy;
        ws->grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
    }

    /* Mode-4 specific fields */
    ws->mode = 4;
    ws->logl_H0 = logl_H0;
    ws->mle_const = 0.5 * (double)n_samples
                    * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

    /* Copy hi_eval_null into workspace-owned buffer */
    ws->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->hi_eval_null) {
        PyErr_NoMemory();
        goto err_ws;
    }
    {
        const double *src = (const double *)PyArray_DATA(hi_eval_null_arr);
        memcpy(ws->hi_eval_null, src, (size_t)n_samples * sizeof(double));
    }

    /* Precompute null-model invariant dot products */
    {
        double ns_ww = 0.0, ns_wy = 0.0, ns_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double h = ws->hi_eval_null[i];
            ns_ww += h * ws->inv_ww[i];
            ns_wy += h * ws->inv_wy[i];
            ns_yy += h * ws->inv_yy[i];
        }
        ws->null_s_ww   = ns_ww;
        ws->null_s_wy   = ns_wy;
        ws->null_s_yy   = ns_yy;
        ws->null_inv_ww  = (ns_ww != 0.0) ? 1.0 / ns_ww : 0.0;
    }

    /* Wrap in PyCapsule */
    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(hi_eval_null_arr);
    return capsule;

err_ws:
    if (ws) {
        Py_XDECREF(ws->eigenvalues_ref);
        Py_XDECREF(ws->uab_inv_ref);
        Py_XDECREF(ws->w_ref);
        Py_XDECREF(ws->Uty_ref);
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->grid_inv);
        free(ws->hi_eval_null);
        free(ws);
    }
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(w_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(hi_eval_null_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_mode4_chunk_fused_c
 *
 * Fused per-chunk mode-4 compute: Score + Wald + LRT from UtG_T directly.
 * Same as compute_mode4_chunk_split_c but computes wx/xx/xy on-the-fly.
 *
 * Python signature:
 *   compute_mode4_chunk_fused_c(
 *       workspace,   # PyCapsule from create_workspace_mode4_fused_c
 *       utg_t,       # (n_snps, n_samples) float64 — UtG.T
 *       n_threads,   # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds, p_scores, lambdas_mle, p_lrts}
 * ------------------------------------------------------------------------- */
static PyObject *compute_mode4_chunk_fused_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "utg_t", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *utg_t_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &utg_t_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_t *ws = (lmm_workspace_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace");
    if (!ws) return NULL;

    /* Validate workspace mode and fused fields */
    if (ws->mode != 4) {
        PyErr_Format(PyExc_ValueError,
            "compute_mode4_chunk_fused_c requires a mode-4 workspace "
            "(got mode=%d). Use create_workspace_mode4_fused_c.", ws->mode);
        return NULL;
    }
    if (!ws->w || !ws->Uty) {
        PyErr_SetString(PyExc_ValueError,
            "compute_mode4_chunk_fused_c requires a fused workspace "
            "(w/Uty pointers are NULL). Use create_workspace_mode4_fused_c.");
        return NULL;
    }

    PyArrayObject *utg_t_arr = NULL;
    mode4_output_t out = {0};
    PyObject *result = NULL;

    utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) return NULL;

    int n_samples = ws->n_samples;

    /* Validate shape: must be 2D (n_snps, n_samples) */
    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_mode4_output(&out, (npy_intp)n_snps) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        goto err_input;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);
    const double *inv_ww = ws->inv_ww;
    const double *inv_wy = ws->inv_wy;
    const double *inv_yy = ws->inv_yy;
    const double *w_ptr = ws->w;
    const double *Uty_ptr = ws->Uty;

    double *out_lambdas     = (double *)PyArray_DATA(out.lambdas);
    double *out_logls       = (double *)PyArray_DATA(out.logls);
    double *out_betas       = (double *)PyArray_DATA(out.betas);
    double *out_ses         = (double *)PyArray_DATA(out.ses);
    double *out_pwalds      = (double *)PyArray_DATA(out.pwalds);
    double *out_p_scores    = (double *)PyArray_DATA(out.p_scores);
    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    int n_grid    = ws->n_grid;
    int n_refine  = ws->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Clamp n_threads to n_snps */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    /* Per-thread scratch buffers:
     * - 3 for wx/xx/xy on-the-fly computation
     * - 1 for MLE golden section refinement (hi_eval_local) */
    double **scratch_wx = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **scratch_xx = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **scratch_xy = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    if (!scratch_wx || !scratch_xx || !scratch_xy || !thread_bufs) {
        free_thread_scratch(scratch_wx, actual_threads);
        free_thread_scratch(scratch_xx, actual_threads);
        free_thread_scratch(scratch_xy, actual_threads);
        free_thread_scratch(thread_bufs, actual_threads);
        decref_mode4_output(&out);
        PyErr_NoMemory();
        goto err_input;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *vwx = scratch_wx[tid];
        double *vxx = scratch_xx[tid];
        double *vxy = scratch_xy[tid];
        double *hi_eval_local = thread_bufs[tid];

        const double *x = utg_t_data + (size_t)snp * n_samples;

        /* Compute wx/xx/xy on-the-fly */
        for (int i = 0; i < n_samples; i++) {
            vwx[i] = w_ptr[i] * x[i];
            vxx[i] = x[i] * x[i];
            vxy[i] = x[i] * Uty_ptr[i];
        }

        /* ---- (a) Score: null-model Pab ---- */
        {
            double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
            #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
            for (int i = 0; i < n_samples; i++) {
                double h = ws->hi_eval_null[i];
                s_wx += h * vwx[i];
                s_xx += h * vxx[i];
                s_xy += h * vxy[i];
            }

            double pab_null[3][6];
            calc_pab_ncvt1_split(ws->null_s_ww, s_wx, ws->null_s_wy,
                                  s_xx, s_xy, ws->null_s_yy, pab_null);

            double score_beta, score_se, score_f;
            int score_valid = score_from_pab(pab_null, n_samples, df,
                                              &score_beta, &score_se, &score_f);

            out_p_scores[snp] = f_to_pvalue(
                score_f, df, score_valid,
                ws->beta_a, ws->beta_b, ws->lbeta_ab);
        }

        /* ---- (b) logdet_iab ---- */
        double iab_s_wx = 0.0, iab_s_xx = 0.0;
        #pragma omp simd reduction(+:iab_s_wx,iab_s_xx)
        for (int i = 0; i < n_samples; i++) {
            iab_s_wx += vwx[i];
            iab_s_xx += vxx[i];
        }

        double iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * ws->iab_inv_ww;
        double logdet_iab = ws->iab_log_ww
                            + ((iab_p1_xx > 0.0) ? log(iab_p1_xx) : 0.0);

        int best_reml_idx, best_mle_idx;
        coarse_grid_mode4_ncvt1_split(
            vwx, vxx, vxy, n_samples,
            ws->hi_eval_grid, ws->logdet_h_grid, ws->grid_inv, n_grid,
            logdet_iab, df, reml_const, ws->mle_const,
            &best_reml_idx, &best_mle_idx
        );

        /* ---- (c) Wald: REML refinement from the shared coarse grid ---- */
        double logl_reml, wald_beta, wald_se, wald_f;
        int wald_valid;
        double lambda_reml = refine_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, ws->lambda_grid, ws->log_l_min, ws->step,
            n_grid, n_refine, best_reml_idx,
            df, reml_const, &logl_reml, &wald_beta, &wald_se, &wald_f,
            &wald_valid
        );

        out_lambdas[snp] = lambda_reml;
        out_logls[snp]   = logl_reml;
        out_betas[snp]   = wald_beta;
        out_ses[snp]     = wald_se;
        out_pwalds[snp]  = f_to_pvalue(
            wald_f, df, wald_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);

        /* ---- (d) LRT: MLE optimization ---- */
        double logl_H1;
        double lambda_mle = refine_lambda_mle_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, n_samples, ws->lambda_grid,
            ws->log_l_min, ws->step, n_grid, n_refine,
            best_mle_idx, ws->mle_const, hi_eval_local, &logl_H1
        );

        out_lambdas_mle[snp] = lambda_mle;

        double lrt_stat = 2.0 * (logl_H1 - ws->logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[snp] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    /* Free per-thread scratch buffers */
    free_thread_scratch(scratch_wx, actual_threads);
    free_thread_scratch(scratch_xx, actual_threads);
    free_thread_scratch(scratch_xy, actual_threads);
    free_thread_scratch(thread_bufs, actual_threads);

    if (warn_betainc_convergence(out_betas, out_pwalds, n_snps) < 0)
        goto err_output;

    result = build_mode4_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(utg_t_arr);
    return result;

err_output:
    decref_mode4_output(&out);
err_input:
    Py_XDECREF(utg_t_arr);
    return NULL;
}

/* =========================================================================
 * FUSED GENERAL Uab — workspace holds UtW(matrix)/Uty, chunk accepts UtG_T
 *
 * Generalizes the n_cvt=1 fused path to arbitrary n_cvt. Instead of 3
 * hardcoded dot products (wx, xx, xy), computes n_var varying dot products
 * on-the-fly using var_a_cols/var_b_cols lookup into UtW columns, UtG_T
 * (the SNP genotype vector), and Uty. Same table-driven Pab recursion as
 * compute_lmm_chunk_general_c -- results are bitwise-identical to the
 * non-fused general path.
 *
 * Memory savings: eliminates (n_snps, n_var, n_samples) tensor.
 * At 100k samples: 75GB (n_cvt=2), 112GB (n_cvt=3), 209GB (n_cvt=5).
 * ========================================================================= */

/* Helper: resolve 0-based column index to the corresponding vector.
 * Columns 0..n_cvt-1 = UtW columns, n_cvt = X (genotype), n_cvt+1 = Uty. */
static inline const double *get_fused_vector(
    const lmm_workspace_general_t *ws,
    int col_0based,
    const double *x)
{
    assert(col_0based >= 0 && col_0based <= ws->n_cvt + 1);
    if (col_0based < ws->n_cvt)
        return ws->UtW + (size_t)col_0based * ws->n_samples;
    if (col_0based == ws->n_cvt)
        return x;
    return ws->Uty;  /* col_0based == n_cvt + 1 */
}

/* -------------------------------------------------------------------------
 * init_fused_general_workspace — shared initialization for Wald-only and
 * mode-4 fused general workspace creators.
 *
 * Populates all common fields of a calloc'd lmm_workspace_general_t:
 * table, eigenvalues, uab_inv, UtW (transposed), Uty, scratch, lambda grid,
 * hi_eval_grid, logdet_h_grid, inv_sums_grid, inv_identity_sums, beta/REML
 * constants, and var_a/var_b column indices.
 *
 * Caller must calloc ws before calling. On success returns 0. On failure
 * returns -1 with Python exception set; caller must free ws via the
 * destructor (all fields are NULL-safe via calloc + free(NULL)).
 *
 * Does NOT set mode-4 fields (hi_eval_null, null_inv_sums, logl_H0,
 * mle_const, mode) — the mode-4 caller sets those after this returns.
 * ------------------------------------------------------------------------- */
static int init_fused_general_workspace(
    lmm_workspace_general_t *ws,
    PyArrayObject *eigenvalues_arr,
    PyArrayObject *uab_inv_arr,
    PyArrayObject *UtW_arr,
    PyArrayObject *Uty_arr,
    PyObject *inv_idx_obj, PyObject *var_idx_obj,
    PyObject *diag_rows_obj, PyObject *diag_cols_obj,
    PyObject *offsets_obj, PyObject *counts_obj, PyObject *entries_obj,
    PyObject *var_a_obj, PyObject *var_b_obj,
    int n_samples, double l_min, double l_max,
    int n_grid, int n_refine, int n_threads, int n_cvt,
    int idx_xx, int idx_xy, int idx_yy)
{
    int n_index = (n_cvt + 3) * (n_cvt + 2) / 2;
    int n_rows  = n_cvt + 2;

    /* Parse invariant_indices to determine n_inv */
    PyArrayObject *inv_idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
        inv_idx_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!inv_idx_arr) return -1;
    int n_inv = (int)PyArray_SIZE(inv_idx_arr);
    Py_DECREF(inv_idx_arr);

    PyArrayObject *var_idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
        var_idx_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!var_idx_arr) return -1;
    int n_var = (int)PyArray_SIZE(var_idx_arr);
    Py_DECREF(var_idx_arr);

    if (n_inv + n_var != n_index) {
        PyErr_Format(PyExc_ValueError,
            "n_inv (%d) + n_var (%d) != n_index (%d)", n_inv, n_var, n_index);
        return -1;
    }

    /* Parse entries to get total count */
    PyArrayObject *entries_arr = (PyArrayObject *)PyArray_FROM_OTF(
        entries_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!entries_arr) return -1;
    int entries_len = (int)PyArray_SIZE(entries_arr);
    Py_DECREF(entries_arr);
    if (entries_len % 4 != 0) {
        PyErr_Format(PyExc_ValueError,
            "entries length (%d) not a multiple of 4", entries_len);
        return -1;
    }
    int n_entries = entries_len / 4;

    /* Store scalars */
    ws->n_samples = n_samples;
    ws->n_grid = n_grid;
    ws->n_refine = n_refine;
    ws->n_cvt = n_cvt;

    /* Fill table */
    ws->table.n_cvt = n_cvt;
    ws->table.n_index = n_index;
    ws->table.n_rows = n_rows;
    ws->table.n_inv = n_inv;
    ws->table.n_var = n_var;
    ws->table.idx_xx = idx_xx;
    ws->table.idx_xy = idx_xy;
    ws->table.idx_yy = idx_yy;
    ws->table.df = n_samples - n_cvt - 1;
    ws->table.n_entries = n_entries;

    /* Parse index arrays into owned copies */
    ws->table.invariant_indices = parse_int32_array(inv_idx_obj, n_inv, "invariant_indices");
    if (!ws->table.invariant_indices) return -1;
    ws->table.varying_indices   = parse_int32_array(var_idx_obj, n_var, "varying_indices");
    if (!ws->table.varying_indices) return -1;
    ws->table.logdet_diag_rows  = parse_int32_array(diag_rows_obj, n_cvt + 1, "logdet_diag_rows");
    if (!ws->table.logdet_diag_rows) return -1;
    ws->table.logdet_diag_cols  = parse_int32_array(diag_cols_obj, n_cvt + 1, "logdet_diag_cols");
    if (!ws->table.logdet_diag_cols) return -1;
    ws->table.level_offsets     = parse_int32_array(offsets_obj, n_rows, "level_offsets");
    if (!ws->table.level_offsets) return -1;
    ws->table.level_counts      = parse_int32_array(counts_obj, n_rows, "level_counts");
    if (!ws->table.level_counts) return -1;

    /* Parse entries (stride-4) into pab_entry_t array */
    {
        int *raw_entries = parse_int32_array(entries_obj, n_entries * 4, "entries");
        if (!raw_entries) return -1;
        ws->table.entries = (pab_entry_t *)malloc(
            (size_t)n_entries * sizeof(pab_entry_t));
        if (!ws->table.entries) {
            free(raw_entries);
            PyErr_NoMemory();
            return -1;
        }
        for (int i = 0; i < n_entries; i++) {
            ws->table.entries[i].index_ab = raw_entries[i * 4 + 0];
            ws->table.entries[i].index_aw = raw_entries[i * 4 + 1];
            ws->table.entries[i].index_bw = raw_entries[i * 4 + 2];
            ws->table.entries[i].index_ww = raw_entries[i * 4 + 3];
        }
        free(raw_entries);
    }

    /* Validate table indices */
    for (int i = 0; i < n_inv; i++) {
        if (ws->table.invariant_indices[i] < 0 ||
            ws->table.invariant_indices[i] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "invariant_indices[%d] = %d out of range [0, %d)",
                i, ws->table.invariant_indices[i], n_index);
            return -1;
        }
    }
    for (int i = 0; i < n_var; i++) {
        if (ws->table.varying_indices[i] < 0 ||
            ws->table.varying_indices[i] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "varying_indices[%d] = %d out of range [0, %d)",
                i, ws->table.varying_indices[i], n_index);
            return -1;
        }
    }
    for (int d = 0; d < n_cvt + 1; d++) {
        if (ws->table.logdet_diag_rows[d] < 0 ||
            ws->table.logdet_diag_rows[d] >= n_rows) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_rows[%d] = %d out of range [0, %d)",
                d, ws->table.logdet_diag_rows[d], n_rows);
            return -1;
        }
        if (ws->table.logdet_diag_cols[d] < 0 ||
            ws->table.logdet_diag_cols[d] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_cols[%d] = %d out of range [0, %d)",
                d, ws->table.logdet_diag_cols[d], n_index);
            return -1;
        }
    }
    for (int p = 0; p < n_rows; p++) {
        if (ws->table.level_offsets[p] < 0 ||
            ws->table.level_counts[p] < 0 ||
            (int64_t)ws->table.level_offsets[p] + ws->table.level_counts[p] > n_entries) {
            PyErr_Format(PyExc_ValueError,
                "level_offsets[%d]=%d + level_counts[%d]=%d exceeds n_entries=%d",
                p, ws->table.level_offsets[p], p, ws->table.level_counts[p], n_entries);
            return -1;
        }
    }
    if (idx_xx < 0 || idx_xx >= n_index ||
        idx_xy < 0 || idx_xy >= n_index ||
        idx_yy < 0 || idx_yy >= n_index) {
        PyErr_SetString(PyExc_ValueError, "idx_xx/xy/yy out of range [0, n_index)");
        return -1;
    }
    for (int i = 0; i < n_entries; i++) {
        const pab_entry_t *e = &ws->table.entries[i];
        if (e->index_ab < 0 || e->index_ab >= n_index ||
            e->index_aw < 0 || e->index_aw >= n_index ||
            e->index_bw < 0 || e->index_bw >= n_index ||
            e->index_ww < 0 || e->index_ww >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "entries[%d] has index out of range [0, %d)", i, n_index);
            return -1;
        }
    }

    /* Parse var_a_cols and var_b_cols */
    ws->var_a_cols = parse_int32_array(var_a_obj, n_var, "var_a_cols");
    if (!ws->var_a_cols) return -1;
    ws->var_b_cols = parse_int32_array(var_b_obj, n_var, "var_b_cols");
    if (!ws->var_b_cols) return -1;

    /* Validate var_a/var_b column indices */
    for (int v = 0; v < n_var; v++) {
        if (ws->var_a_cols[v] < 0 || ws->var_a_cols[v] > n_cvt + 1 ||
            ws->var_b_cols[v] < 0 || ws->var_b_cols[v] > n_cvt + 1) {
            PyErr_Format(PyExc_ValueError,
                "var_a_cols[%d]=%d or var_b_cols[%d]=%d out of range [0, %d]",
                v, ws->var_a_cols[v], v, ws->var_b_cols[v], n_cvt + 1);
            return -1;
        }
    }

    /* Copy eigenvalues (owned) */
    ws->eigenvalues = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!ws->eigenvalues) { PyErr_NoMemory(); return -1; }
    memcpy(ws->eigenvalues, PyArray_DATA(eigenvalues_arr),
           (size_t)n_samples * sizeof(double));

    /* Validate and borrow invariant Uab pointer */
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != n_inv ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant must be shape (%d, %d), got (%lld, %lld)",
            n_inv, n_samples,
            (long long)(PyArray_NDIM(uab_inv_arr) >= 1 ? PyArray_DIM(uab_inv_arr, 0) : -1),
            (long long)(PyArray_NDIM(uab_inv_arr) >= 2 ? PyArray_DIM(uab_inv_arr, 1) : -1));
        return -1;
    }
    Py_INCREF(uab_inv_arr);
    ws->uab_inv_ref = (PyObject *)uab_inv_arr;
    ws->uab_inv = (const double *)PyArray_DATA(uab_inv_arr);

    /* Transpose UtW from row-major (n_samples, n_cvt) to column-major
     * (n_cvt, n_samples) for cache-friendly per-column access. */
    ws->utw_transposed = (double *)malloc(
        (size_t)n_cvt * (size_t)n_samples * sizeof(double));
    if (!ws->utw_transposed) { PyErr_NoMemory(); return -1; }
    {
        const double *src = (const double *)PyArray_DATA(UtW_arr);
        for (int c = 0; c < n_cvt; c++) {
            double *dst = ws->utw_transposed + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                dst[i] = src[(size_t)i * n_cvt + c];
        }
    }
    ws->UtW = ws->utw_transposed;

    /* Borrow Uty pointer */
    Py_INCREF(Uty_arr);
    ws->Uty_ref = (PyObject *)Uty_arr;
    ws->Uty = (const double *)PyArray_DATA(Uty_arr);

    /* Allocate per-thread scratch: n_var * n_samples per thread */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = n_threads;
    if (actual_threads < 1) actual_threads = 1;
#endif
    ws->actual_threads = actual_threads;
    ws->scratch_flat = (double *)malloc(
        (size_t)actual_threads * (size_t)n_var * (size_t)n_samples * sizeof(double));
    if (!ws->scratch_flat) { PyErr_NoMemory(); return -1; }

    /* Per-thread heap buffers for Pab recursion (avoids stack overflow) */
    int pab_size = n_rows * n_index;
    ws->pab_size = pab_size;
    ws->pab_per_thread = (double *)malloc(
        (size_t)actual_threads * (size_t)pab_size * sizeof(double));
    if (!ws->pab_per_thread) { PyErr_NoMemory(); return -1; }
    ws->row0_per_thread = (double *)malloc(
        (size_t)actual_threads * (size_t)n_index * sizeof(double));
    if (!ws->row0_per_thread) { PyErr_NoMemory(); return -1; }

    /* Compute df, reml_const, beta params */
    int df = ws->table.df;
    ws->beta_a = (double)df / 2.0;
    ws->beta_b = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);
    ws->reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Build lambda grid */
    double log_l_min = log(l_min);
    double log_l_max_v = log(l_max);
    double step = (log_l_max_v - log_l_min) / (double)(n_grid - 1);

    ws->lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->inv_sums_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_inv * sizeof(double));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->inv_sums_grid) {
        PyErr_NoMemory();
        return -1;
    }

    for (int g = 0; g < n_grid; g++)
        ws->lambda_grid[g] = exp(log_l_min + g * step);

    /* Precompute hi_eval_grid, logdet_h_grid, and invariant sums */
    for (int g = 0; g < n_grid; g++) {
        double lam = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
        }
        ws->logdet_h_grid[g] = logdet;

        double *inv_sums = ws->inv_sums_grid + (size_t)g * n_inv;
        for (int c = 0; c < n_inv; c++) {
            double s = 0.0;
            const double *col = ws->uab_inv + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += hi_row[i] * col[i];
            inv_sums[c] = s;
        }
    }

    /* Precompute invariant identity sums */
    ws->inv_identity_sums = (double *)malloc((size_t)n_inv * sizeof(double));
    if (!ws->inv_identity_sums) { PyErr_NoMemory(); return -1; }
    for (int c = 0; c < n_inv; c++) {
        double s = 0.0;
        const double *col = ws->uab_inv + (size_t)c * n_samples;
        for (int i = 0; i < n_samples; i++)
            s += col[i];
        ws->inv_identity_sums[c] = s;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * free_fused_general_workspace — cleanup helper for error paths in
 * create_workspace_fused_general_c_py and create_workspace_mode4_fused_general_c_py.
 *
 * Frees all workspace fields and the workspace itself. All pointers are
 * NULL-safe (ws was calloc'd, so unset fields are NULL → free(NULL) is no-op).
 * ------------------------------------------------------------------------- */
static void free_fused_general_workspace(lmm_workspace_general_t *ws)
{
    if (!ws) return;
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->inv_sums_grid);
    free(ws->eigenvalues);
    free(ws->inv_identity_sums);
    free(ws->table.invariant_indices);
    free(ws->table.varying_indices);
    free(ws->table.logdet_diag_rows);
    free(ws->table.logdet_diag_cols);
    free(ws->table.level_offsets);
    free(ws->table.level_counts);
    free(ws->table.entries);
    Py_XDECREF(ws->uab_inv_ref);
    free(ws->utw_transposed);
    free(ws->var_a_cols);
    free(ws->var_b_cols);
    free(ws->scratch_flat);
    free(ws->pab_per_thread);
    free(ws->row0_per_thread);
    Py_XDECREF(ws->Uty_ref);
    free(ws->hi_eval_null);
    free(ws->null_inv_sums);
    free(ws->uab_snp_flat);
    free(ws);
}

/* -------------------------------------------------------------------------
 * create_workspace_fused_general_c
 *
 * Like create_workspace_general_c but additionally stores UtW (transposed
 * to column-major), Uty, and var_a_cols/var_b_cols for on-the-fly dot
 * product computation. Allocates per-thread scratch buffers.
 *
 * Python signature:
 *   create_workspace_fused_general_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (n_inv, n_samples) float64 — SoA
 *       UtW,              # (n_samples, n_cvt) float64 — row-major
 *       Uty,              # (n_samples,) float64
 *       n_samples,        # int
 *       l_min, l_max,     # float
 *       n_grid, n_refine, n_threads,  # int
 *       n_cvt,            # int
 *       invariant_indices, varying_indices,    # int32
 *       logdet_diag_rows, logdet_diag_cols,    # int32
 *       level_offsets, level_counts, entries,   # int32
 *       idx_xx, idx_xy, idx_yy,                # int
 *       var_a_cols, var_b_cols                  # int32
 *   ) -> PyCapsule wrapping lmm_workspace_general_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_fused_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "UtW", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "n_cvt",
        "invariant_indices", "varying_indices",
        "logdet_diag_rows", "logdet_diag_cols",
        "level_offsets", "level_counts", "entries",
        "idx_xx", "idx_xy", "idx_yy",
        "var_a_cols", "var_b_cols",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *UtW_obj, *Uty_obj;
    PyObject *inv_idx_obj, *var_idx_obj;
    PyObject *diag_rows_obj, *diag_cols_obj;
    PyObject *offsets_obj, *counts_obj, *entries_obj;
    PyObject *var_a_obj, *var_b_obj;
    int n_samples, n_grid, n_refine, n_threads, n_cvt;
    int idx_xx, idx_xy, idx_yy;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiiiiOOOOOOOiiiOO", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &UtW_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &n_cvt,
            &inv_idx_obj, &var_idx_obj,
            &diag_rows_obj, &diag_cols_obj,
            &offsets_obj, &counts_obj, &entries_obj,
            &idx_xx, &idx_xy, &idx_yy,
            &var_a_obj, &var_b_obj)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError,
            "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }

    /* Convert NumPy arrays (needed for shape validation and data access) */
    PyArrayObject *eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0) {
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }

    PyArrayObject *uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) { Py_DECREF(eigenvalues_arr); return NULL; }

    /* n_inv not yet known — validated inside init_fused_general_workspace */

    PyArrayObject *UtW_arr = (PyArrayObject *)PyArray_FROM_OTF(
        UtW_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!UtW_arr) { Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr); return NULL; }

    if (PyArray_NDIM(UtW_arr) != 2 ||
        PyArray_DIM(UtW_arr, 0) != n_samples ||
        PyArray_DIM(UtW_arr, 1) != n_cvt) {
        PyErr_Format(PyExc_ValueError,
            "UtW must be shape (%d, %d)", n_samples, n_cvt);
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr); Py_DECREF(UtW_arr);
        return NULL;
    }

    PyArrayObject *Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) {
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr); Py_DECREF(UtW_arr);
        return NULL;
    }

    if (PyArray_NDIM(Uty_arr) != 1 ||
        PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Uty must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr);
        return NULL;
    }

    /* Allocate workspace (calloc zeros all pointers for safe cleanup) */
    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)calloc(
        1, sizeof(lmm_workspace_general_t));
    if (!ws) {
        PyErr_NoMemory();
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr);
        return NULL;
    }

    /* Delegate common initialization */
    if (init_fused_general_workspace(
            ws, eigenvalues_arr, uab_inv_arr, UtW_arr, Uty_arr,
            inv_idx_obj, var_idx_obj,
            diag_rows_obj, diag_cols_obj,
            offsets_obj, counts_obj, entries_obj,
            var_a_obj, var_b_obj,
            n_samples, l_min, l_max,
            n_grid, n_refine, n_threads, n_cvt,
            idx_xx, idx_xy, idx_yy) < 0) {
        free_fused_general_workspace(ws);
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr);
        return NULL;
    }

    /* Wrap in PyCapsule */
    PyObject *capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) {
        free_fused_general_workspace(ws);
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr);
        return NULL;
    }

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(UtW_arr);
    Py_DECREF(Uty_arr);
    return capsule;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_fused_general_c
 *
 * Per-chunk Wald compute using fused general workspace. Computes n_var
 * varying dot products on-the-fly from UtW/Uty/UtG_T per SNP, then
 * feeds into the same table-driven Pab recursion + golden section as
 * compute_lmm_chunk_general_c.
 *
 * Python signature:
 *   compute_lmm_chunk_fused_general_c(
 *       workspace,   # PyCapsule from create_workspace_fused_general_c
 *       utg_t,       # (n_snps, n_samples) float64
 *       n_threads,   # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_chunk_fused_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "utg_t", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *utg_t_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &utg_t_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace_general");
    if (!ws) return NULL;

    /* Validate workspace has fused fields */
    if (!ws->UtW || !ws->Uty) {
        PyErr_SetString(PyExc_ValueError,
            "compute_lmm_chunk_fused_general_c requires a fused general workspace "
            "(UtW/Uty pointers are NULL). Use create_workspace_fused_general_c.");
        return NULL;
    }

    PyArrayObject *utg_t_arr = NULL;
    output_arrays_t out = {0};
    PyObject *result = NULL;

    utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) return NULL;

    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_inv = ws->table.n_inv;

    /* Validate shape: must be 2D (n_snps, n_samples) */
    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        goto err_input_fg;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input_fg;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input_fg;
    }

    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input_fg;

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    double *lambdas = (double *)PyArray_DATA(out.lambdas);
    double *logls   = (double *)PyArray_DATA(out.logls);
    double *betas   = (double *)PyArray_DATA(out.betas);
    double *ses     = (double *)PyArray_DATA(out.ses);
    double *pwalds  = (double *)PyArray_DATA(out.pwalds);

    int n_grid = ws->n_grid;
    int n_refine = ws->n_refine;
    int df = ws->table.df;
    int n_index = ws->table.n_index;
    double reml_const = ws->reml_const;

    /* Compute log_l_min and step from lambda_grid */
    double log_l_min = log(ws->lambda_grid[0]);
    double step = (n_grid > 1)
        ? (log(ws->lambda_grid[n_grid - 1]) - log_l_min) / (double)(n_grid - 1)
        : 0.0;

    /* Clamp n_threads */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
    if (actual_threads > ws->actual_threads) actual_threads = ws->actual_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const double *x = utg_t_data + (size_t)snp * n_samples;
        double *scratch = ws->scratch_flat +
            (size_t)tid * (size_t)n_var * (size_t)n_samples;
        double *my_pab = ws->pab_per_thread + (size_t)tid * ws->pab_size;
        double *my_row0 = ws->row0_per_thread + (size_t)tid * n_index;

        /* Compute n_var varying columns on-the-fly */
        for (int v = 0; v < n_var; v++) {
            double *out_v = scratch + (size_t)v * n_samples;
            const double *a = get_fused_vector(ws, ws->var_a_cols[v], x);
            const double *b = get_fused_vector(ws, ws->var_b_cols[v], x);
            #pragma omp simd
            for (int i = 0; i < n_samples; i++)
                out_v[i] = a[i] * b[i];
        }

        /* Compute per-SNP logdet_iab at identity (same as non-fused general).
         * Reuse per-thread heap buffer (consumed before my_row0 needed). */
        double *iab_row0 = my_row0;
        for (int i = 0; i < n_index; i++) iab_row0[i] = 0.0;

        for (int c = 0; c < n_inv; c++)
            iab_row0[ws->table.invariant_indices[c]] = ws->inv_identity_sums[c];
        for (int c = 0; c < n_var; c++) {
            double s = 0.0;
            const double *col = scratch + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += col[i];
            iab_row0[ws->table.varying_indices[c]] = s;
        }

        double logdet_iab = logdet_from_row0(
            iab_row0, &ws->table, ws->table.n_cvt, my_pab);

        /* Golden section optimization — uses scratch as uab_var */
        double logl_opt, beta, se, f_stat;
        int is_valid;
        double lambda_opt = golden_section_lambda_general(
            ws->uab_inv, scratch, ws->eigenvalues,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->inv_sums_grid,
            log_l_min, step, n_grid, n_refine,
            logdet_iab, reml_const, &ws->table,
            &logl_opt, &beta, &se, &f_stat, &is_valid,
            my_row0, my_pab
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;
        pwalds[snp]  = f_to_pvalue(
            f_stat, df, is_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output_fg;

    result = build_result_dict(&out);
    if (!result) goto err_input_fg;

    Py_DECREF(utg_t_arr);
    return result;

err_output_fg:
    decref_output_arrays(&out);
err_input_fg:
    Py_XDECREF(utg_t_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * create_workspace_mode4_fused_general_c
 *
 * Extends fused general workspace with mode-4 fields: hi_eval_null,
 * logl_H0, mle_const, and null_inv_sums for Score and LRT.
 *
 * Python signature: same as create_workspace_fused_general_c plus
 *   hi_eval_null (n_samples,) float64, logl_H0 float
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_mode4_fused_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "UtW", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "n_cvt",
        "invariant_indices", "varying_indices",
        "logdet_diag_rows", "logdet_diag_cols",
        "level_offsets", "level_counts", "entries",
        "idx_xx", "idx_xy", "idx_yy",
        "var_a_cols", "var_b_cols",
        "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *UtW_obj, *Uty_obj;
    PyObject *inv_idx_obj, *var_idx_obj;
    PyObject *diag_rows_obj, *diag_cols_obj;
    PyObject *offsets_obj, *counts_obj, *entries_obj;
    PyObject *var_a_obj, *var_b_obj;
    PyObject *hi_eval_null_obj;
    int n_samples, n_grid, n_refine, n_threads, n_cvt;
    int idx_xx, idx_xy, idx_yy;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiiiiOOOOOOOiiiOOOd", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &UtW_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &n_cvt,
            &inv_idx_obj, &var_idx_obj,
            &diag_rows_obj, &diag_cols_obj,
            &offsets_obj, &counts_obj, &entries_obj,
            &idx_xx, &idx_xy, &idx_yy,
            &var_a_obj, &var_b_obj,
            &hi_eval_null_obj, &logl_H0)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError,
            "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }
    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    /* Convert NumPy arrays */
    PyArrayObject *eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr); return NULL;
    }
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0) {
        Py_DECREF(eigenvalues_arr); return NULL;
    }

    PyArrayObject *uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) { Py_DECREF(eigenvalues_arr); return NULL; }

    PyArrayObject *UtW_arr = (PyArrayObject *)PyArray_FROM_OTF(
        UtW_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!UtW_arr) { Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr); return NULL; }
    if (PyArray_NDIM(UtW_arr) != 2 ||
        PyArray_DIM(UtW_arr, 0) != n_samples || PyArray_DIM(UtW_arr, 1) != n_cvt) {
        PyErr_Format(PyExc_ValueError, "UtW must be shape (%d, %d)", n_samples, n_cvt);
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr); Py_DECREF(UtW_arr);
        return NULL;
    }

    PyArrayObject *Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) {
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr); Py_DECREF(UtW_arr);
        return NULL;
    }
    if (PyArray_NDIM(Uty_arr) != 1 || PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Uty must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); return NULL;
    }

    PyArrayObject *hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) {
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); return NULL;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "hi_eval_null must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); Py_DECREF(hi_eval_null_arr);
        return NULL;
    }

    /* Validate Hi_eval_null */
    {
        const double *hi_null = (const double *)PyArray_DATA(hi_eval_null_arr);
        for (int i = 0; i < n_samples; i++) {
            if (!isfinite(hi_null[i]) || hi_null[i] <= 0.0) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", hi_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not finite positive.", i, buf);
                Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
                Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); Py_DECREF(hi_eval_null_arr);
                return NULL;
            }
        }
    }

    /* Allocate workspace (calloc zeros all pointers for safe cleanup) */
    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)calloc(
        1, sizeof(lmm_workspace_general_t));
    if (!ws) {
        PyErr_NoMemory();
        Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
        Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); Py_DECREF(hi_eval_null_arr);
        return NULL;
    }

    /* Delegate common initialization */
    if (init_fused_general_workspace(
            ws, eigenvalues_arr, uab_inv_arr, UtW_arr, Uty_arr,
            inv_idx_obj, var_idx_obj,
            diag_rows_obj, diag_cols_obj,
            offsets_obj, counts_obj, entries_obj,
            var_a_obj, var_b_obj,
            n_samples, l_min, l_max,
            n_grid, n_refine, n_threads, n_cvt,
            idx_xx, idx_xy, idx_yy) < 0) {
        goto err_ws_m4fg;
    }

    /* Mode-4 specific fields */
    ws->mode = 4;
    ws->logl_H0 = logl_H0;
    ws->mle_const = 0.5 * (double)n_samples
                    * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

    /* Copy hi_eval_null */
    ws->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->hi_eval_null) { PyErr_NoMemory(); goto err_ws_m4fg; }
    memcpy(ws->hi_eval_null,
           (const double *)PyArray_DATA(hi_eval_null_arr),
           (size_t)n_samples * sizeof(double));

    /* Precompute null-model invariant sums */
    int n_inv = ws->table.n_inv;
    ws->null_inv_sums = (double *)malloc((size_t)n_inv * sizeof(double));
    if (!ws->null_inv_sums) { PyErr_NoMemory(); goto err_ws_m4fg; }
    for (int c = 0; c < n_inv; c++) {
        double s = 0.0;
        const double *col = ws->uab_inv + (size_t)c * n_samples;
        for (int i = 0; i < n_samples; i++)
            s += ws->hi_eval_null[i] * col[i];
        ws->null_inv_sums[c] = s;
    }

    /* Pre-allocate per-thread LRT buffer (avoids per-SNP malloc in OpenMP loop).
     * Each thread needs (n_index * n_samples) doubles for row-major uab_snp. */
    {
        int n_index = ws->table.n_index;
        ws->uab_snp_flat = (double *)malloc(
            (size_t)ws->actual_threads * (size_t)n_index
            * (size_t)n_samples * sizeof(double));
        if (!ws->uab_snp_flat) { PyErr_NoMemory(); goto err_ws_m4fg; }
    }

    /* Wrap in PyCapsule */
    PyObject *capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) goto err_ws_m4fg;

    Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
    Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); Py_DECREF(hi_eval_null_arr);
    return capsule;

err_ws_m4fg:
    free_fused_general_workspace(ws);
    Py_DECREF(eigenvalues_arr); Py_DECREF(uab_inv_arr);
    Py_DECREF(UtW_arr); Py_DECREF(Uty_arr); Py_DECREF(hi_eval_null_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_mode4_chunk_fused_general_c
 *
 * Fused per-chunk mode-4 for general n_cvt: Score + Wald + LRT from UtG_T.
 * Computes varying dot products on-the-fly, then uses table-driven Pab
 * recursion for all three statistics.
 *
 * Python signature:
 *   compute_mode4_chunk_fused_general_c(
 *       workspace,   # PyCapsule from create_workspace_mode4_fused_general_c
 *       utg_t,       # (n_snps, n_samples) float64
 *       n_threads,   # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds, p_scores, lambdas_mle, p_lrts}
 * ------------------------------------------------------------------------- */
static PyObject *compute_mode4_chunk_fused_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "utg_t", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *utg_t_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &utg_t_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace_general");
    if (!ws) return NULL;

    if (ws->mode != 4) {
        PyErr_Format(PyExc_ValueError,
            "compute_mode4_chunk_fused_general_c requires a mode-4 workspace "
            "(got mode=%d).", ws->mode);
        return NULL;
    }
    if (!ws->UtW || !ws->Uty) {
        PyErr_SetString(PyExc_ValueError,
            "compute_mode4_chunk_fused_general_c requires a fused general workspace.");
        return NULL;
    }

    PyArrayObject *utg_t_arr = NULL;
    mode4_output_t out = {0};
    PyObject *result = NULL;

    utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) return NULL;

    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_inv = ws->table.n_inv;

    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        goto err_input_m4fg;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input_m4fg;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input_m4fg;
    }

    if (alloc_mode4_output(&out, (npy_intp)n_snps) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        goto err_input_m4fg;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    double *out_lambdas     = (double *)PyArray_DATA(out.lambdas);
    double *out_logls       = (double *)PyArray_DATA(out.logls);
    double *out_betas       = (double *)PyArray_DATA(out.betas);
    double *out_ses         = (double *)PyArray_DATA(out.ses);
    double *out_pwalds      = (double *)PyArray_DATA(out.pwalds);
    double *out_p_scores    = (double *)PyArray_DATA(out.p_scores);
    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    int n_grid = ws->n_grid;
    int n_refine = ws->n_refine;
    int df = ws->table.df;
    int n_index = ws->table.n_index;
    double reml_const = ws->reml_const;

    double log_l_min = log(ws->lambda_grid[0]);
    double step = (n_grid > 1)
        ? (log(ws->lambda_grid[n_grid - 1]) - log_l_min) / (double)(n_grid - 1)
        : 0.0;

    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
    if (actual_threads > ws->actual_threads) actual_threads = ws->actual_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const double *x = utg_t_data + (size_t)snp * n_samples;
        double *scratch = ws->scratch_flat +
            (size_t)tid * (size_t)n_var * (size_t)n_samples;
        double *my_pab = ws->pab_per_thread + (size_t)tid * ws->pab_size;
        double *my_row0 = ws->row0_per_thread + (size_t)tid * n_index;

        /* Compute n_var varying columns on-the-fly */
        for (int v = 0; v < n_var; v++) {
            double *out_v = scratch + (size_t)v * n_samples;
            const double *a = get_fused_vector(ws, ws->var_a_cols[v], x);
            const double *b = get_fused_vector(ws, ws->var_b_cols[v], x);
            #pragma omp simd
            for (int i = 0; i < n_samples; i++)
                out_v[i] = a[i] * b[i];
        }

        /* ---- (a) Score: null-model Pab ---- */
        {
            double *null_row0 = my_row0;  /* reuse per-thread heap buffer */
            for (int i = 0; i < n_index; i++) null_row0[i] = 0.0;

            /* Invariant null sums from precomputed workspace */
            for (int c = 0; c < n_inv; c++)
                null_row0[ws->table.invariant_indices[c]] = ws->null_inv_sums[c];
            /* Varying null sums: weight scratch by hi_eval_null */
            for (int c = 0; c < n_var; c++) {
                double s = 0.0;
                const double *col = scratch + (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    s += ws->hi_eval_null[i] * col[i];
                null_row0[ws->table.varying_indices[c]] = s;
            }

            calc_pab_general(null_row0, &ws->table, my_pab);

            double score_beta, score_se, score_f;
            int score_valid = score_from_pab_general(
                my_pab, &ws->table, n_samples,
                &score_beta, &score_se, &score_f);

            out_p_scores[snp] = f_to_pvalue(
                score_f, df, score_valid,
                ws->beta_a, ws->beta_b, ws->lbeta_ab);
        }

        /* ---- (b) logdet_iab ---- */
        double *iab_row0 = my_row0;  /* reuse per-thread heap buffer */
        for (int i = 0; i < n_index; i++) iab_row0[i] = 0.0;

        for (int c = 0; c < n_inv; c++)
            iab_row0[ws->table.invariant_indices[c]] = ws->inv_identity_sums[c];
        for (int c = 0; c < n_var; c++) {
            double s = 0.0;
            const double *col = scratch + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++) s += col[i];
            iab_row0[ws->table.varying_indices[c]] = s;
        }

        double logdet_iab = logdet_from_row0(
            iab_row0, &ws->table, ws->table.n_cvt, my_pab);

        /* ---- (c) Wald: REML optimization ---- */
        double logl_reml, wald_beta, wald_se, wald_f;
        int wald_valid;
        double lambda_reml = golden_section_lambda_general(
            ws->uab_inv, scratch, ws->eigenvalues,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->inv_sums_grid,
            log_l_min, step, n_grid, n_refine,
            logdet_iab, reml_const, &ws->table,
            &logl_reml, &wald_beta, &wald_se, &wald_f, &wald_valid,
            my_row0, my_pab
        );

        out_lambdas[snp] = lambda_reml;
        out_logls[snp]   = logl_reml;
        out_betas[snp]   = wald_beta;
        out_ses[snp]     = wald_se;
        out_pwalds[snp]  = f_to_pvalue(
            wald_f, df, wald_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);

        /* ---- (d) LRT: MLE optimization ---- */
        {
            /* MLE requires the full (n_samples, n_index) Uab for one SNP
             * in row-major layout (mle_logl_general_cached accesses as
             * uab_snp[sample * n_index + col]).
             * Assemble from ws->uab_inv (invariant) + scratch (varying).
             * Uses pre-allocated per-thread buffer from workspace to avoid
             * per-SNP malloc inside the OpenMP loop. */
            double *uab_snp = ws->uab_snp_flat +
                (size_t)tid * (size_t)n_index * (size_t)n_samples;

            /* Zero fill then scatter invariant and varying columns
             * into row-major layout. */
            memset(uab_snp, 0,
                   (size_t)n_index * (size_t)n_samples * sizeof(double));
            for (int c = 0; c < n_inv; c++) {
                int idx = ws->table.invariant_indices[c];
                const double *src = ws->uab_inv + (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    uab_snp[(size_t)i * n_index + idx] = src[i];
            }
            for (int c = 0; c < n_var; c++) {
                int idx = ws->table.varying_indices[c];
                const double *src = scratch + (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    uab_snp[(size_t)i * n_index + idx] = src[i];
            }

            double logl_H1;
            double lambda_mle = golden_section_lambda_mle_general(
                uab_snp, ws->eigenvalues, n_samples,
                ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
                log_l_min, step, n_grid, n_refine,
                ws->mle_const, &ws->table,
                &logl_H1,
                my_row0, my_pab
            );

            out_lambdas_mle[snp] = lambda_mle;

            double lrt_stat = 2.0 * (logl_H1 - ws->logl_H0);
            if (lrt_stat < 0.0) lrt_stat = 0.0;
            out_p_lrts[snp] = chi2_sf_c(lrt_stat);
        }
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(out_betas, out_pwalds, n_snps) < 0)
        goto err_output_m4fg;

    result = build_mode4_result_dict(&out);
    if (!result) goto err_input_m4fg;

    Py_DECREF(utg_t_arr);
    return result;

err_output_m4fg:
    decref_mode4_output(&out);
err_input_m4fg:
    Py_XDECREF(utg_t_arr);
    return NULL;
}

/* =========================================================================
 * SoA-NATIVE SCORE SPLIT — compute_score_split_c
 *
 * Score test accepting SoA split data (uab_varying_soa + uab_invariant_soa)
 * instead of full Uab. Eliminates the need for reconstruct_uab_from_soa.
 *
 * Mirrors compute_score_batch_c but computes Pab dot products from split
 * arrays using the same pattern as the fused mode-4 Score section.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_score_split_c
 *
 * Args: eigenvalues (n_samples,), uab_varying_soa (n_snps, 3, n_samples),
 *       uab_invariant_soa (3, n_samples), Hi_eval_null (n_samples,),
 *       n_samples, n_threads
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_split_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_var_obj, *uab_inv_obj, *hi_eval_null_obj;
    int n_samples, n_threads;
    PyArrayObject *eigenvalues_arr = NULL, *uab_var_arr = NULL;
    PyArrayObject *uab_inv_arr = NULL, *hi_eval_null_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOOii",
            &eigenvalues_obj, &uab_var_obj, &uab_inv_obj,
            &hi_eval_null_obj, &n_samples, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_score_split;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) goto err_input_score_split;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input_score_split;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input_score_split;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input_score_split;
    }
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying_soa must be shape (n_snps, 3, %d)", n_samples);
        goto err_input_score_split;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (3, %d)", n_samples);
        goto err_input_score_split;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Hi_eval_null must be shape (n_samples,)");
        goto err_input_score_split;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input_score_split;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input_score_split;
    }

    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
    const double *uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);
    const double *hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0)
        goto err_input_score_split;

    /* Validate Hi_eval_null */
    for (int i = 0; i < n_samples; i++) {
        char buf[64];
        if (!isfinite(hi_eval_null[i])) {
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not finite. "
                "Null model optimization may have failed.", i, buf);
            goto err_input_score_split;
        }
        if (hi_eval_null[i] <= 0.0) {
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not positive. "
                "Check kinship matrix conditioning.", i, buf);
            goto err_input_score_split;
        }
    }

    /* Invariant SoA pointers: rows [ww, wy, yy] */
    const double *inv_ww = uab_inv_data;
    const double *inv_wy = uab_inv_data + (size_t)n_samples;
    const double *inv_yy = uab_inv_data + (size_t)2 * n_samples;

    /* Pre-compute invariant null-model dot products (shared across SNPs) */
    double null_s_ww = 0.0, null_s_wy = 0.0, null_s_yy = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval_null[i];
        null_s_ww += h * inv_ww[i];
        null_s_wy += h * inv_wy[i];
        null_s_yy += h * inv_yy[i];
    }
    /* Allocate output arrays */
    score_output_t out;
    if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        goto err_input_score_split;
    }

    double *out_betas    = (double *)PyArray_DATA(out.betas);
    double *out_ses      = (double *)PyArray_DATA(out.ses);
    double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

    /* Pre-compute F-distribution constants */
    int df = n_samples - 2;  /* n_cvt=1: df = n - n_cvt - 1 */
    double a = (double)df / 2.0;
    double b = 0.5;
    double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);

    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        const double *snp_base = uab_var_data + (size_t)s * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;

        /* Compute varying null-model dot products */
        double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
        #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
        for (int i = 0; i < n_samples; i++) {
            double h = hi_eval_null[i];
            s_wx += h * vwx[i];
            s_xx += h * vxx[i];
            s_xy += h * vxy[i];
        }

        /* Build Pab from split sums */
        double pab[3][6];
        calc_pab_ncvt1_split(null_s_ww, s_wx, null_s_wy,
                              s_xx, s_xy, null_s_yy, pab);

        double beta, se, f_stat;
        int is_valid = score_from_pab(pab, n_samples, df, &beta, &se, &f_stat);

        out_betas[s] = beta;
        out_ses[s] = se;
        out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
        decref_score_output(&out);
        Py_DECREF(hi_eval_null_arr);
        Py_DECREF(uab_inv_arr);
        Py_DECREF(uab_var_arr);
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }

    Py_DECREF(hi_eval_null_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(uab_var_arr);
    Py_DECREF(eigenvalues_arr);
    return build_score_result_dict(&out);

err_input_score_split:
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * FUSED SCORE — compute_score_fused_c
 *
 * Score test accepting utg_t directly instead of pre-materialized
 * uab_varying_soa. Computes wx/xx/xy dot products on-the-fly from
 * utg_t columns, eliminating the 3x intermediate buffer.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_score_fused_c
 *
 * Args: utg_t (n_snps, n_samples), w (n_samples,), Uty (n_samples,),
 *       Hi_eval_null (n_samples,), uab_invariant_soa (3, n_samples),
 *       eigenvalues (n_samples,), n_samples, n_threads
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_fused_c(PyObject *self, PyObject *args)
{
    PyObject *utg_t_obj, *w_obj, *Uty_obj, *hi_eval_null_obj;
    PyObject *uab_inv_obj, *eigenvalues_obj;
    int n_samples, n_threads;
    PyArrayObject *utg_t_arr = NULL, *w_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *eigenvalues_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOOOOii",
            &utg_t_obj, &w_obj, &Uty_obj, &hi_eval_null_obj,
            &uab_inv_obj, &eigenvalues_obj, &n_samples, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) goto err_input_score_fused;

    w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) goto err_input_score_fused;

    Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) goto err_input_score_fused;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input_score_fused;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input_score_fused;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_score_fused;

    /* Validate shapes */
    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        goto err_input_score_fused;
    }
    if (PyArray_NDIM(w_arr) != 1 || PyArray_DIM(w_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "w must be shape (n_samples,)");
        goto err_input_score_fused;
    }
    if (PyArray_NDIM(Uty_arr) != 1 || PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Uty must be shape (n_samples,)");
        goto err_input_score_fused;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Hi_eval_null must be shape (n_samples,)");
        goto err_input_score_fused;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (3, %d)", n_samples);
        goto err_input_score_fused;
    }
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input_score_fused;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input_score_fused;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input_score_fused;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);
    const double *w_data = (const double *)PyArray_DATA(w_arr);
    const double *Uty_data = (const double *)PyArray_DATA(Uty_arr);
    const double *hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);
    const double *uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);
    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0)
        goto err_input_score_fused;

    /* Validate Hi_eval_null */
    for (int i = 0; i < n_samples; i++) {
        char buf[64];
        if (!isfinite(hi_eval_null[i])) {
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not finite. "
                "Null model optimization may have failed.", i, buf);
            goto err_input_score_fused;
        }
        if (hi_eval_null[i] <= 0.0) {
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not positive. "
                "Check kinship matrix conditioning.", i, buf);
            goto err_input_score_fused;
        }
    }

    /* Invariant SoA pointers: rows [ww, wy, yy] */
    const double *inv_ww = uab_inv_data;
    const double *inv_wy = uab_inv_data + (size_t)n_samples;
    const double *inv_yy = uab_inv_data + (size_t)2 * n_samples;

    /* Precompute SNP-invariant vectors: h_null * w and h_null * Uty */
    double *h_null_w = alloc_aligned_doubles((size_t)n_samples);
    double *h_null_Uty = alloc_aligned_doubles((size_t)n_samples);
    if (!h_null_w || !h_null_Uty) {
        free(h_null_w);
        free(h_null_Uty);
        PyErr_NoMemory();
        goto err_input_score_fused;
    }
    for (int i = 0; i < n_samples; i++) {
        h_null_w[i]   = hi_eval_null[i] * w_data[i];
        h_null_Uty[i] = hi_eval_null[i] * Uty_data[i];
    }

    /* Pre-compute invariant null-model dot products (shared across SNPs) */
    double null_s_ww = 0.0, null_s_wy = 0.0, null_s_yy = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval_null[i];
        null_s_ww += h * inv_ww[i];
        null_s_wy += h * inv_wy[i];
        null_s_yy += h * inv_yy[i];
    }

    /* Allocate output arrays */
    score_output_t out;
    if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
        free(h_null_w);
        free(h_null_Uty);
        PyErr_NoMemory();
        goto err_input_score_fused;
    }

    double *out_betas    = (double *)PyArray_DATA(out.betas);
    double *out_ses      = (double *)PyArray_DATA(out.ses);
    double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

    /* Pre-compute F-distribution constants */
    int df = n_samples - 2;  /* n_cvt=1: df = n - n_cvt - 1 */
    double a = (double)df / 2.0;
    double b = 0.5;
    double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);

    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        const double *x = utg_t_data + (size_t)s * n_samples;

        /* Compute varying null-model dot products on-the-fly from utg_t */
        double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
        #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
        for (int i = 0; i < n_samples; i++) {
            s_wx += h_null_w[i]   * x[i];
            s_xx += hi_eval_null[i] * x[i] * x[i];
            s_xy += h_null_Uty[i] * x[i];
        }

        /* Build Pab from split sums */
        double pab[3][6];
        calc_pab_ncvt1_split(null_s_ww, s_wx, null_s_wy,
                              s_xx, s_xy, null_s_yy, pab);

        double beta, se, f_stat;
        int is_valid = score_from_pab(pab, n_samples, df, &beta, &se, &f_stat);

        out_betas[s] = beta;
        out_ses[s] = se;
        out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    free(h_null_w);
    free(h_null_Uty);

    if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
        decref_score_output(&out);
        Py_DECREF(eigenvalues_arr);
        Py_DECREF(uab_inv_arr);
        Py_DECREF(hi_eval_null_arr);
        Py_DECREF(Uty_arr);
        Py_DECREF(w_arr);
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(hi_eval_null_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(w_arr);
    Py_DECREF(utg_t_arr);
    return build_score_result_dict(&out);

err_input_score_fused:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(w_arr);
    Py_XDECREF(utg_t_arr);
    return NULL;
}

/* =========================================================================
 * PERSISTENT SCORE WORKSPACE — create_workspace_score_fused_c / compute_score_fused_ws_c
 *
 * Moves all SNP-invariant state into a PyCapsule workspace, eliminating
 * per-chunk malloc/free and redundant precomputation of h_null_w, h_null_Uty,
 * null dot products, and F-distribution constants.
 * ========================================================================= */

typedef struct {
    int n_samples;
    int df;
    double a, b, lbeta_ab;
    /* Precomputed invariant vectors (owned) */
    double *h_null_w;       /* (n_samples,) hi_eval_null * w */
    double *h_null_Uty;     /* (n_samples,) hi_eval_null * Uty */
    /* Precomputed invariant dot products */
    double null_s_ww, null_s_wy, null_s_yy;
    /* Raw data pointers into INCREF'd arrays (refs owned by workspace) */
    const double *hi_eval_null;
    const double *uab_inv_data;
    const double *eigenvalues;
    PyObject *hi_eval_null_ref;
    PyObject *uab_inv_ref;
    PyObject *eigenvalues_ref;
} lmm_workspace_score_t;

static void lmm_workspace_score_destructor(PyObject *cap)
{
    lmm_workspace_score_t *ws =
        (lmm_workspace_score_t *)PyCapsule_GetPointer(cap, "lmm_workspace_score_fused");
    if (!ws) return;
    free(ws->h_null_w);
    free(ws->h_null_Uty);
    Py_XDECREF(ws->hi_eval_null_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->eigenvalues_ref);
    free(ws);
}

/* -------------------------------------------------------------------------
 * create_workspace_score_fused_c
 *
 * Python signature:
 *   create_workspace_score_fused_c(
 *       w,                # (n_samples,) float64
 *       Uty,              # (n_samples,) float64
 *       Hi_eval_null,     # (n_samples,) float64
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant_soa,# (3, n_samples) float64
 *       n_samples,        # int
 *       n_threads,        # int
 *   ) -> PyCapsule wrapping lmm_workspace_score_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_score_fused_c_py(
    PyObject *self, PyObject *args)
{
    PyObject *w_obj, *Uty_obj, *hi_eval_null_obj;
    PyObject *eigenvalues_obj, *uab_inv_obj;
    int n_samples, n_threads;

    if (!PyArg_ParseTuple(args, "OOOOOii",
            &w_obj, &Uty_obj, &hi_eval_null_obj,
            &eigenvalues_obj, &uab_inv_obj, &n_samples, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }

    PyArrayObject *w_arr = NULL, *Uty_arr = NULL, *hi_eval_null_arr = NULL;
    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    lmm_workspace_score_t *ws = NULL;

    w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) return NULL;

    Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) goto err_score_ws_create;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_score_ws_create;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_score_ws_create;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_score_ws_create;

    /* Validate shapes */
    if (PyArray_NDIM(w_arr) != 1 || PyArray_DIM(w_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "w must be shape (n_samples,)");
        goto err_score_ws_create;
    }
    if (PyArray_NDIM(Uty_arr) != 1 || PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Uty must be shape (n_samples,)");
        goto err_score_ws_create;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Hi_eval_null must be shape (n_samples,)");
        goto err_score_ws_create;
    }
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_score_ws_create;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (3, %d)", n_samples);
        goto err_score_ws_create;
    }

    /* Validate Hi_eval_null values */
    {
        const double *hi = (const double *)PyArray_DATA(hi_eval_null_arr);
        for (int i = 0; i < n_samples; i++) {
            if (!isfinite(hi[i]) || hi[i] <= 0.0) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", hi[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not finite positive.", i, buf);
                goto err_score_ws_create;
            }
        }
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_score_ws_create;

    /* Allocate workspace */
    ws = (lmm_workspace_score_t *)calloc(1, sizeof(lmm_workspace_score_t));
    if (!ws) { PyErr_NoMemory(); goto err_score_ws_create; }

    ws->n_samples = n_samples;
    ws->df = n_samples - 2;
    ws->a = (double)ws->df / 2.0;
    ws->b = 0.5;
    ws->lbeta_ab = lgamma(ws->a) + lgamma(ws->b) - lgamma(ws->a + ws->b);

    /* Precompute h_null_w and h_null_Uty */
    ws->h_null_w = alloc_aligned_doubles((size_t)n_samples);
    ws->h_null_Uty = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->h_null_w || !ws->h_null_Uty) {
        PyErr_NoMemory();
        goto err_score_ws_alloc;
    }

    {
        const double *w_data = (const double *)PyArray_DATA(w_arr);
        const double *Uty_data = (const double *)PyArray_DATA(Uty_arr);
        const double *hi = (const double *)PyArray_DATA(hi_eval_null_arr);
        const double *inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
        const double *inv_wy = inv_ww + (size_t)n_samples;
        const double *inv_yy = inv_ww + (size_t)2 * n_samples;

        for (int i = 0; i < n_samples; i++) {
            ws->h_null_w[i]   = hi[i] * w_data[i];
            ws->h_null_Uty[i] = hi[i] * Uty_data[i];
        }

        /* Precompute invariant null-model dot products */
        double s_ww = 0.0, s_wy = 0.0, s_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double h = hi[i];
            s_ww += h * inv_ww[i];
            s_wy += h * inv_wy[i];
            s_yy += h * inv_yy[i];
        }
        ws->null_s_ww = s_ww;
        ws->null_s_wy = s_wy;
        ws->null_s_yy = s_yy;
    }

    /* Borrow array pointers via Py_INCREF */
    Py_INCREF(hi_eval_null_arr);
    Py_INCREF(uab_inv_arr);
    Py_INCREF(eigenvalues_arr);
    ws->hi_eval_null_ref = (PyObject *)hi_eval_null_arr;
    ws->uab_inv_ref      = (PyObject *)uab_inv_arr;
    ws->eigenvalues_ref  = (PyObject *)eigenvalues_arr;
    ws->hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);
    ws->uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);
    ws->eigenvalues  = (const double *)PyArray_DATA(eigenvalues_arr);

    /* Release OTF refs — workspace holds its own Py_INCREF'd refs */
    Py_DECREF(hi_eval_null_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(eigenvalues_arr);

    /* Release input arrays that are NOT stored in workspace */
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);

    PyObject *capsule = PyCapsule_New(ws, "lmm_workspace_score_fused",
                                      lmm_workspace_score_destructor);
    if (!capsule) goto err_score_ws_alloc;
    return capsule;

err_score_ws_alloc:
    free(ws->h_null_w);
    free(ws->h_null_Uty);
    /* Defensive: XDECREF borrowed refs (NULL from calloc if INCREF not yet reached) */
    Py_XDECREF(ws->hi_eval_null_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->eigenvalues_ref);
    free(ws);
err_score_ws_create:
    Py_XDECREF(w_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_score_fused_ws_c
 *
 * Python signature:
 *   compute_score_fused_ws_c(workspace, utg_t, n_threads)
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_fused_ws_c_py(PyObject *self, PyObject *args)
{
    PyObject *capsule_obj, *utg_t_obj;
    int n_threads;

    if (!PyArg_ParseTuple(args, "OOi", &capsule_obj, &utg_t_obj, &n_threads))
        return NULL;

    lmm_workspace_score_t *ws = (lmm_workspace_score_t *)
        PyCapsule_GetPointer(capsule_obj, "lmm_workspace_score_fused");
    if (!ws) return NULL;  /* PyCapsule_GetPointer sets ValueError on name mismatch */

    PyArrayObject *utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) return NULL;

    int n_samples = ws->n_samples;

    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        Py_DECREF(utg_t_arr);
        return NULL;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    score_output_t out;
    if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    double *out_betas    = (double *)PyArray_DATA(out.betas);
    double *out_ses      = (double *)PyArray_DATA(out.ses);
    double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

    /* Read precomputed invariants from workspace */
    const double *h_null_w   = ws->h_null_w;
    const double *h_null_Uty = ws->h_null_Uty;
    const double *hi_eval_null = ws->hi_eval_null;
    double null_s_ww = ws->null_s_ww;
    double null_s_wy = ws->null_s_wy;
    double null_s_yy = ws->null_s_yy;
    int df       = ws->df;
    double a     = ws->a;
    double b_val = ws->b;
    double lbeta_ab = ws->lbeta_ab;

    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        const double *x = utg_t_data + (size_t)s * n_samples;

        /* Compute varying null-model dot products on-the-fly from utg_t */
        double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
        #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
        for (int i = 0; i < n_samples; i++) {
            s_wx += h_null_w[i]   * x[i];
            s_xx += hi_eval_null[i] * x[i] * x[i];
            s_xy += h_null_Uty[i] * x[i];
        }

        /* Build Pab from split sums */
        double pab[3][6];
        calc_pab_ncvt1_split(null_s_ww, s_wx, null_s_wy,
                              s_xx, s_xy, null_s_yy, pab);

        double beta, se, f_stat;
        int is_valid = score_from_pab(pab, n_samples, df, &beta, &se, &f_stat);

        out_betas[s] = beta;
        out_ses[s] = se;
        out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b_val, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
        decref_score_output(&out);
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    Py_DECREF(utg_t_arr);
    return build_score_result_dict(&out);
}

/* =========================================================================
 * SoA-NATIVE LRT SPLIT — compute_lrt_split_c
 *
 * LRT test accepting SoA split data (uab_varying_soa + uab_invariant_soa)
 * instead of full Uab. Eliminates the need for reconstruct_uab_from_soa.
 *
 * Mirrors compute_lrt_batch_c but computes MLE log-likelihood from split
 * arrays using the same pattern as golden_section_lambda_mle_ncvt1_split.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_lrt_split_c
 *
 * Args: eigenvalues (n_samples,), uab_varying_soa (n_snps, 3, n_samples),
 *       uab_invariant_soa (3, n_samples), n_samples,
 *       l_min, l_max, n_grid, n_refine, logl_H0, n_threads
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_split_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_var_obj, *uab_inv_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTuple(args, "OOOiddiidi",
            &eigenvalues_obj, &uab_var_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    PyArrayObject *eigenvalues_arr = NULL, *uab_var_arr = NULL, *uab_inv_arr = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_lrt_split;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) goto err_input_lrt_split;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input_lrt_split;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input_lrt_split;
    }
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying_soa must be shape (n_snps, 3, %d)", n_samples);
        goto err_input_lrt_split;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (3, %d)", n_samples);
        goto err_input_lrt_split;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input_lrt_split;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input_lrt_split;
    }

    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
    const double *uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0)
        goto err_input_lrt_split;

    /* Invariant SoA pointers: rows [ww, wy, yy] */
    const double *inv_ww = uab_inv_data;
    const double *inv_wy = uab_inv_data + (size_t)n_samples;
    const double *inv_yy = uab_inv_data + (size_t)2 * n_samples;

    /* Allocate output arrays */
    lrt_output_t out;
    if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        goto err_input_lrt_split;
    }

    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    /* MLE constant and grid */
    double n = (double)n_samples;
    double mle_const = 0.5 * n * (log(n) - log(2.0 * M_PI) - 1.0);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

    /* Build lambda grid */
    double *lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) {
        decref_lrt_output(&out);
        goto err_nomem_lrt_split;
    }
    for (int g = 0; g < n_grid; g++)
        lambda_grid[g] = exp(log_l_min + g * step);

    /* Pre-compute hi_eval_grid, logdet_h_grid, and grid_inv (invariant sums) */
    double *hi_eval_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_samples * sizeof(double));
    double *logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid_invariant_t *grid_inv = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));
    if (!hi_eval_grid || !logdet_h_grid || !grid_inv) {
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        free(grid_inv);
        decref_lrt_output(&out);
        goto err_nomem_lrt_split;
    }
    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double gs_ww = 0.0, gs_wy = 0.0, gs_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi[i] = h;
            logdet += log(v);
            gs_ww += h * inv_ww[i];
            gs_wy += h * inv_wy[i];
            gs_yy += h * inv_yy[i];
        }
        logdet_h_grid[g] = logdet;
        grid_inv[g].s_ww = gs_ww;
        grid_inv[g].s_wy = gs_wy;
        grid_inv[g].s_yy = gs_yy;
        grid_inv[g].log_s_ww = (gs_ww > 0.0) ? log(gs_ww) : 0.0;
    }

    /* Pre-allocate per-thread hi_eval buffers */
    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    if (!thread_bufs) {
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        free(grid_inv);
        decref_lrt_output(&out);
        goto err_nomem_lrt_split;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];

        const double *snp_base = uab_var_data + (size_t)s * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;

        double logl_H1;
        double lam_mle = golden_section_lambda_mle_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            eigenvalues, n_samples,
            lambda_grid, hi_eval_grid, logdet_h_grid,
            grid_inv, log_l_min, step, n_grid, n_refine,
            mle_const, hi_eval_local, &logl_H1
        );
        out_lambdas_mle[s] = lam_mle;

        double lrt_stat = 2.0 * (logl_H1 - logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[s] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    free_thread_scratch(thread_bufs, actual_threads);

    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    free(grid_inv);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(uab_var_arr);
    Py_DECREF(eigenvalues_arr);

    return build_lrt_result_dict(&out);

err_nomem_lrt_split:
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(eigenvalues_arr);
    return PyErr_NoMemory();

err_input_lrt_split:
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * SoA-NATIVE GENERAL SCORE SPLIT — compute_score_split_general_c
 *
 * Score test for arbitrary n_cvt accepting SoA split data
 * (uab_varying_soa + uab_invariant_soa + pab_table_dict) directly.
 * Eliminates the need for reconstruct_uab_from_soa + batch dispatch.
 *
 * Mirrors the Score section of compute_mode4_chunk_fused_general_c but
 * reads from pre-computed SoA arrays instead of fused UtW/Uty vectors.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_score_split_general_c
 *
 * Args: eigenvalues (n_samples,), uab_varying_soa (n_snps, n_var, n_samples),
 *       uab_invariant_soa (n_inv, n_samples), Hi_eval_null (n_samples,),
 *       n_samples, n_cvt, pab_table_dict, n_threads
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_split_general_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_var_obj, *uab_inv_obj, *hi_eval_null_obj;
    PyObject *pab_table_dict;
    int n_samples, n_cvt, n_threads;
    PyArrayObject *eigenvalues_arr = NULL, *uab_var_arr = NULL;
    PyArrayObject *uab_inv_arr = NULL, *hi_eval_null_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOOiiOi",
            &eigenvalues_obj, &uab_var_obj, &uab_inv_obj,
            &hi_eval_null_obj, &n_samples, &n_cvt,
            &pab_table_dict, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }
    if (!PyDict_Check(pab_table_dict)) {
        PyErr_SetString(PyExc_TypeError, "pab_table_dict must be a dict");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_score_split_gen;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) goto err_input_score_split_gen;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input_score_split_gen;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input_score_split_gen;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input_score_split_gen;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Hi_eval_null must be shape (n_samples,)");
        goto err_input_score_split_gen;
    }

    /* Parse pab_table first to get n_inv, n_var for shape validation */
    pab_table_t table;
    if (parse_pab_table_from_dict(pab_table_dict, &table, n_samples) < 0)
        goto err_input_score_split_gen;

    /* Validate SoA array shapes against pab_table dimensions */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != table.n_var ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying_soa must be shape (n_snps, %d, %d)",
            table.n_var, n_samples);
        free_pab_table(&table);
        goto err_input_score_split_gen;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != table.n_inv ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (%d, %d)",
            table.n_inv, n_samples);
        free_pab_table(&table);
        goto err_input_score_split_gen;
    }

    {
        npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
        if (n_snps_raw > INT_MAX || n_snps_raw == 0) {
            PyErr_SetString(PyExc_ValueError, "n_snps must be > 0 and <= INT_MAX");
            free_pab_table(&table);
            goto err_input_score_split_gen;
        }
        int n_snps = (int)n_snps_raw;

        const double *eigenvalues  = (const double *)PyArray_DATA(eigenvalues_arr);
        const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
        const double *uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);
        const double *hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);

        if (validate_eigenvalues(eigenvalues, n_samples) < 0) {
            free_pab_table(&table);
            goto err_input_score_split_gen;
        }

        /* Validate Hi_eval_null */
        for (int i = 0; i < n_samples; i++) {
            if (!isfinite(hi_eval_null[i]) || hi_eval_null[i] <= 0.0) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "Hi_eval_null[%d] = %s is not finite/positive.", i, buf);
                free_pab_table(&table);
                goto err_input_score_split_gen;
            }
        }

        int n_inv = table.n_inv;
        int n_var = table.n_var;
        int n_index = table.n_index;

        /* Pre-compute invariant null-model dot products (shared across SNPs) */
        double inv_null_sums[MAX_N_INDEX];
        for (int c = 0; c < n_inv; c++) {
            double s = 0.0;
            const double *col = uab_inv_data + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += hi_eval_null[i] * col[i];
            inv_null_sums[c] = s;
        }

        /* Allocate outputs */
        score_output_t out;
        if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
            free_pab_table(&table);
            PyErr_NoMemory();
            goto err_input_score_split_gen;
        }

        double *out_betas    = (double *)PyArray_DATA(out.betas);
        double *out_ses      = (double *)PyArray_DATA(out.ses);
        double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

        /* F-distribution constants */
        int df = table.df;
        double a = (double)df / 2.0;
        double b = 0.5;
        double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);

        int actual_threads = 1;
#ifdef _OPENMP
        if (n_threads > 0) {
            actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
        } else {
            actual_threads = omp_get_max_threads();
            if (actual_threads > n_snps) actual_threads = n_snps;
        }
        if (actual_threads < 1) actual_threads = 1;
#else
        (void)n_threads;
#endif

        /* Per-thread heap buffers for Pab recursion and row0 */
        int ssg_pab_size = table.n_rows * n_index;
        double *ssg_pab_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)ssg_pab_size * sizeof(double));
        double *ssg_row0_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)n_index * sizeof(double));
        if (!ssg_pab_heap || !ssg_row0_heap) {
            free(ssg_pab_heap);
            free(ssg_row0_heap);
            free_pab_table(&table);
            decref_score_output(&out);
            Py_DECREF(hi_eval_null_arr);
            Py_DECREF(uab_inv_arr); Py_DECREF(uab_var_arr);
            Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }

        Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
        for (int s = 0; s < n_snps; s++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            double *my_pab = ssg_pab_heap + (size_t)tid * ssg_pab_size;

            /* Build null_row0 for this SNP (per-thread heap buffer) */
            double *null_row0 = ssg_row0_heap + (size_t)tid * n_index;
            for (int c = 0; c < n_index; c++) null_row0[c] = 0.0;

            /* Place invariant null sums at their indices */
            for (int c = 0; c < n_inv; c++)
                null_row0[table.invariant_indices[c]] = inv_null_sums[c];

            /* Compute varying null sums: weight varying SoA by hi_eval_null */
            for (int c = 0; c < n_var; c++) {
                double sv = 0.0;
                const double *col = uab_var_data +
                    (size_t)s * n_var * n_samples +
                    (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    sv += hi_eval_null[i] * col[i];
                null_row0[table.varying_indices[c]] = sv;
            }

            /* Full Pab via table-driven recursion */
            calc_pab_general(null_row0, &table, my_pab);

            double beta, se, f_stat;
            int is_valid = score_from_pab_general(my_pab, &table, n_samples,
                                                  &beta, &se, &f_stat);

            out_betas[s]    = beta;
            out_ses[s]      = se;
            out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b, lbeta_ab);
        }

        Py_END_ALLOW_THREADS
        free(ssg_pab_heap);
        free(ssg_row0_heap);

        free_pab_table(&table);

        if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
            decref_score_output(&out);
            Py_DECREF(hi_eval_null_arr);
            Py_DECREF(uab_inv_arr);
            Py_DECREF(uab_var_arr);
            Py_DECREF(eigenvalues_arr);
            return NULL;
        }

        Py_DECREF(hi_eval_null_arr);
        Py_DECREF(uab_inv_arr);
        Py_DECREF(uab_var_arr);
        Py_DECREF(eigenvalues_arr);
        return build_score_result_dict(&out);
    }

err_input_score_split_gen:
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * SoA-NATIVE GENERAL LRT SPLIT — compute_lrt_split_general_c
 *
 * LRT test for arbitrary n_cvt accepting SoA split data
 * (uab_varying_soa + uab_invariant_soa + pab_table_dict) directly.
 * Assembles per-SNP uab_snp in row-major layout for mle_logl_general.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_lrt_split_general_c
 *
 * Args: eigenvalues (n_samples,), uab_varying_soa (n_snps, n_var, n_samples),
 *       uab_invariant_soa (n_inv, n_samples), n_samples, n_cvt,
 *       pab_table_dict, l_min, l_max, n_grid, n_refine, logl_H0, n_threads
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_split_general_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_var_obj, *uab_inv_obj, *pab_table_dict;
    int n_samples, n_cvt, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;
    PyArrayObject *eigenvalues_arr = NULL, *uab_var_arr = NULL, *uab_inv_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOiiOddiidi",
            &eigenvalues_obj, &uab_var_obj, &uab_inv_obj,
            &n_samples, &n_cvt,
            &pab_table_dict,
            &l_min, &l_max,
            &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }
    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }
    if (!PyDict_Check(pab_table_dict)) {
        PyErr_SetString(PyExc_TypeError, "pab_table_dict must be a dict");
        return NULL;
    }

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_lrt_split_gen;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) goto err_input_lrt_split_gen;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input_lrt_split_gen;

    /* Parse pab_table first for shape validation */
    pab_table_t table;
    if (parse_pab_table_from_dict(pab_table_dict, &table, n_samples) < 0)
        goto err_input_lrt_split_gen;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        free_pab_table(&table);
        goto err_input_lrt_split_gen;
    }
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != table.n_var ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying_soa must be shape (n_snps, %d, %d)",
            table.n_var, n_samples);
        free_pab_table(&table);
        goto err_input_lrt_split_gen;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != table.n_inv ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (%d, %d)",
            table.n_inv, n_samples);
        free_pab_table(&table);
        goto err_input_lrt_split_gen;
    }

    {
        npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
        if (n_snps_raw > INT_MAX || n_snps_raw == 0) {
            PyErr_SetString(PyExc_ValueError, "n_snps must be > 0 and <= INT_MAX");
            free_pab_table(&table);
            goto err_input_lrt_split_gen;
        }
        int n_snps = (int)n_snps_raw;

        const double *eigenvalues  = (const double *)PyArray_DATA(eigenvalues_arr);
        const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
        const double *uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);

        if (validate_eigenvalues(eigenvalues, n_samples) < 0) {
            free_pab_table(&table);
            goto err_input_lrt_split_gen;
        }

        int n_inv = table.n_inv;
        int n_var = table.n_var;
        int n_index = table.n_index;

        /* Allocate outputs */
        lrt_output_t out;
        if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
            free_pab_table(&table);
            PyErr_NoMemory();
            goto err_input_lrt_split_gen;
        }

        double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
        double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

        /* Pre-compute MLE constant and lambda grid */
        double n_d = (double)n_samples;
        double mle_const = 0.5 * n_d * (log(n_d) - log(2.0 * M_PI) - 1.0);

        double log_l_min = log(l_min);
        double log_l_max = log(l_max);
        double step_val = (log_l_max - log_l_min) / (double)(n_grid - 1);

        double *lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
        if (!lambda_grid) {
            decref_lrt_output(&out);
            free_pab_table(&table);
            Py_DECREF(uab_inv_arr); Py_DECREF(uab_var_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }
        for (int g = 0; g < n_grid; g++)
            lambda_grid[g] = exp(log_l_min + g * step_val);

        /* Pre-compute hi_eval_grid and logdet_h_grid */
        double *hi_eval_grid = (double *)malloc(
            (size_t)n_grid * (size_t)n_samples * sizeof(double));
        double *logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
        if (!hi_eval_grid || !logdet_h_grid) {
            free(lambda_grid); free(hi_eval_grid); free(logdet_h_grid);
            decref_lrt_output(&out);
            free_pab_table(&table);
            Py_DECREF(uab_inv_arr); Py_DECREF(uab_var_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }
        for (int g = 0; g < n_grid; g++) {
            double lam = lambda_grid[g];
            double *hi = hi_eval_grid + (size_t)g * n_samples;
            double logdet = 0.0;
            for (int i = 0; i < n_samples; i++) {
                double v = lam * eigenvalues[i] + 1.0;
                hi[i] = 1.0 / v;
                logdet += log(v);
            }
            logdet_h_grid[g] = logdet;
        }

        int actual_threads = 1;
#ifdef _OPENMP
        if (n_threads > 0) {
            actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
        } else {
            actual_threads = omp_get_max_threads();
            if (actual_threads > n_snps) actual_threads = n_snps;
        }
        if (actual_threads < 1) actual_threads = 1;
#else
        (void)n_threads;
#endif

        /* Allocate per-thread uab_snp + Pab recursion buffers */
        int lsg_pab_size = table.n_rows * n_index;
        double *uab_snp_flat = (double *)malloc(
            (size_t)actual_threads * (size_t)n_index * (size_t)n_samples * sizeof(double));
        double *lsg_pab_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)lsg_pab_size * sizeof(double));
        double *lsg_row0_heap = (double *)malloc(
            (size_t)actual_threads * (size_t)n_index * sizeof(double));
        if (!uab_snp_flat || !lsg_pab_heap || !lsg_row0_heap) {
            free(uab_snp_flat); free(lsg_pab_heap); free(lsg_row0_heap);
            free(lambda_grid); free(hi_eval_grid); free(logdet_h_grid);
            decref_lrt_output(&out);
            free_pab_table(&table);
            Py_DECREF(uab_inv_arr); Py_DECREF(uab_var_arr); Py_DECREF(eigenvalues_arr);
            return PyErr_NoMemory();
        }

        Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
        for (int s = 0; s < n_snps; s++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            double *my_pab = lsg_pab_heap + (size_t)tid * lsg_pab_size;
            double *my_row0 = lsg_row0_heap + (size_t)tid * n_index;

            /* Assemble per-SNP uab_snp in row-major (n_samples, n_index) layout
             * matching mle_logl_general_cached expectation. */
            double *uab_snp = uab_snp_flat +
                (size_t)tid * (size_t)n_index * (size_t)n_samples;

            memset(uab_snp, 0,
                   (size_t)n_index * (size_t)n_samples * sizeof(double));

            /* Scatter invariant columns */
            for (int c = 0; c < n_inv; c++) {
                int idx = table.invariant_indices[c];
                const double *src = uab_inv_data + (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    uab_snp[(size_t)i * n_index + idx] = src[i];
            }
            /* Scatter varying columns */
            for (int c = 0; c < n_var; c++) {
                int idx = table.varying_indices[c];
                const double *src = uab_var_data +
                    (size_t)s * n_var * n_samples +
                    (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    uab_snp[(size_t)i * n_index + idx] = src[i];
            }

            double logl_H1;
            double lam_mle = golden_section_lambda_mle_general(
                uab_snp, eigenvalues, n_samples,
                lambda_grid, hi_eval_grid, logdet_h_grid,
                log_l_min, step_val, n_grid, n_refine,
                mle_const, &table, &logl_H1,
                my_row0, my_pab
            );
            out_lambdas_mle[s] = lam_mle;

            double lrt_stat = 2.0 * (logl_H1 - logl_H0);
            if (lrt_stat < 0.0) lrt_stat = 0.0;
            out_p_lrts[s] = chi2_sf_c(lrt_stat);
        }

        Py_END_ALLOW_THREADS

        free(uab_snp_flat);
        free(lsg_pab_heap);
        free(lsg_row0_heap);
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        free_pab_table(&table);

        Py_DECREF(uab_inv_arr);
        Py_DECREF(uab_var_arr);
        Py_DECREF(eigenvalues_arr);
        return build_lrt_result_dict(&out);
    }

err_input_lrt_split_gen:
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * FUSED LRT — compute_lrt_fused_c
 *
 * LRT test accepting utg_t directly instead of pre-materialized
 * uab_varying_soa. Computes wx/xx/xy on-the-fly from utg_t columns
 * into per-thread scratch, then calls the existing golden section function.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_lrt_fused_c
 *
 * Args: utg_t (n_snps, n_samples), w (n_samples,), Uty (n_samples,),
 *       eigenvalues (n_samples,), uab_invariant_soa (3, n_samples),
 *       n_samples, l_min, l_max, n_grid, n_refine, logl_H0, n_threads
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_fused_c(PyObject *self, PyObject *args)
{
    PyObject *utg_t_obj, *w_obj, *Uty_obj, *eigenvalues_obj, *uab_inv_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTuple(args, "OOOOOiddiidi",
            &utg_t_obj, &w_obj, &Uty_obj, &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    PyArrayObject *utg_t_arr = NULL, *w_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;

    utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) goto err_input_lrt_fused;

    w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) goto err_input_lrt_fused;

    Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) goto err_input_lrt_fused;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input_lrt_fused;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input_lrt_fused;

    /* Validate shapes */
    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        goto err_input_lrt_fused;
    }
    if (PyArray_NDIM(w_arr) != 1 || PyArray_DIM(w_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "w must be shape (n_samples,)");
        goto err_input_lrt_fused;
    }
    if (PyArray_NDIM(Uty_arr) != 1 || PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Uty must be shape (n_samples,)");
        goto err_input_lrt_fused;
    }
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input_lrt_fused;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (3, %d)", n_samples);
        goto err_input_lrt_fused;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input_lrt_fused;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input_lrt_fused;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);
    const double *w_data = (const double *)PyArray_DATA(w_arr);
    const double *Uty_data = (const double *)PyArray_DATA(Uty_arr);
    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_inv_data = (const double *)PyArray_DATA(uab_inv_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0)
        goto err_input_lrt_fused;

    /* Invariant SoA pointers: rows [ww, wy, yy] */
    const double *inv_ww = uab_inv_data;
    const double *inv_wy = uab_inv_data + (size_t)n_samples;
    const double *inv_yy = uab_inv_data + (size_t)2 * n_samples;

    /* Allocate output arrays */
    lrt_output_t out;
    if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        goto err_input_lrt_fused;
    }

    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    /* MLE constant and grid */
    double n = (double)n_samples;
    double mle_const = 0.5 * n * (log(n) - log(2.0 * M_PI) - 1.0);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

    /* Build lambda grid */
    double *lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) {
        decref_lrt_output(&out);
        goto err_nomem_lrt_fused;
    }
    for (int g = 0; g < n_grid; g++)
        lambda_grid[g] = exp(log_l_min + g * step);

    /* Pre-compute hi_eval_grid, logdet_h_grid, and grid_inv (invariant sums) */
    double *hi_eval_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_samples * sizeof(double));
    double *logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid_invariant_t *grid_inv = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));
    if (!hi_eval_grid || !logdet_h_grid || !grid_inv) {
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        free(grid_inv);
        decref_lrt_output(&out);
        goto err_nomem_lrt_fused;
    }
    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double gs_ww = 0.0, gs_wy = 0.0, gs_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi[i] = h;
            logdet += log(v);
            gs_ww += h * inv_ww[i];
            gs_wy += h * inv_wy[i];
            gs_yy += h * inv_yy[i];
        }
        logdet_h_grid[g] = logdet;
        grid_inv[g].s_ww = gs_ww;
        grid_inv[g].s_wy = gs_wy;
        grid_inv[g].s_yy = gs_yy;
        grid_inv[g].log_s_ww = (gs_ww > 0.0) ? log(gs_ww) : 0.0;
    }

    /* Pre-allocate per-thread hi_eval buffers and scratch for vwx/vxx/vxy */
    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **thread_scratch =
        alloc_thread_scratch(actual_threads, (size_t)3 * n_samples);
    if (!thread_bufs || !thread_scratch) {
        free_thread_scratch(thread_bufs, actual_threads);
        free_thread_scratch(thread_scratch, actual_threads);
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        free(grid_inv);
        decref_lrt_output(&out);
        goto err_nomem_lrt_fused;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];
        double *scratch = thread_scratch[tid];
        double *vwx_local = scratch;
        double *vxx_local = scratch + n_samples;
        double *vxy_local = scratch + 2 * n_samples;

        const double *x = utg_t_data + (size_t)s * n_samples;

        /* Compute vwx/vxx/vxy on-the-fly from utg_t column */
        for (int i = 0; i < n_samples; i++) {
            vwx_local[i] = w_data[i] * x[i];
            vxx_local[i] = x[i] * x[i];
            vxy_local[i] = Uty_data[i] * x[i];
        }

        double logl_H1;
        double lam_mle = golden_section_lambda_mle_ncvt1_split(
            vwx_local, vxx_local, vxy_local, inv_ww, inv_wy, inv_yy,
            eigenvalues, n_samples,
            lambda_grid, hi_eval_grid, logdet_h_grid,
            grid_inv, log_l_min, step, n_grid, n_refine,
            mle_const, hi_eval_local, &logl_H1
        );
        out_lambdas_mle[s] = lam_mle;

        double lrt_stat = 2.0 * (logl_H1 - logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[s] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    free_thread_scratch(thread_bufs, actual_threads);
    free_thread_scratch(thread_scratch, actual_threads);

    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    free(grid_inv);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(w_arr);
    Py_DECREF(utg_t_arr);

    return build_lrt_result_dict(&out);

err_nomem_lrt_fused:
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(w_arr);
    Py_XDECREF(utg_t_arr);
    return PyErr_NoMemory();

err_input_lrt_fused:
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(w_arr);
    Py_XDECREF(utg_t_arr);
    return NULL;
}

/* =========================================================================
 * PERSISTENT LRT WORKSPACE — create_workspace_lrt_fused_c / compute_lrt_fused_ws_c
 *
 * Moves all SNP-invariant state into a PyCapsule workspace, eliminating
 * per-chunk malloc/free of lambda_grid, hi_eval_grid, logdet_h_grid,
 * and grid_inv.  Per-thread scratch buffers are allocated per-call in
 * compute_lrt_fused_ws_c for thread safety and adaptive retuning.
 * ========================================================================= */

typedef struct {
    int n_samples;
    int n_grid;
    int n_refine;
    double log_l_min, step, mle_const, logl_H0;
    /* Precomputed grid data (owned) */
    double *lambda_grid;      /* (n_grid,) */
    double *hi_eval_grid;     /* (n_grid * n_samples) */
    double *logdet_h_grid;    /* (n_grid,) */
    grid_invariant_t *grid_inv;  /* (n_grid,) */
    /* Raw data pointers into INCREF'd arrays (refs owned by workspace) */
    const double *eigenvalues;
    const double *inv_ww;
    const double *inv_wy;
    const double *inv_yy;
    const double *w_data;
    const double *Uty_data;
    PyObject *eigenvalues_ref;
    PyObject *uab_inv_ref;
    PyObject *w_ref;
    PyObject *Uty_ref;
} lmm_workspace_lrt_t;

static void lmm_workspace_lrt_destructor(PyObject *cap)
{
    lmm_workspace_lrt_t *ws =
        (lmm_workspace_lrt_t *)PyCapsule_GetPointer(cap, "lmm_workspace_lrt_fused");
    if (!ws) return;
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->grid_inv);
    Py_XDECREF(ws->eigenvalues_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->w_ref);
    Py_XDECREF(ws->Uty_ref);
    free(ws);
}

/* -------------------------------------------------------------------------
 * create_workspace_lrt_fused_c
 *
 * Python signature:
 *   create_workspace_lrt_fused_c(
 *       w,                # (n_samples,) float64
 *       Uty,              # (n_samples,) float64
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant_soa,# (3, n_samples) float64
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       logl_H0,          # float
 *       n_threads,        # int
 *   ) -> PyCapsule wrapping lmm_workspace_lrt_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_lrt_fused_c_py(
    PyObject *self, PyObject *args)
{
    PyObject *w_obj, *Uty_obj, *eigenvalues_obj, *uab_inv_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTuple(args, "OOOOiddiidi",
            &w_obj, &Uty_obj, &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    PyArrayObject *w_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    lmm_workspace_lrt_t *ws = NULL;

    w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) return NULL;

    Uty_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Uty_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!Uty_arr) goto err_lrt_ws_create;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_lrt_ws_create;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_lrt_ws_create;

    /* Validate shapes */
    if (PyArray_NDIM(w_arr) != 1 || PyArray_DIM(w_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "w must be shape (n_samples,)");
        goto err_lrt_ws_create;
    }
    if (PyArray_NDIM(Uty_arr) != 1 || PyArray_DIM(Uty_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "Uty must be shape (n_samples,)");
        goto err_lrt_ws_create;
    }
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_lrt_ws_create;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant_soa must be shape (3, %d)", n_samples);
        goto err_lrt_ws_create;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_lrt_ws_create;

    /* Allocate workspace */
    ws = (lmm_workspace_lrt_t *)calloc(1, sizeof(lmm_workspace_lrt_t));
    if (!ws) { PyErr_NoMemory(); goto err_lrt_ws_create; }

    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->logl_H0   = logl_H0;

    double n = (double)n_samples;
    ws->mle_const  = 0.5 * n * (log(n) - log(2.0 * M_PI) - 1.0);
    ws->log_l_min  = log(l_min);
    double log_l_max = log(l_max);
    ws->step       = (log_l_max - ws->log_l_min) / (double)(n_grid - 1);

    /* Borrow array pointers via Py_INCREF */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    Py_INCREF(w_arr);
    Py_INCREF(Uty_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;
    ws->w_ref           = (PyObject *)w_arr;
    ws->Uty_ref         = (PyObject *)Uty_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->w_data      = (const double *)PyArray_DATA(w_arr);
    ws->Uty_data    = (const double *)PyArray_DATA(Uty_arr);
    const double *uab_data = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_ww = uab_data;
    ws->inv_wy = uab_data + (size_t)n_samples;
    ws->inv_yy = uab_data + (size_t)2 * n_samples;

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_lrt_ws_alloc;
    }

    /* Build lambda grid + invariant dot products */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam = ws->lambda_grid[g];
        double *hi = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double gs_ww = 0.0, gs_wy = 0.0, gs_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi[i] = h;
            logdet += log(v);
            gs_ww += h * ws->inv_ww[i];
            gs_wy += h * ws->inv_wy[i];
            gs_yy += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;
        ws->grid_inv[g].s_ww = gs_ww;
        ws->grid_inv[g].s_wy = gs_wy;
        ws->grid_inv[g].s_yy = gs_yy;
        ws->grid_inv[g].log_s_ww = (gs_ww > 0.0) ? log(gs_ww) : 0.0;
    }

    /* n_threads is accepted for API symmetry but not stored — scratch buffers
     * are allocated per-call in compute_lrt_fused_ws_c to avoid thread-safety
     * issues and to allow adaptive thread retuning between chunks. */
    (void)n_threads;

    /* Release the OTF refs (workspace has its own Py_INCREF'd refs) */
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);

    PyObject *capsule = PyCapsule_New(ws, "lmm_workspace_lrt_fused",
                                      lmm_workspace_lrt_destructor);
    if (!capsule) goto err_lrt_ws_alloc;
    return capsule;

err_lrt_ws_alloc:
    /* Manual cleanup — ws was calloc'd so NULL fields are safe to free/skip */
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->grid_inv);
    /* Release INCREF'd refs (already INCREF'd before goto) */
    Py_XDECREF(ws->eigenvalues_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->w_ref);
    Py_XDECREF(ws->Uty_ref);
    free(ws);
    /* Fall through to release OTF array refs */
err_lrt_ws_create:
    Py_XDECREF(w_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lrt_fused_ws_c
 *
 * Python signature:
 *   compute_lrt_fused_ws_c(workspace, utg_t, n_threads)
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_fused_ws_c_py(PyObject *self, PyObject *args)
{
    PyObject *capsule_obj, *utg_t_obj;
    int n_threads;

    if (!PyArg_ParseTuple(args, "OOi", &capsule_obj, &utg_t_obj, &n_threads))
        return NULL;

    lmm_workspace_lrt_t *ws = (lmm_workspace_lrt_t *)
        PyCapsule_GetPointer(capsule_obj, "lmm_workspace_lrt_fused");
    if (!ws) return NULL;  /* PyCapsule_GetPointer sets ValueError on name mismatch */

    PyArrayObject *utg_t_arr = (PyArrayObject *)PyArray_FROM_OTF(
        utg_t_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!utg_t_arr) return NULL;

    int n_samples = ws->n_samples;

    if (PyArray_NDIM(utg_t_arr) != 2 ||
        PyArray_DIM(utg_t_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "utg_t must be shape (n_snps, %d)", n_samples);
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    npy_intp n_snps_raw = PyArray_DIM(utg_t_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        Py_DECREF(utg_t_arr);
        return NULL;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    lrt_output_t out;
    if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    /* Determine thread count — scratch is per-call so no workspace cap */
    int actual_threads = 1;
#ifdef _OPENMP
    {
        int max_t = (n_threads > 0) ? n_threads : omp_get_max_threads();
        actual_threads = (max_t < n_snps) ? max_t : n_snps;
        if (actual_threads < 1) actual_threads = 1;
    }
#else
    (void)n_threads;
#endif

    /* Allocate per-thread scratch buffers (thread-safe, adapts to retuned n_threads) */
    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **thread_scratch =
        alloc_thread_scratch(actual_threads, (size_t)3 * n_samples);
    if (!thread_bufs || !thread_scratch) {
        free_thread_scratch(thread_bufs, actual_threads);
        free_thread_scratch(thread_scratch, actual_threads);
        decref_lrt_output(&out);
        Py_DECREF(utg_t_arr);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];
        double *scratch = thread_scratch[tid];
        double *vwx_local = scratch;
        double *vxx_local = scratch + n_samples;
        double *vxy_local = scratch + 2 * n_samples;

        const double *x = utg_t_data + (size_t)s * n_samples;

        /* Compute vwx/vxx/vxy on-the-fly from utg_t column */
        for (int i = 0; i < n_samples; i++) {
            vwx_local[i] = ws->w_data[i] * x[i];
            vxx_local[i] = x[i] * x[i];
            vxy_local[i] = ws->Uty_data[i] * x[i];
        }

        double logl_H1;
        double lam_mle = golden_section_lambda_mle_ncvt1_split(
            vwx_local, vxx_local, vxy_local,
            ws->inv_ww, ws->inv_wy, ws->inv_yy,
            ws->eigenvalues, n_samples,
            ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->grid_inv, ws->log_l_min, ws->step,
            ws->n_grid, ws->n_refine,
            ws->mle_const, hi_eval_local, &logl_H1
        );
        out_lambdas_mle[s] = lam_mle;

        double lrt_stat = 2.0 * (logl_H1 - ws->logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[s] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    /* Free per-call scratch */
    free_thread_scratch(thread_bufs, actual_threads);
    free_thread_scratch(thread_scratch, actual_threads);

    Py_DECREF(utg_t_arr);
    return build_lrt_result_dict(&out);
}

/* -------------------------------------------------------------------------
 * _get_aligned_alloc_test_ptr
 *
 * Debug function: verify aligned_alloc returns 32-byte-aligned pointers.
 * Returns the pointer value as a Python int for assertion in tests.
 * ------------------------------------------------------------------------- */
static PyObject *_get_aligned_alloc_test_ptr(PyObject *self, PyObject *args)
{
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }
    double *p = alloc_aligned_doubles((size_t)n);
    if (!p) return PyErr_NoMemory();
    uintptr_t addr = (uintptr_t)p;
    free(p);
    return PyLong_FromUnsignedLongLong((unsigned long long)addr);
}

/* -------------------------------------------------------------------------
 * Module definition
 * ------------------------------------------------------------------------- */

#ifdef JAMMA_SENTINEL_UB
/* Phase 116.1 sanitizer sentinel: deliberately reads 1 byte past a 4-byte
 * heap allocation. Under -fsanitize=address this MUST abort with a
 * heap-buffer-overflow trace pointing at this source line. Without ASAN,
 * returns garbage from past the buffer end. Compile with
 * -DJAMMA_SENTINEL_UB to enable; the asan-sentinel-meta-test workflow
 * job sets that macro and asserts the workflow exits non-zero with the
 * expected ASAN frame. Do NOT enable in any other build path.
 */
static PyObject *jamma_sentinel_oob(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    char *buf = (char *)malloc(4);
    if (!buf) {
        PyErr_NoMemory();
        return NULL;
    }
    /* 1-byte heap OOB — ASAN must catch this. */
    char x = buf[5];
    free(buf);
    return PyLong_FromLong((long)x);
}
#endif

/* =========================================================================
 * MODULE REGISTRATION — methods[], PyModuleDef, PyInit__lmm_accel
 *
 * Every entry point in the file is named here.  That is the module defining
 * itself, not a family depending on another family, so scripts/
 * lmm_accel_sections.py excludes this block when it counts cross-section
 * coupling.  Without the banner the whole table reads as part of whichever
 * section precedes it, and 28 entry points look shared when none are.
 * ========================================================================= */

static PyMethodDef methods[] = {
    {
        "compute_lmm_batch_c",
        (PyCFunction)compute_lmm_batch_c,
        METH_VARARGS | METH_KEYWORDS,
        "Batch REML Wald pipeline for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Args:\n"
        "    eigenvalues: (n_samples,) float64 kinship eigenvalues\n"
        "    Uab_batch:   (n_snps, n_samples, 6) float64\n"
        "    Iab_batch:   (n_snps, 3, 6) float64 identity-weighted Pab\n"
        "    n_samples:   int\n"
        "    l_min:       float, minimum lambda\n"
        "    l_max:       float, maximum lambda\n"
        "    n_grid:      int, coarse grid points\n"
        "    n_refine:    int, golden section iterations (>= 1; caller typically passes >= 20)\n"
        "    n_threads:   int, OpenMP thread count\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds — each (n_snps,) float64\n"
    },
    {
        "compute_lmm_batch_split_c",
        (PyCFunction)compute_lmm_batch_split_c,
        METH_VARARGS | METH_KEYWORDS,
        "Split-Uab REML Wald pipeline for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Separates SNP-invariant Uab columns (ww, wy, yy) from varying\n"
        "columns (wx, xx, xy) to halve per-SNP DRAM traffic.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_varying:   (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    Iab_batch:     (n_snps, 3, 6) float64\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds\n"
    },
    {
        "create_workspace_split_c",
        (PyCFunction)create_workspace_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a persistent workspace for the split-Uab REML pipeline.\n"
        "\n"
        "Precomputes lambda_grid, hi_eval_grid, logdet_h_grid, grid_inv, and\n"
        "invariant Iab column sums once per run. The workspace is reused across\n"
        "all chunks — eliminating per-chunk C malloc and grid precomputation.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_t (opaque; pass to compute_lmm_chunk_split_c)\n"
    },
    {
        "compute_lmm_chunk_split_c",
        (PyCFunction)compute_lmm_chunk_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Per-chunk REML Wald using a pre-built workspace (no Iab_batch needed).\n"
        "\n"
        "Uses precomputed grids from create_workspace_split_c. logdet_iab is\n"
        "computed internally from raw Uab column sums. OpenMP schedule is static\n"
        "for uniform SNP cost. No per-chunk malloc.\n"
        "\n"
        "Args:\n"
        "    workspace:    PyCapsule from create_workspace_split_c\n"
        "    uab_varying:  (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    n_threads:    int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds\n"
    },
    {
        "create_workspace_general_c",
        (PyCFunction)create_workspace_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a persistent workspace for general n_cvt REML pipeline.\n"
        "\n"
        "Precomputes lambda_grid, hi_eval_grid, invariant column sums.\n"
        "Accepts recursion table arrays from build_pab_table_for_c().\n"
    },
    {
        "compute_lmm_chunk_general_c",
        (PyCFunction)compute_lmm_chunk_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Per-chunk REML Wald for general n_cvt using a pre-built workspace.\n"
        "\n"
        "OpenMP parallel over SNPs. Table-driven Pab recursion.\n"
    },
    {
        "compute_lrt_batch_c",
        (PyCFunction)compute_lrt_batch_c,
        METH_VARARGS,
        "Batch LRT for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Per-SNP MLE golden section optimization + chi2_sf p-value.\n"
        "\n"
        "Args:\n"
        "    eigenvalues: (n_samples,) float64\n"
        "    Uab_batch:   (n_snps, n_samples, 6) float64\n"
        "    n_samples:   int\n"
        "    l_min:       float\n"
        "    l_max:       float\n"
        "    n_grid:      int\n"
        "    n_refine:    int\n"
        "    logl_H0:     float, null model MLE log-likelihood\n"
        "    n_threads:   int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "compute_score_batch_c",
        (PyCFunction)compute_score_batch_c,
        METH_VARARGS,
        "Batch Score test for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Uses fixed null-model Hi_eval (no per-SNP optimization).\n"
        "\n"
        "Args:\n"
        "    eigenvalues:  (n_samples,) float64\n"
        "    Uab_batch:    (n_snps, n_samples, 6) float64\n"
        "    Hi_eval_null: (n_samples,) float64 — null-model weights\n"
        "    n_samples:    int\n"
        "    n_threads:    int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "compute_score_batch_general_c",
        (PyCFunction)compute_score_batch_general_c,
        METH_VARARGS,
        "Batch Score test for arbitrary n_cvt with optional OpenMP.\n"
        "\n"
        "Uses fixed null-model Hi_eval and table-driven Pab recursion.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    Uab_batch:     (n_snps, n_samples, n_index) float64\n"
        "    Hi_eval_null:  (n_samples,) float64 — null-model weights\n"
        "    n_samples:     int\n"
        "    n_cvt:         int\n"
        "    pab_table_dict: dict from build_pab_table_for_c(n_cvt)\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "compute_lrt_batch_general_c",
        (PyCFunction)compute_lrt_batch_general_c,
        METH_VARARGS,
        "Batch LRT for arbitrary n_cvt with optional OpenMP.\n"
        "\n"
        "Per-SNP MLE golden section + chi2_sf using table-driven Pab recursion.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    Uab_batch:     (n_snps, n_samples, n_index) float64\n"
        "    n_samples:     int\n"
        "    n_cvt:         int\n"
        "    pab_table_dict: dict from build_pab_table_for_c(n_cvt)\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    logl_H0:       float — null model MLE log-likelihood\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "create_workspace_mode4_split_c",
        (PyCFunction)create_workspace_mode4_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a mode-4 workspace for fused Wald/Score/LRT pipeline.\n"
        "\n"
        "Extends the standard split workspace with null-model Hi_eval,\n"
        "MLE constant, and null log-likelihood for LRT computation.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "    hi_eval_null:  (n_samples,) float64 — null-model Hi_eval\n"
        "    logl_H0:       float — null MLE log-likelihood\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_t (mode=4)\n"
    },
    {
        "compute_mode4_chunk_split_c",
        (PyCFunction)compute_mode4_chunk_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Fused per-chunk mode-4 compute: Wald/Score/LRT from SoA split data.\n"
        "\n"
        "Single OpenMP parallel loop produces all 8 output arrays.\n"
        "Requires a mode-4 workspace from create_workspace_mode4_split_c.\n"
        "\n"
        "Args:\n"
        "    workspace:    PyCapsule from create_workspace_mode4_split_c\n"
        "    uab_varying:  (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    n_threads:    int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds, p_scores,\n"
        "                    lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "compute_score_split_c",
        (PyCFunction)compute_score_split_c,
        METH_VARARGS,
        "SoA-native Score test for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Accepts split SoA data instead of full Uab batch.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_varying_soa:   (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    uab_invariant_soa: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    Hi_eval_null:      (n_samples,) float64 — null-model weights\n"
        "    n_samples:         int\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "compute_lrt_split_c",
        (PyCFunction)compute_lrt_split_c,
        METH_VARARGS,
        "SoA-native LRT for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Accepts split SoA data instead of full Uab batch.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_varying_soa:   (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    uab_invariant_soa: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:         int\n"
        "    l_min:             float\n"
        "    l_max:             float\n"
        "    n_grid:            int\n"
        "    n_refine:          int\n"
        "    logl_H0:           float — null model MLE log-likelihood\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "compute_score_split_general_c",
        (PyCFunction)compute_score_split_general_c,
        METH_VARARGS,
        "SoA-native Score test for general n_cvt with optional OpenMP.\n"
        "\n"
        "Accepts split SoA data + pab_table_dict instead of full Uab batch.\n"
        "Eliminates reconstruct_uab_from_soa for n_cvt>1 Score dispatch.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_varying_soa:   (n_snps, n_var, n_samples) float64\n"
        "    uab_invariant_soa: (n_inv, n_samples) float64\n"
        "    Hi_eval_null:      (n_samples,) float64 — null-model weights\n"
        "    n_samples:         int\n"
        "    n_cvt:             int\n"
        "    pab_table_dict:    dict — from build_pab_table_for_c(n_cvt)\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "compute_lrt_split_general_c",
        (PyCFunction)compute_lrt_split_general_c,
        METH_VARARGS,
        "SoA-native LRT for general n_cvt with optional OpenMP.\n"
        "\n"
        "Accepts split SoA data + pab_table_dict instead of full Uab batch.\n"
        "Eliminates reconstruct_uab_from_soa for n_cvt>1 LRT dispatch.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_varying_soa:   (n_snps, n_var, n_samples) float64\n"
        "    uab_invariant_soa: (n_inv, n_samples) float64\n"
        "    n_samples:         int\n"
        "    n_cvt:             int\n"
        "    pab_table_dict:    dict — from build_pab_table_for_c(n_cvt)\n"
        "    l_min:             float\n"
        "    l_max:             float\n"
        "    n_grid:            int\n"
        "    n_refine:          int\n"
        "    logl_H0:           float — null model MLE log-likelihood\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "create_workspace_fused_c",
        (PyCFunction)create_workspace_fused_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a fused workspace holding w/Uty for on-the-fly Uab computation.\n"
        "\n"
        "Eliminates the (n_snps, 3, n_samples) uab_varying_soa intermediate\n"
        "by computing wx/xx/xy from UtG_T columns in thread-local scratch.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    w:             (n_samples,) float64 — UtW[:,0]\n"
        "    Uty:           (n_samples,) float64 — rotated phenotype\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_t (fused)\n"
    },
    {
        "compute_lmm_chunk_fused_c",
        (PyCFunction)compute_lmm_chunk_fused_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Fused per-chunk REML Wald from UtG_T directly.\n"
        "\n"
        "Computes wx/xx/xy on-the-fly from UtG_T and w/Uty in workspace.\n"
        "Bitwise-identical to compute_lmm_chunk_split_c.\n"
        "\n"
        "Args:\n"
        "    workspace:  PyCapsule from create_workspace_fused_c\n"
        "    utg_t:      (n_snps, n_samples) float64 — UtG.T\n"
        "    n_threads:  int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds\n"
    },
    {
        "create_workspace_mode4_fused_c",
        (PyCFunction)create_workspace_mode4_fused_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a fused mode-4 workspace with w/Uty + null model.\n"
        "\n"
        "Extends fused workspace with Hi_eval_null, logl_H0 for Score/LRT.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    w:             (n_samples,) float64 — UtW[:,0]\n"
        "    Uty:           (n_samples,) float64 — rotated phenotype\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "    hi_eval_null:  (n_samples,) float64 — null-model Hi_eval\n"
        "    logl_H0:       float — null MLE log-likelihood\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_t (mode=4, fused)\n"
    },
    {
        "compute_mode4_chunk_fused_c",
        (PyCFunction)compute_mode4_chunk_fused_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Fused per-chunk mode-4 compute from UtG_T directly.\n"
        "\n"
        "Score + Wald + LRT with on-the-fly wx/xx/xy computation.\n"
        "Bitwise-identical to compute_mode4_chunk_split_c.\n"
        "\n"
        "Args:\n"
        "    workspace:  PyCapsule from create_workspace_mode4_fused_c\n"
        "    utg_t:      (n_snps, n_samples) float64 — UtG.T\n"
        "    n_threads:  int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds, p_scores,\n"
        "                    lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "create_workspace_fused_general_c",
        (PyCFunction)create_workspace_fused_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create fused general workspace for n_cvt >= 2 Wald computation.\n"
        "\n"
        "Stores UtW (transposed to column-major), Uty, and var_a/b_cols\n"
        "for on-the-fly varying Uab computation from UtG_T.\n"
    },
    {
        "compute_lmm_chunk_fused_general_c",
        (PyCFunction)compute_lmm_chunk_fused_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Compute Wald chunk from UtG_T using fused general workspace.\n"
        "\n"
        "Per-SNP varying dot products computed on-the-fly.\n"
        "Same results as compute_lmm_chunk_general_c.\n"
    },
    {
        "create_workspace_mode4_fused_general_c",
        (PyCFunction)create_workspace_mode4_fused_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create mode-4 fused general workspace for n_cvt >= 2.\n"
        "\n"
        "Extends fused general workspace with Hi_eval_null and logl_H0\n"
        "for Score/LRT computation.\n"
    },
    {
        "compute_mode4_chunk_fused_general_c",
        (PyCFunction)compute_mode4_chunk_fused_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Compute mode-4 chunk from UtG_T using fused general workspace.\n"
        "\n"
        "Score + Wald + LRT with on-the-fly varying dot products.\n"
    },
    {
        "compute_score_fused_c",
        (PyCFunction)compute_score_fused_c,
        METH_VARARGS,
        "Fused Score test from utg_t (no uab_varying_soa). n_cvt=1 only.\n"
        "\n"
        "Computes wx/xx/xy dot products on-the-fly from utg_t columns,\n"
        "eliminating the (n_snps, 3, n_samples) intermediate buffer.\n"
        "\n"
        "Args:\n"
        "    utg_t:             (n_snps, n_samples) float64 — UtG.T\n"
        "    w:                 (n_samples,) float64 — UtW[:,0]\n"
        "    Uty:               (n_samples,) float64 — rotated phenotype\n"
        "    Hi_eval_null:      (n_samples,) float64 — null-model weights\n"
        "    uab_invariant_soa: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    n_samples:         int\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "compute_lrt_fused_c",
        (PyCFunction)compute_lrt_fused_c,
        METH_VARARGS,
        "Fused LRT from utg_t (no uab_varying_soa). n_cvt=1 only.\n"
        "\n"
        "Computes wx/xx/xy on-the-fly into per-thread scratch,\n"
        "then calls golden_section_lambda_mle_ncvt1_split.\n"
        "\n"
        "Args:\n"
        "    utg_t:             (n_snps, n_samples) float64 — UtG.T\n"
        "    w:                 (n_samples,) float64 — UtW[:,0]\n"
        "    Uty:               (n_samples,) float64 — rotated phenotype\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_invariant_soa: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:         int\n"
        "    l_min:             float\n"
        "    l_max:             float\n"
        "    n_grid:            int\n"
        "    n_refine:          int\n"
        "    logl_H0:           float — null model MLE log-likelihood\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "_get_aligned_alloc_test_ptr",
        (PyCFunction)_get_aligned_alloc_test_ptr,
        METH_VARARGS,
        "Debug: return address of an aligned_alloc buffer for alignment testing."
    },
    {
        "create_workspace_score_fused_c",
        (PyCFunction)create_workspace_score_fused_c_py,
        METH_VARARGS,
        "Create a persistent Score workspace (PyCapsule).\n"
        "\n"
        "Precomputes h_null_w, h_null_Uty, null dot products, and\n"
        "F-distribution constants once per run.\n"
        "\n"
        "Args:\n"
        "    w:                 (n_samples,) float64 — UtW[:,0]\n"
        "    Uty:               (n_samples,) float64 — rotated phenotype\n"
        "    Hi_eval_null:      (n_samples,) float64 — null-model weights\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_invariant_soa: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:         int\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_score_t\n"
    },
    {
        "compute_score_fused_ws_c",
        (PyCFunction)compute_score_fused_ws_c_py,
        METH_VARARGS,
        "Compute Score test using a pre-built workspace.\n"
        "\n"
        "Args:\n"
        "    workspace: PyCapsule from create_workspace_score_fused_c\n"
        "    utg_t:     (n_snps, n_samples) float64 — UtG.T\n"
        "    n_threads: int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "create_workspace_lrt_fused_c",
        (PyCFunction)create_workspace_lrt_fused_c_py,
        METH_VARARGS,
        "Create a persistent LRT workspace (PyCapsule).\n"
        "\n"
        "Precomputes lambda_grid, hi_eval_grid, logdet_h_grid, grid_inv,\n"
        "and per-thread scratch buffers once per run.\n"
        "\n"
        "Args:\n"
        "    w:                 (n_samples,) float64\n"
        "    Uty:               (n_samples,) float64\n"
        "    eigenvalues:       (n_samples,) float64\n"
        "    uab_invariant_soa: (3, n_samples) float64\n"
        "    n_samples:         int\n"
        "    l_min:             float\n"
        "    l_max:             float\n"
        "    n_grid:            int\n"
        "    n_refine:          int\n"
        "    logl_H0:           float\n"
        "    n_threads:         int\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_lrt_t\n"
    },
    {
        "compute_lrt_fused_ws_c",
        (PyCFunction)compute_lrt_fused_ws_c_py,
        METH_VARARGS,
        "Compute LRT using a pre-built workspace.\n"
        "\n"
        "Args:\n"
        "    workspace: PyCapsule from create_workspace_lrt_fused_c\n"
        "    utg_t:     (n_snps, n_samples) float64 — UtG.T\n"
        "    n_threads: int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
#ifdef JAMMA_SENTINEL_UB
    {
        "jamma_sentinel_oob",
        (PyCFunction)jamma_sentinel_oob,
        METH_NOARGS,
        "Phase 116.1 sanitizer sentinel — deliberately reads past a heap "
        "allocation. Under ASAN this aborts with heap-buffer-overflow; "
        "without ASAN it returns garbage. Only compiled when "
        "-DJAMMA_SENTINEL_UB is set at build time."
    },
#endif
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_lmm_accel",
    "C extension: per-SNP REML/MLE pipelines (Wald, Score, LRT, fused mode-4) with OpenMP parallelism (n_cvt=1 + general n_cvt).",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__lmm_accel(void)
{
    import_array();  /* returns NULL on failure (NumPy Python 3 macro) */
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    /* ABI version — Python side checks this to detect stale .so files */
    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) { Py_DECREF(m); return NULL; }

    /* Expose whether this .so was compiled with OpenMP support */
#ifdef _OPENMP
    if (PyModule_AddIntConstant(m, "HAS_OPENMP", 1) < 0) { Py_DECREF(m); return NULL; }
#else
    if (PyModule_AddIntConstant(m, "HAS_OPENMP", 0) < 0) { Py_DECREF(m); return NULL; }
#endif

    return m;
}
