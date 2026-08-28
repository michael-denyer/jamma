/*
 * _lmm_kernels_ncvt1.h — the single-covariate numerical kernels.
 *
 * The fixed 3x6 Pab recursion, the REML and MLE likelihood evaluations, and
 * the coarse-grid / golden-section / refinement lambda optimizers for the
 * n_cvt = 1 path. Counterparts of the table-driven kernels in
 * _lmm_kernels_general; the two sets are disjoint under transitive closure,
 * which is what makes this a translation-unit boundary rather than a cut
 * chosen for tidiness.
 *
 * The optimizers call the likelihood evaluations roughly 70 times per SNP and
 * both live in this unit, so that inner call still inlines. Callers left in
 * _lmm_accel.c are the batch and fused entry points, which invoke these once
 * per SNP.
 *
 * Pure double arithmetic: no CPython, no NumPy, no OpenMP, no workspace state.
 * It needs only the shapes in _lmm_types.h.
 */

#ifndef JAMMA_LMM_KERNELS_NCVT1_H
#define JAMMA_LMM_KERNELS_NCVT1_H

#include "_lmm_types.h"

#include <math.h>



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
);

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
);

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
);

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
);






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
);

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
);


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
);

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
);


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
);

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
);

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
);

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
);


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
);


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
);

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
);

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
);


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
);

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
 * calc_pab_ncvt1_split
 *
 * Compute Pab from separated varying + invariant dot product sums.
 * The caller provides the 6 pre-accumulated sums (3 varying + 3 invariant).
 * ------------------------------------------------------------------------- */
void calc_pab_ncvt1_split(
    double s_ww, double s_wx, double s_wy,
    double s_xx, double s_xy, double s_yy,
    double pab[3][6]
);

#endif /* JAMMA_LMM_KERNELS_NCVT1_H */
