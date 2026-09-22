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
 * both live in this unit, so that inner call still inlines. Callers in
 * _lmm_accel_ncvt1.c invoke these once per SNP.
 *
 * Pure double arithmetic: no CPython, no NumPy, no OpenMP, no workspace state.
 * It needs only the shapes in _lmm_types.h.
 */

#ifndef JAMMA_LMM_KERNELS_NCVT1_H
#define JAMMA_LMM_KERNELS_NCVT1_H

#include "_lmm_types.h"

#include <math.h>

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
);

/* -------------------------------------------------------------------------
 * refine_lambda_mle_ncvt1_split
 *
 * Golden section refinement for MLE using a caller-selected coarse bracket.
 *
 * Returns optimal MLE lambda; writes log-likelihood to *logl_out.
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
    double log_l_min, double step,
    int n_grid, int n_refine,
    int best_idx,
    double mle_const,
    double *logl_out
);

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
