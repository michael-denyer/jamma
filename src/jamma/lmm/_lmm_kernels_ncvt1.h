/*
 * _lmm_kernels_ncvt1.h — the single-covariate numerical kernels.
 *
 * The fixed 3x6 Pab recursion, the REML and MLE likelihood evaluations, and
 * the coarse grid and refiners that drive _lmm_lambda_search.h for the
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
#include "_lmm_lambda_search.h"

#include <math.h>

/* One SNP's refinement inputs: the SoA varying (var_*) and invariant (inv_*)
 * columns, each n_samples long and stride-1, and the constants every
 * likelihood evaluation reads. The lambda optimiser's context. */
typedef struct {
    const double *var_wx, *var_xx, *var_xy;
    const double *inv_ww, *inv_wy, *inv_yy;
    const double *eigenvalues;
    int n_samples;
    double logdet_iab, reml_const, mle_const;
} ncvt1_snp_t;

/* REML lambda from the caller's coarse-grid index best_idx (< 0 marks a fully
 * degenerate SNP). Writes the REML logl and the Wald statistics at the
 * optimum. */
double refine_lambda_ncvt1_split(
    const ncvt1_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    int df,
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

/* MLE lambda from the caller's coarse-grid index. Returns the optimal
 * lambda; writes the log-likelihood to *logl_out. */
double refine_lambda_mle_ncvt1_split(
    const ncvt1_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
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
