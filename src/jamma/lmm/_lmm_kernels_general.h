/*
 * _lmm_kernels_general.h — the arbitrary-n_cvt numerical kernels.
 *
 * The Pab recursion, likelihood evaluations, statistic extraction and lambda
 * optimizers for the general (arbitrary covariate count) path. These are the
 * table-driven counterparts of the single-covariate kernels, and the two sets
 * are disjoint: no general kernel calls an ncvt1 kernel or the reverse, which
 * is what makes this a translation-unit boundary rather than an arbitrary cut.
 *
 * Callers are the entry points in _lmm_accel_general.c. They keep the workspace
 * structs and all CPython marshalling; only the arithmetic moved.
 *
 * Pure double arithmetic over the Pab layout: no CPython, no NumPy, no
 * OpenMP, no workspace state. It needs only the table shape from
 * _lmm_types.h, so unlike _lmm_support.h it carries no import_array()
 * handling.
 */

#ifndef JAMMA_LMM_KERNELS_GENERAL_H
#define JAMMA_LMM_KERNELS_GENERAL_H

#include "_lmm_types.h"
#include "_lmm_lambda_search.h"

#include <math.h>

/* -------------------------------------------------------------------------
 * calc_pab_general — Table-driven Pab recursion for arbitrary n_cvt.
 *
 * Row 0 from row0 array (dot product sums), rows 1..n_rows-1 from entries.
 * Output in pab[n_rows * n_index], row-major.
 * ------------------------------------------------------------------------- */
void calc_pab_general(
    const double *row0,
    const pab_table_t *t,
    double *pab
);

/* One SNP's lambda-search inputs and the caller's scratch. uab_inv and
 * uab_var are the SoA invariant and varying Uab columns, each n_samples long.
 * row0 holds at least n_index doubles; pab and dpab hold at least
 * n_rows * n_index, and only the REML Newton polish uses dpab. The lambda
 * optimiser's context. */
typedef struct {
    const double *uab_inv;
    const double *uab_var;
    const double *eigenvalues;
    int n_samples;
    const pab_table_t *t;
    double logdet_iab, reml_const, mle_const;
    double *row0, *pab, *dpab;
} general_snp_t;

/* Coarse-grid index of the best REML (or MLE) logl, from the grid's cached
 * Hi_eval, logdet(H) and invariant sums; -1 when every point is degenerate. */
int coarse_grid_reml_general(
    const general_snp_t *snp,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const double *inv_sums_grid,    /* (n_grid, n_inv) */
    int n_grid
);

int coarse_grid_mle_general(
    const general_snp_t *snp,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const double *inv_sums_grid,    /* (n_grid, n_inv) */
    int n_grid
);

/* REML lambda from coarse-grid index best_idx (< 0 marks a fully degenerate
 * SNP). Writes the REML logl and the Wald statistics at the optimum. */
double refine_lambda_general(
    const general_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
);

/* MLE lambda from coarse-grid index best_idx. Returns the optimal lambda;
 * writes the log-likelihood to *logl_out. */
double refine_lambda_mle_general(
    const general_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    double *logl_out
);

/* -------------------------------------------------------------------------
 * logdet_from_row0 — compute logdet(Iab) from identity dot products.
 *
 * Encapsulates the identity Pab prepass: calls calc_pab_general into the
 * caller-provided scratch buffer, then extracts diagonal entries for logdet.
 *
 * row0:        n_index identity-weighted dot products
 * t:           pab_table_t with logdet_diag_rows/cols
 * pab_scratch: caller-provided buffer of at least n_rows * n_index doubles
 *
 * Returns logdet value, NAN if any diagonal <= 0 (logdet_diag_term).
 * ------------------------------------------------------------------------- */
static inline double logdet_from_row0(
    const double *row0,
    const pab_table_t *t,
    double *pab_scratch)
{
    calc_pab_general(row0, t, pab_scratch);

    int ni = t->n_index;
    int n_cvt = t->n_cvt;
    double logdet = 0.0;
    for (int d = 0; d < n_cvt + 1; d++) {
        double val = pab_scratch[t->logdet_diag_rows[d] * ni
                                 + t->logdet_diag_cols[d]];
        logdet += logdet_diag_term(val);
    }
    return logdet;
}

#endif /* JAMMA_LMM_KERNELS_GENERAL_H */
