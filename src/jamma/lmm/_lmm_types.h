/*
 * _lmm_types.h — data shapes shared between the numerics and the boundary.
 *
 * The Pab table describes the recursion's index layout for a given n_cvt. The
 * marshalling layer in _lmm_support.c builds one from a Python dict; the
 * numerical kernels only read it. That makes it the one type both sides need,
 * and it is plain C data with no CPython in it, so it lives here rather than
 * in _lmm_support.h. A kernel that needs the layout should not have to include
 * <Python.h> to get it.
 *
 * grid_invariant_t is here for the same reason: the workspace creators and
 * batch entry points in _lmm_accel.c fill it, the ncvt1 kernels read it, so it
 * spans the same boundary in the opposite direction.
 */

#ifndef JAMMA_LMM_TYPES_H
#define JAMMA_LMM_TYPES_H

/* REML_SENTINEL expands to -INFINITY. Included here rather than left to the
 * includer so the macro cannot compile in one unit and fail in the next. */
#include <math.h>

/* Table-driven Pab bounds. MAX_N_CVT=100 -> MAX_N_INDEX=5253 (~42KB per
 * array); functions holding two such arrays peak at ~84KB, well inside an
 * OpenMP thread stack (2-4MB). */
#define MAX_N_CVT    100
#define MAX_N_INDEX  ((MAX_N_CVT + 3) * (MAX_N_CVT + 2) / 2)  /* 5253 */
#define MAX_N_ROWS   (MAX_N_CVT + 2)                          /* 102 */
#define MAX_PAB_SIZE (MAX_N_ROWS * MAX_N_INDEX)               /* 535806 */

/* Floor for P_yy before the log in a REML/MLE tail. Mirrors _P_YY_MIN in
 * likelihood.py; both sides must agree or the C and NumPy paths diverge on
 * near-degenerate SNPs. Shared here because the likelihood kernels and the
 * test statistics both clamp against it. */
#define P_YY_MIN 1e-8

/* REML sentinel: replaces NaN log-likelihood from degenerate P_yy.
 * reml_finish returns NaN when P_yy < 0; the golden section callers
 * map NaN -> REML_SENTINEL so the > comparison skips degenerate points
 * without needing an isnan() guard on every iteration.
 * Matches the Python path's np.where(isnan, -inf, logl).
 * Here rather than in _lmm_support.h for the same reason as P_YY_MIN: the
 * lambda optimizers read it and must not need <Python.h> to do so. */
#define REML_SENTINEL (-INFINITY)

/* Pre-computed invariant dot products for one coarse grid point.
 * Memory: n_grid * sizeof(grid_invariant_t) ~ 50 * 32 = 1.6 KB (fits L1). */
typedef struct {
    double s_ww;       /* sum of hi * ww */
    double s_wy;       /* sum of hi * wy */
    double s_yy;       /* sum of hi * yy */
    double log_s_ww;   /* log(s_ww) if > 0, else 0 */
} grid_invariant_t;

typedef struct {
    int index_ab, index_aw, index_bw, index_ww;
} pab_entry_t;

typedef struct {
    int n_cvt, n_index, n_rows, n_inv, n_var;
    int idx_xx, idx_xy, idx_yy;
    int df;  /* n_samples - n_cvt - 1 */
    int *invariant_indices;  /* (n_inv,) */
    int *varying_indices;    /* (n_var,) */
    int *logdet_diag_rows;   /* (n_cvt+1,) */
    int *logdet_diag_cols;   /* (n_cvt+1,) */
    int *level_offsets;      /* (n_rows,) — offset into entries per level */
    int *level_counts;       /* (n_rows,) — count per level */
    pab_entry_t *entries;    /* all entries concatenated */
    int n_entries;
    /* The two fused-vector columns whose product is each varying Uab column,
     * 0..n_cvt-1 for UtW columns, n_cvt for x, n_cvt+1 for Uty. (n_var,) */
    int *var_a_cols;
    int *var_b_cols;
} pab_table_t;

#endif /* JAMMA_LMM_TYPES_H */
