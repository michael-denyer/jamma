/*
 * _lmm_types.h — data shapes shared between the numerics and the boundary.
 *
 * The Pab table describes the recursion's index layout for a given n_cvt. The
 * marshalling layer in _lmm_support.c builds one from a Python dict; the
 * numerical kernels only read it. That makes it the one type both sides need,
 * and it is plain C data with no CPython in it, so it lives here rather than
 * in _lmm_support.h. A kernel that needs the layout should not have to include
 * <Python.h> to get it.
 */

#ifndef JAMMA_LMM_TYPES_H
#define JAMMA_LMM_TYPES_H

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
} pab_table_t;

#endif /* JAMMA_LMM_TYPES_H */
