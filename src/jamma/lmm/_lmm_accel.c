/*
 * _lmm_accel.c — C extension implementing per-SNP REML/MLE pipelines
 * for Wald, Score, and LRT tests (n_cvt=1 and general n_cvt).
 *
 * Exported functions, one per DispatchPath the runner can select:
 *   FUSED          create_workspace_fused_c, compute_lmm_chunk_fused_c,
 *                  create_workspace_mode4_fused_c, compute_mode4_chunk_fused_c
 *   FUSED_GENERAL  create_workspace_fused_general_c,
 *                  compute_lmm_chunk_fused_general_c,
 *                  create_workspace_mode4_fused_general_c,
 *                  compute_mode4_chunk_fused_general_c
 *   FUSED_SCORE_WS create_workspace_score_fused_c, compute_score_fused_ws_c
 *   FUSED_LRT_WS   create_workspace_lrt_fused_c, compute_lrt_fused_ws_c
 *   SOA_SPLIT      compute_score_split_general_c, compute_lrt_split_general_c
 *
 * -DJAMMA_SENTINEL_UB enables a heap-OOB sentinel function
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

/* _lmm_support.h must stay first: it is what pulls in <Python.h>, and CPython
 * requires that before any standard header. The concrete failure here is M_PI,
 * which is not C11 — glibc's <math.h> only defines it under __USE_XOPEN, set by
 * the _XOPEN_SOURCE that Python.h defines. Let another header reach <math.h>
 * first and the include guard blocks the later expansion, so every M_PI below
 * fails to compile on Linux while macOS, whose libc defines it
 * unconditionally, builds clean. */
#include "_lmm_support.h"

#include "_lmm_stats.h"
#include "_lmm_tests.h"
#include "_lmm_kernels_ncvt1.h"
#include "_lmm_kernels_general.h"
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

/* ABI version: bump when function signatures or array layout expectations change.
 * The Python side checks this at import time to detect stale .so files. */
#define ABI_VERSION 12  /* v12: Unreachable batch/split entry points removed */

/* P_YY_MIN and REML_SENTINEL moved to _lmm_types.h when the general kernels
 * left this file: both units read them, and two copies could drift. */


/* =========================================================================
 * Shared helpers — eliminate duplication across full/split paths
 * ========================================================================= */


/* REML log-likelihood tail: logdet_pab + P_yy guard + REML formula.
 * Shared by reml_logl_ncvt1, reml_logl_ncvt1_cached, reml_logl_ncvt1_split. */


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

/* grid_invariant_t moved to _lmm_types.h when the ncvt1 kernels left this
 * file: the workspace creators and batch entry points here build it, the
 * kernels there read it, so it spans the boundary. */

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

/* Owner of every allocation and array ref in the struct. NULL-safe on
 * every field, so it serves both the capsule destructor and each creator's
 * error path. */
static void lmm_workspace_free(lmm_workspace_t *ws)
{
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

static void lmm_workspace_destructor(PyObject *cap)
{
    lmm_workspace_free(
        (lmm_workspace_t *)PyCapsule_GetPointer(cap, "lmm_workspace"));
}


/* =========================================================================
 * GENERAL n_cvt support — table-driven Pab recursion for arbitrary covariates
 *
 * Adds the lmm_workspace_general_t workspace type, which accepts n_cvt as a
 * parameter. The n_cvt=1 code path is separate and unchanged.
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
    /* Fused Uab fields. Every lmm_workspace_general_t is fused now (the
     * non-fused general workspace was deleted); the NULL checks on these
     * fields elsewhere in this file are defensive, not a real code path. */
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
static void lmm_workspace_general_free(lmm_workspace_general_t *ws)
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

static void lmm_workspace_general_destructor(PyObject *cap)
{
    lmm_workspace_general_free((lmm_workspace_general_t *)
        PyCapsule_GetPointer(cap, "lmm_workspace_general"));
}


/* =========================================================================
 * FUSED Uab — workspace holds w/Uty, chunk accepts UtG_T directly
 *
 * Eliminates the (n_snps, 3, n_samples) uab_varying_soa intermediate
 * allocation by computing wx/xx/xy products on-the-fly from UtG_T columns
 * in thread-local scratch buffers. Same FP operations in the same order
 * as the *_ncvt1_split helpers (golden_section_optimize_lambda_split_ncvt1_numpy)
 * — results are bitwise-identical.
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * create_workspace_fused_c
 *
 * Allocates the per-run Wald workspace, holding the invariant Uab block plus
 * the two vectors the chunk kernel forms the varying columns from:
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
/* Fill a calloc'd n_cvt=1 workspace from validated inputs: the scalar
 * constants, the borrowed array pointers (INCREF'd here, released by
 * lmm_workspace_free), the invariant Iab scalar and the lambda grid.
 * 0, or -1 with PyErr set. */
static int init_ncvt1_workspace(
    lmm_workspace_t *ws,
    PyArrayObject *eigenvalues_arr, PyArrayObject *uab_inv_arr,
    PyArrayObject *w_arr, PyArrayObject *Uty_arr,
    int n_samples, double l_min, double l_max, int n_grid, int n_refine)
{
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

    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    Py_INCREF(w_arr);
    Py_INCREF(Uty_arr);
    ws->w = (const double *)PyArray_DATA(w_arr);
    ws->Uty = (const double *)PyArray_DATA(Uty_arr);
    ws->w_ref = (PyObject *)w_arr;
    ws->Uty_ref = (PyObject *)Uty_arr;

    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));
    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        return -1;
    }

    build_grid_ncvt1(n_grid, n_samples, ws->log_l_min, ws->step,
                     ws->eigenvalues, ws->inv_ww, ws->inv_wy, ws->inv_yy,
                     ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
                     ws->grid_inv);
    return 0;
}

/* Mode-4 extras on an initialised n_cvt=1 workspace: the owned copy of the
 * null-model Hi_eval, its invariant dot products, and the MLE constant.
 * 0, or -1 with PyErr set. */
static int init_ncvt1_null_model(
    lmm_workspace_t *ws, const double *hi_eval_null, double logl_H0)
{
    int n_samples = ws->n_samples;
    ws->mode = 4;
    ws->logl_H0 = logl_H0;
    ws->mle_const = 0.5 * (double)n_samples
                    * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

    ws->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->hi_eval_null) {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(ws->hi_eval_null, hi_eval_null, (size_t)n_samples * sizeof(double));

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
    return 0;
}

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

    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_input;
    uab_inv_arr = take_matrix(uab_inv_obj, 3, n_samples, "uab_invariant");
    if (!uab_inv_arr) goto err_input;
    w_arr = take_vector(w_obj, n_samples, "w");
    if (!w_arr) goto err_input;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_input;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    if (init_ncvt1_workspace(ws, eigenvalues_arr, uab_inv_arr, w_arr, Uty_arr,
                             n_samples, l_min, l_max, n_grid, n_refine) < 0)
        goto err_ws;

    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    return capsule;

err_ws:
    lmm_workspace_free(ws);
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
 * computes wx/xx/xy on-the-fly from the w/Uty stored in the workspace, rather
 * than taking them prebuilt. The arithmetic and its order are unchanged by
 * that, so results do not depend on which form the caller supplies.
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

    int n_samples = ws->n_samples;
    int n_snps;
    utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;

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

        /* From here, the per-SNP body works on wx/xx/xy like any other */
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
    if (validate_logl_H0(logl_H0) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *w_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_input;
    uab_inv_arr = take_matrix(uab_inv_obj, 3, n_samples, "uab_invariant");
    if (!uab_inv_arr) goto err_input;
    w_arr = take_vector(w_obj, n_samples, "w");
    if (!w_arr) goto err_input;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_input;
    hi_eval_null_arr = take_vector(hi_eval_null_obj, n_samples, "hi_eval_null");
    if (!hi_eval_null_arr) goto err_input;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;
    if (validate_hi_eval_null(
            (const double *)PyArray_DATA(hi_eval_null_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    if (init_ncvt1_workspace(ws, eigenvalues_arr, uab_inv_arr, w_arr, Uty_arr,
                             n_samples, l_min, l_max, n_grid, n_refine) < 0)
        goto err_ws;
    if (init_ncvt1_null_model(
            ws, (const double *)PyArray_DATA(hi_eval_null_arr), logl_H0) < 0)
        goto err_ws;

    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(hi_eval_null_arr);
    return capsule;

err_ws:
    lmm_workspace_free(ws);
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
 * Fused per-chunk mode-4 compute: Score + Wald + LRT from UtG_T directly,
 * computing wx/xx/xy on-the-fly from the w/Uty stored in the workspace.
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

    int n_samples = ws->n_samples;
    int n_snps;
    utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;

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
 * (the SNP genotype vector), and Uty, then feeds them into the table-driven
 * Pab recursion. Forming the varying columns here rather than taking them
 * prebuilt does not change the arithmetic or its order.
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
 * create_workspace_fused_general_c
 *
 * Allocates the per-run general-n_cvt Wald workspace. Beyond the invariant
 * Uab block it stores UtW (transposed to column-major), Uty, and
 * var_a_cols/var_b_cols, which the chunk kernel needs to form the varying dot
 * products on-the-fly. Allocates per-thread scratch buffers.
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
    if (validate_n_cvt(n_cvt) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *UtW_arr = NULL, *Uty_arr = NULL;
    lmm_workspace_general_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_input;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;
    /* uab_invariant has n_inv rows, known only once the index table is
     * parsed; init_fused_general_workspace checks its shape. */
    uab_inv_arr = take_array(uab_inv_obj);
    if (!uab_inv_arr) goto err_input;
    UtW_arr = take_matrix(UtW_obj, n_samples, n_cvt, "UtW");
    if (!UtW_arr) goto err_input;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_input;

    ws = (lmm_workspace_general_t *)calloc(1, sizeof(lmm_workspace_general_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    if (init_fused_general_workspace(
            ws, eigenvalues_arr, uab_inv_arr, UtW_arr, Uty_arr,
            inv_idx_obj, var_idx_obj,
            diag_rows_obj, diag_cols_obj,
            offsets_obj, counts_obj, entries_obj,
            var_a_obj, var_b_obj,
            n_samples, l_min, l_max,
            n_grid, n_refine, n_threads, n_cvt,
            idx_xx, idx_xy, idx_yy) < 0)
        goto err_ws;

    capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(UtW_arr);
    Py_DECREF(Uty_arr);
    return capsule;

err_ws:
    lmm_workspace_general_free(ws);
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(UtW_arr);
    Py_XDECREF(Uty_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_fused_general_c
 *
 * Per-chunk Wald compute using fused general workspace. Computes n_var
 * varying dot products on-the-fly from UtW/Uty/UtG_T per SNP, then feeds
 * them into the table-driven Pab recursion and golden section.
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

    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_inv = ws->table.n_inv;
    int n_snps;
    utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;
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

        /* Compute per-SNP logdet_iab at identity.
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
    if (validate_n_cvt(n_cvt) < 0)
        return NULL;
    if (validate_logl_H0(logl_H0) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *UtW_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_general_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_input;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;
    uab_inv_arr = take_array(uab_inv_obj);
    if (!uab_inv_arr) goto err_input;
    UtW_arr = take_matrix(UtW_obj, n_samples, n_cvt, "UtW");
    if (!UtW_arr) goto err_input;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_input;
    hi_eval_null_arr = take_vector(hi_eval_null_obj, n_samples, "hi_eval_null");
    if (!hi_eval_null_arr) goto err_input;
    if (validate_hi_eval_null(
            (const double *)PyArray_DATA(hi_eval_null_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_general_t *)calloc(1, sizeof(lmm_workspace_general_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    if (init_fused_general_workspace(
            ws, eigenvalues_arr, uab_inv_arr, UtW_arr, Uty_arr,
            inv_idx_obj, var_idx_obj,
            diag_rows_obj, diag_cols_obj,
            offsets_obj, counts_obj, entries_obj,
            var_a_obj, var_b_obj,
            n_samples, l_min, l_max,
            n_grid, n_refine, n_threads, n_cvt,
            idx_xx, idx_xy, idx_yy) < 0)
        goto err_ws;

    /* Mode-4 specific fields */
    ws->mode = 4;
    ws->logl_H0 = logl_H0;
    ws->mle_const = 0.5 * (double)n_samples
                    * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

    ws->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->hi_eval_null) { PyErr_NoMemory(); goto err_ws; }
    memcpy(ws->hi_eval_null,
           (const double *)PyArray_DATA(hi_eval_null_arr),
           (size_t)n_samples * sizeof(double));

    /* Precompute null-model invariant sums */
    int n_inv = ws->table.n_inv;
    ws->null_inv_sums = (double *)malloc((size_t)n_inv * sizeof(double));
    if (!ws->null_inv_sums) { PyErr_NoMemory(); goto err_ws; }
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
        if (!ws->uab_snp_flat) { PyErr_NoMemory(); goto err_ws; }
    }

    capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(UtW_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(hi_eval_null_arr);
    return capsule;

err_ws:
    lmm_workspace_general_free(ws);
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(UtW_arr);
    Py_XDECREF(Uty_arr);
    Py_XDECREF(hi_eval_null_arr);
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

    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_inv = ws->table.n_inv;
    int n_snps;
    utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;
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

static void lmm_workspace_score_free(lmm_workspace_score_t *ws)
{
    if (!ws) return;
    free(ws->h_null_w);
    free(ws->h_null_Uty);
    Py_XDECREF(ws->hi_eval_null_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->eigenvalues_ref);
    free(ws);
}

static void lmm_workspace_score_destructor(PyObject *cap)
{
    lmm_workspace_score_free((lmm_workspace_score_t *)
        PyCapsule_GetPointer(cap, "lmm_workspace_score_fused"));
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
    PyObject *capsule = NULL;

    w_arr = take_vector(w_obj, n_samples, "w");
    if (!w_arr) goto err_input;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_input;
    hi_eval_null_arr = take_vector(hi_eval_null_obj, n_samples, "Hi_eval_null");
    if (!hi_eval_null_arr) goto err_input;
    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_input;
    uab_inv_arr = take_matrix(uab_inv_obj, 3, n_samples, "uab_invariant_soa");
    if (!uab_inv_arr) goto err_input;
    if (validate_hi_eval_null(
            (const double *)PyArray_DATA(hi_eval_null_arr), n_samples) < 0)
        goto err_input;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_score_t *)calloc(1, sizeof(lmm_workspace_score_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

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
        goto err_ws;
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

    capsule = PyCapsule_New(ws, "lmm_workspace_score_fused",
                            lmm_workspace_score_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(hi_eval_null_arr);
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return capsule;

err_ws:
    lmm_workspace_score_free(ws);
err_input:
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

    int n_samples = ws->n_samples;
    int n_snps;
    PyArrayObject *utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;
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
    if (validate_n_cvt(n_cvt) < 0)
        return NULL;
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

        if (validate_hi_eval_null(hi_eval_null, n_samples) < 0) {
            free_pab_table(&table);
            goto err_input_score_split_gen;
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
    if (validate_n_cvt(n_cvt) < 0)
        return NULL;
    if (validate_logl_H0(logl_H0) < 0)
        return NULL;
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

static void lmm_workspace_lrt_free(lmm_workspace_lrt_t *ws)
{
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

static void lmm_workspace_lrt_destructor(PyObject *cap)
{
    lmm_workspace_lrt_free((lmm_workspace_lrt_t *)
        PyCapsule_GetPointer(cap, "lmm_workspace_lrt_fused"));
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
    if (validate_logl_H0(logl_H0) < 0)
        return NULL;

    PyArrayObject *w_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    lmm_workspace_lrt_t *ws = NULL;
    PyObject *capsule = NULL;

    w_arr = take_vector(w_obj, n_samples, "w");
    if (!w_arr) goto err_input;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_input;
    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_input;
    uab_inv_arr = take_matrix(uab_inv_obj, 3, n_samples, "uab_invariant_soa");
    if (!uab_inv_arr) goto err_input;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_lrt_t *)calloc(1, sizeof(lmm_workspace_lrt_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

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
    build_grid_ncvt1(n_grid, n_samples, ws->log_l_min, ws->step,
                     ws->eigenvalues, ws->inv_ww, ws->inv_wy, ws->inv_yy,
                     ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
                     ws->grid_inv);

    /* n_threads is accepted for API symmetry but not stored — scratch buffers
     * are allocated per-call in compute_lrt_fused_ws_c to avoid thread-safety
     * issues and to allow adaptive thread retuning between chunks. */
    (void)n_threads;

    capsule = PyCapsule_New(ws, "lmm_workspace_lrt_fused",
                            lmm_workspace_lrt_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return capsule;

err_ws:
    lmm_workspace_lrt_free(ws);
err_input:
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

    int n_samples = ws->n_samples;
    int n_snps;
    PyArrayObject *utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;
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
/* Sanitizer sentinel: deliberately reads 1 byte past a 4-byte
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
 * section precedes it, and 16 entry points look shared when none are.
 * ========================================================================= */

static PyMethodDef methods[] = {
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
        "Forms the varying Uab columns from w/Uty rather than taking them\n"
        "prebuilt; the arithmetic and its order are unchanged.\n"
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
        "Forms the varying Uab columns from w/Uty rather than taking them\n"
        "prebuilt; the arithmetic and its order are unchanged.\n"
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
        "Forms the varying Uab columns from UtW/Uty rather than taking them\n"
        "prebuilt; the arithmetic and its order are unchanged.\n"
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
        "Sanitizer sentinel — deliberately reads past a heap "
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
