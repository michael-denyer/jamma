/* General-covariate workspace ownership and Python compute entry points. */

#define NO_IMPORT_ARRAY
#include "_lmm_accel_internal.h"

#include "_lmm_kernels_general.h"
#include "_lmm_stats.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
/* Coarse-grid block: every general workspace allocates one regardless of
 * mode (unlike the ncvt1 grid, which mode 3 skips), so this is never NULL. */
typedef struct {
    double *lambda_grid;    /* (n_grid,) */
    double log_l_min, step; /* bracket endpoints as computed at creation */
    double *hi_eval_grid;   /* (n_grid * n_samples) */
    double *logdet_h_grid;  /* (n_grid,) */
    double *inv_sums_grid;  /* (n_grid * n_inv) — precomputed invariant dot products */
    int n_grid, n_refine;
} general_grid_t;

/* Null-model block: modes 3 and 4 carry hi_eval_null and its invariant sums.
 * NULL unless the mode wants it. */
typedef struct {
    double *hi_eval_null;    /* (n_samples,) owned */
    double *null_inv_sums;   /* (n_inv,) precomputed null-model invariant sums, owned */
} general_null_model_t;

/* LRT block: modes 2 and 4 carry logl_H0, mle_const and the per-thread LRT
 * buffer. NULL unless the mode wants it. */
typedef struct {
    double logl_H0;
    double mle_const;
    /* Pre-allocated per-thread LRT buffer.
     * (actual_threads * n_index * n_samples) doubles, row-major per SNP.
     * Avoids per-SNP malloc inside OpenMP loop. */
    double *uab_snp_flat;
} general_lrt_t;

typedef struct {
    /* Fixed params */
    double *eigenvalues;    /* (n_samples,) — owned copy */
    double reml_const;
    int n_samples;
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
    double *scratch_flat;       /* (actual_threads * n_var * n_samples) owned */
    int actual_threads;         /* for scratch deallocation sizing */
    /* Per-thread heap buffers for Pab recursion (replaces stack arrays) */
    double *pab_per_thread;     /* (actual_threads * pab_size) owned */
    double *row0_per_thread;    /* (actual_threads * n_index) owned */
    int pab_size;               /* n_rows * n_index for this workspace */
    PyObject *Uty_ref;          /* keeps Uty array alive */
    int mode;                   /* 1 Wald, 2 LRT, 3 Score, 4 all three */
    /* Sub-blocks: grid is always present; null_model and lrt are NULL when
     * the owning mode does not use them, so ws->lrt == NULL is the contract
     * rather than a comment. */
    general_grid_t *grid;
    general_null_model_t *null_model;
    general_lrt_t *lrt;
} lmm_workspace_general_t;

/* PyCapsule destructor for general workspace */
static void lmm_workspace_general_free(lmm_workspace_general_t *ws)
{
    if (!ws) return;
    if (ws->grid) {
        free(ws->grid->lambda_grid);
        free(ws->grid->hi_eval_grid);
        free(ws->grid->logdet_h_grid);
        free(ws->grid->inv_sums_grid);
        free(ws->grid);
    }
    free(ws->eigenvalues);
    free(ws->inv_identity_sums);
    free_pab_table(&ws->table);
    Py_XDECREF(ws->uab_inv_ref);
    /* Fused general fields */
    free(ws->utw_transposed);
    free(ws->scratch_flat);
    free(ws->pab_per_thread);
    free(ws->row0_per_thread);
    Py_XDECREF(ws->Uty_ref);
    if (ws->null_model) {
        free(ws->null_model->hi_eval_null);
        free(ws->null_model->null_inv_sums);
        free(ws->null_model);
    }
    if (ws->lrt) {
        free(ws->lrt->uab_snp_flat);
        free(ws->lrt);
    }
    free(ws);
}

static void lmm_workspace_general_destructor(PyObject *cap)
{
    lmm_workspace_general_free((lmm_workspace_general_t *)
        PyCapsule_GetPointer(cap, "lmm_workspace_general"));
}

/* The workspace behind a capsule, or NULL with PyErr set on a type mismatch.
 * One compute entry point serves every general workspace regardless of the
 * lmm_mode it was created for, so there is no mode guard here: the compute
 * itself reads ws->mode to decide which blocks to run. */
static lmm_workspace_general_t *general_workspace_any_mode(
    PyObject *cap, const char *fn)
{
    (void)fn;
    return (lmm_workspace_general_t *)
        PyCapsule_GetPointer(cap, "lmm_workspace_general");
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

/* Fill a calloc'd general workspace whose table has already been parsed:
 * eigenvalues, uab_inv, UtW (transposed), Uty, per-thread scratch, the
 * lambda grid and its invariant sums, and the beta/REML constants. 0, or -1
 * with PyErr set; the caller frees ws through lmm_workspace_general_free. */
static int init_fused_general_workspace(
    lmm_workspace_general_t *ws,
    PyArrayObject *eigenvalues_arr,
    PyArrayObject *uab_inv_arr,
    PyArrayObject *UtW_arr,
    PyArrayObject *Uty_arr,
    int n_samples, double l_min, double l_max,
    int n_grid, int n_refine, int n_threads)
{
    int n_cvt   = ws->table.n_cvt;
    int n_index = ws->table.n_index;
    int n_rows  = ws->table.n_rows;
    int n_inv   = ws->table.n_inv;
    int n_var   = ws->table.n_var;

    ws->n_samples = n_samples;
    ws->n_cvt = n_cvt;

    /* Copy eigenvalues (owned) */
    ws->eigenvalues = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!ws->eigenvalues) { PyErr_NoMemory(); return -1; }
    memcpy(ws->eigenvalues, PyArray_DATA(eigenvalues_arr),
           (size_t)n_samples * sizeof(double));

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

    general_grid_t *grid = (general_grid_t *)calloc(1, sizeof(general_grid_t));
    if (!grid) { PyErr_NoMemory(); return -1; }
    grid->n_grid = n_grid;
    grid->n_refine = n_refine;
    grid->log_l_min = log_l_min;
    grid->step = step;

    grid->lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid->hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    grid->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid->inv_sums_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_inv * sizeof(double));

    if (!grid->lambda_grid || !grid->hi_eval_grid ||
        !grid->logdet_h_grid || !grid->inv_sums_grid) {
        free(grid->lambda_grid);
        free(grid->hi_eval_grid);
        free(grid->logdet_h_grid);
        free(grid->inv_sums_grid);
        free(grid);
        PyErr_NoMemory();
        return -1;
    }

    for (int g = 0; g < n_grid; g++)
        grid->lambda_grid[g] = exp(log_l_min + g * step);

    /* Precompute hi_eval_grid, logdet_h_grid, and invariant sums */
    for (int g = 0; g < n_grid; g++) {
        double lam = grid->lambda_grid[g];
        double *hi_row = grid->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
        }
        grid->logdet_h_grid[g] = logdet;

        double *inv_sums = grid->inv_sums_grid + (size_t)g * n_inv;
        for (int c = 0; c < n_inv; c++) {
            double s = 0.0;
            const double *col = ws->uab_inv + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += hi_row[i] * col[i];
            inv_sums[c] = s;
        }
    }
    ws->grid = grid;

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
 * create_workspace_general_c
 *
 * Python signature:
 *   create_workspace_general_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (n_inv, n_samples) float64 — SoA
 *       UtW,              # (n_samples, n_cvt) float64 — row-major
 *       Uty,              # (n_samples,) float64
 *       n_samples, l_min, l_max, n_grid, n_refine, n_threads,
 *       pab_table,        # dict, PabCTable._asdict()
 *       *, lmm_mode, hi_eval_null=None, logl_H0=None,
 *   ) -> PyCapsule
 *
 * n_cvt and every index array come from pab_table. lmm_mode is 1 (Wald),
 * 2 (LRT), 3 (Score) or 4 (all three); hi_eval_null is required by 3 and 4,
 * logl_H0 by 2 and 4, matching create_workspace_ncvt1_c's mode contract.
 * ------------------------------------------------------------------------- */
PyObject *create_workspace_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "UtW", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "pab_table", "lmm_mode", "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *UtW_obj, *Uty_obj, *pab_table;
    PyObject *hi_eval_null_obj = NULL, *logl_H0_obj = NULL;
    int n_samples, n_grid, n_refine, n_threads, lmm_mode = 0;
    double l_min, l_max, logl_H0 = 0.0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiiiO|$iOO", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &UtW_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &pab_table, &lmm_mode, &hi_eval_null_obj, &logl_H0_obj)) {
        return NULL;
    }
    if (lmm_mode < 1 || lmm_mode > 4) {
        PyErr_Format(PyExc_ValueError,
            "lmm_mode must be 1, 2, 3 or 4, got %d", lmm_mode);
        return NULL;
    }
    int wants_hi = (lmm_mode == 3 || lmm_mode == 4);
    int wants_logl = (lmm_mode == 2 || lmm_mode == 4);
    if (hi_eval_null_obj == Py_None) hi_eval_null_obj = NULL;
    if (logl_H0_obj == Py_None) logl_H0_obj = NULL;
    if (wants_hi != (hi_eval_null_obj != NULL)) {
        PyErr_Format(PyExc_ValueError,
            "lmm_mode=%d %s hi_eval_null", lmm_mode,
            wants_hi ? "requires" : "does not take");
        return NULL;
    }
    if (wants_logl != (logl_H0_obj != NULL)) {
        PyErr_Format(PyExc_ValueError,
            "lmm_mode=%d %s logl_H0", lmm_mode,
            wants_logl ? "requires" : "does not take");
        return NULL;
    }
    if (wants_logl) {
        logl_H0 = PyFloat_AsDouble(logl_H0_obj);
        if (logl_H0 == -1.0 && PyErr_Occurred()) return NULL;
        if (validate_logl_H0(logl_H0) < 0) return NULL;
    }
    if (!PyDict_Check(pab_table)) {
        PyErr_SetString(PyExc_TypeError, "pab_table must be a dict");
        return NULL;
    }
    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *UtW_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_general_t *ws = NULL;
    PyObject *capsule = NULL;

    ws = (lmm_workspace_general_t *)calloc(1, sizeof(lmm_workspace_general_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    ws->mode = lmm_mode;
    if (parse_pab_table_from_dict(pab_table, &ws->table, n_samples) < 0)
        goto err_ws;
    int n_cvt = ws->table.n_cvt;

    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_ws;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_ws;
    uab_inv_arr = take_matrix(uab_inv_obj, ws->table.n_inv, n_samples, "uab_invariant");
    if (!uab_inv_arr) goto err_ws;
    UtW_arr = take_matrix(UtW_obj, n_samples, n_cvt, "UtW");
    if (!UtW_arr) goto err_ws;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_ws;
    if (wants_hi) {
        hi_eval_null_arr = take_vector(hi_eval_null_obj, n_samples, "hi_eval_null");
        if (!hi_eval_null_arr) goto err_ws;
        if (validate_hi_eval_null(
                (const double *)PyArray_DATA(hi_eval_null_arr), n_samples) < 0)
            goto err_ws;
    }

    if (init_fused_general_workspace(
            ws, eigenvalues_arr, uab_inv_arr, UtW_arr, Uty_arr,
            n_samples, l_min, l_max, n_grid, n_refine, n_threads) < 0)
        goto err_ws;

    if (wants_hi) {
        general_null_model_t *nm =
            (general_null_model_t *)calloc(1, sizeof(general_null_model_t));
        if (!nm) { PyErr_NoMemory(); goto err_ws; }

        nm->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
        if (!nm->hi_eval_null) { free(nm); PyErr_NoMemory(); goto err_ws; }
        memcpy(nm->hi_eval_null,
               (const double *)PyArray_DATA(hi_eval_null_arr),
               (size_t)n_samples * sizeof(double));

        /* Precompute null-model invariant sums */
        int n_inv = ws->table.n_inv;
        nm->null_inv_sums = (double *)malloc((size_t)n_inv * sizeof(double));
        if (!nm->null_inv_sums) {
            free(nm->hi_eval_null);
            free(nm);
            PyErr_NoMemory();
            goto err_ws;
        }
        for (int c = 0; c < n_inv; c++) {
            double s = 0.0;
            const double *col = ws->uab_inv + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += nm->hi_eval_null[i] * col[i];
            nm->null_inv_sums[c] = s;
        }
        ws->null_model = nm;
    }
    if (wants_logl) {
        general_lrt_t *lrt = (general_lrt_t *)calloc(1, sizeof(general_lrt_t));
        if (!lrt) { PyErr_NoMemory(); goto err_ws; }

        lrt->logl_H0 = logl_H0;
        lrt->mle_const = 0.5 * (double)n_samples
                         * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

        /* Pre-allocate per-thread LRT buffer (avoids per-SNP malloc in OpenMP loop).
         * Each thread needs (n_index * n_samples) doubles for row-major uab_snp. */
        int n_index = ws->table.n_index;
        lrt->uab_snp_flat = (double *)malloc(
            (size_t)ws->actual_threads * (size_t)n_index
            * (size_t)n_samples * sizeof(double));
        if (!lrt->uab_snp_flat) { free(lrt); PyErr_NoMemory(); goto err_ws; }
        ws->lrt = lrt;
    }

    capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(UtW_arr);
    Py_DECREF(Uty_arr);
    Py_XDECREF(hi_eval_null_arr);
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
 * compute_lmm_chunk_fused_general_c
 *
 * Per-chunk compute for one general (n_cvt >= 2) workspace, any lmm_mode.
 * Computes n_var varying dot products on-the-fly from UtW/Uty/UtG_T per SNP,
 * then feeds them into the table-driven Pab recursion and golden section.
 *
 * The workspace's lmm_mode picks which blocks of the per-SNP body run
 * (do_score, do_reml, do_lrt below) and which output arrays come back.
 *
 * Python signature:
 *   compute_lmm_chunk_fused_general_c(
 *       workspace,   # PyCapsule from create_workspace_general_c, any lmm_mode
 *       utg_t,       # (n_snps, n_samples) float64
 *       n_threads,   # int
 *   ) -> dict, keys depending on lmm_mode:
 *        1: lambdas, logls, betas, ses, pwalds
 *        2: lambdas_mle, p_lrts
 *        3: betas, ses, p_scores
 *        4: all eight keys above
 * ------------------------------------------------------------------------- */
PyObject *compute_lmm_chunk_fused_general_c_py(
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

    lmm_workspace_general_t *ws = general_workspace_any_mode(
        capsule_obj, "compute_lmm_chunk_fused_general_c");
    if (!ws) return NULL;

    const int do_score = (ws->mode == 3 || ws->mode == 4);
    const int do_reml  = (ws->mode == 1 || ws->mode == 4);
    const int do_lrt   = (ws->mode == 2 || ws->mode == 4);

    PyArrayObject *utg_t_arr = NULL;
    lmm_output_t out = {0};
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

    if (alloc_lmm_output(&out, (npy_intp)n_snps, ws->mode) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        goto err_input_fg;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    /* betas/ses hold Wald's beta/se whenever REML runs (modes 1 and 4);
     * mode 3 has no Wald block, so they hold Score's beta/se instead, the
     * same shape the ncvt1 Score loop returns standalone. Mode 2 (LRT
     * alone) allocates neither, so both stay NULL. */
    const int has_beta_se   = do_reml || do_score;
    double *out_lambdas     = do_reml    ? (double *)PyArray_DATA(out.lambdas)     : NULL;
    double *out_logls       = do_reml    ? (double *)PyArray_DATA(out.logls)       : NULL;
    double *out_betas       = has_beta_se ? (double *)PyArray_DATA(out.betas)     : NULL;
    double *out_ses         = has_beta_se ? (double *)PyArray_DATA(out.ses)       : NULL;
    double *out_pwalds      = do_reml    ? (double *)PyArray_DATA(out.pwalds)      : NULL;
    double *out_p_scores    = do_score   ? (double *)PyArray_DATA(out.p_scores)    : NULL;
    double *out_lambdas_mle = do_lrt     ? (double *)PyArray_DATA(out.lambdas_mle) : NULL;
    double *out_p_lrts      = do_lrt     ? (double *)PyArray_DATA(out.p_lrts)      : NULL;

    int n_grid = ws->grid->n_grid;
    int n_refine = ws->grid->n_refine;
    int df = ws->table.df;
    int n_index = ws->table.n_index;
    double reml_const = ws->reml_const;

    double log_l_min = ws->grid->log_l_min;
    double step = ws->grid->step;

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
            const double *a = get_fused_vector(ws, ws->table.var_a_cols[v], x);
            const double *b = get_fused_vector(ws, ws->table.var_b_cols[v], x);
            #pragma omp simd
            for (int i = 0; i < n_samples; i++)
                out_v[i] = a[i] * b[i];
        }

        /* ---- (a) Score: null-model Pab ---- */
        if (do_score) {
            double *null_row0 = my_row0;  /* reuse per-thread heap buffer */
            for (int i = 0; i < n_index; i++) null_row0[i] = 0.0;

            /* Invariant null sums from precomputed workspace */
            for (int c = 0; c < n_inv; c++)
                null_row0[ws->table.invariant_indices[c]] = ws->null_model->null_inv_sums[c];
            /* Varying null sums: weight scratch by hi_eval_null */
            for (int c = 0; c < n_var; c++) {
                double s = 0.0;
                const double *col = scratch + (size_t)c * n_samples;
                for (int i = 0; i < n_samples; i++)
                    s += ws->null_model->hi_eval_null[i] * col[i];
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

            /* Mode 3 has no Wald block below, so betas/ses carry Score's
             * beta/se here, the same shape a standalone Score compute
             * returns. Mode 4 overwrites both with Wald's below. */
            if (!do_reml) {
                out_betas[snp] = score_beta;
                out_ses[snp]   = score_se;
            }
        }

        /* ---- (b) Wald: REML optimization ---- */
        if (do_reml) {
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

            double logl_reml, wald_beta, wald_se, wald_f;
            int wald_valid;
            double lambda_reml = golden_section_lambda_general(
                ws->uab_inv, scratch, ws->eigenvalues,
                n_samples, ws->grid->lambda_grid, ws->grid->hi_eval_grid,
                ws->grid->logdet_h_grid, ws->grid->inv_sums_grid,
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
        }

        /* ---- (c) LRT: MLE optimization ---- */
        if (do_lrt) {
            /* MLE requires the full (n_samples, n_index) Uab for one SNP
             * in row-major layout (mle_logl_general_cached accesses as
             * uab_snp[sample * n_index + col]).
             * Assemble from ws->uab_inv (invariant) + scratch (varying).
             * Uses pre-allocated per-thread buffer from workspace to avoid
             * per-SNP malloc inside the OpenMP loop. */
            double *uab_snp = ws->lrt->uab_snp_flat +
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
                ws->grid->lambda_grid, ws->grid->hi_eval_grid, ws->grid->logdet_h_grid,
                log_l_min, step, n_grid, n_refine,
                ws->lrt->mle_const, &ws->table,
                &logl_H1,
                my_row0, my_pab
            );

            out_lambdas_mle[snp] = lambda_mle;

            double lrt_stat = 2.0 * (logl_H1 - ws->lrt->logl_H0);
            if (lrt_stat < 0.0) lrt_stat = 0.0;
            out_p_lrts[snp] = chi2_sf_c(lrt_stat);
        }
    }

    Py_END_ALLOW_THREADS

    if (do_reml && warn_betainc_convergence(out_betas, out_pwalds, n_snps) < 0)
        goto err_output_fg;

    result = build_lmm_result_dict(&out);
    if (!result) goto err_input_fg;

    Py_DECREF(utg_t_arr);
    return result;

err_output_fg:
    decref_lmm_output(&out);
err_input_fg:
    Py_XDECREF(utg_t_arr);
    return NULL;
}
