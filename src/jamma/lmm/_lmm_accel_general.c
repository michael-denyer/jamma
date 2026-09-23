/* General-covariate workspace ownership and Python compute entry points. */

#define NO_IMPORT_ARRAY
#include "_lmm_accel_internal.h"

#include "_lmm_kernels_general.h"
#include "_lmm_stats.h"
#include "_lmm_logdet.h"
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
 * The workspace constructs its canonical recursion table once from n_cvt.
 * Per-SNP kernels walk that immutable table.
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
#define GENERAL_CAPSULE "lmm_workspace_general"

/* Coarse-grid block: the lambda grid and its invariant dot products.
 * NULL for mode 3 (Score does no lambda search), non-NULL otherwise. */
typedef struct {
    lambda_search_t search;
    double *hi_eval_grid;   /* (n_grid * n_samples) */
    double *logdet_h_grid;  /* (n_grid,) */
    double *inv_sums_grid;  /* (n_grid * n_inv) — precomputed invariant dot products */
} general_grid_t;

/* Null-model block: modes 3 and 4 carry hi_eval_null and its invariant sums.
 * NULL unless the mode wants it. */
typedef struct {
    double *hi_eval_null;    /* (n_samples,) owned */
    double *null_inv_sums;   /* (n_inv,) precomputed null-model invariant sums, owned */
} general_null_model_t;

/* LRT block: modes 2 and 4 carry logl_H0 and mle_const. NULL unless the mode
 * wants it. */
typedef struct {
    double logl_H0;
    double mle_const;
} general_lrt_t;

/* Element count of every double buffer the family allocates; zero for a
 * buffer the mode does not use. The creator allocates from it and
 * general_layout_bytes prices it, so the plan and the allocation cannot
 * drift. The per-thread counts are for one of the creator's threads. */
typedef struct {
    int n_cvt;
    size_t retained;            /* Python-held eigenvalues, UtW, Uty,
                                 * Hi_eval_null and invariant SoA */
    size_t eigenvalues;
    size_t utw_transposed;
    size_t grid_points;         /* lambda_grid and logdet_h_grid each */
    size_t hi_eval_grid;        /* aligned */
    size_t inv_sums_grid;
    size_t inv_identity_sums;
    size_t hi_eval_null;        /* aligned */
    size_t null_inv_sums;
    size_t scratch, pab, dpab, row0;  /* per thread */
} general_layout_t;

static general_layout_t general_layout(int n_cvt, int n_samples, int n_grid,
                                       lmm_tests_t tests)
{
    size_t n = (size_t)n_samples;
    size_t rows = (size_t)n_cvt + 2;
    size_t index = ((size_t)n_cvt + 3) * rows / 2;
    size_t inv = index - rows;
    general_layout_t l = {0};
    l.n_cvt = n_cvt;
    l.retained = (3 + (size_t)n_cvt + inv) * n;
    l.eigenvalues = n;
    l.utw_transposed = (size_t)n_cvt * n;
    l.scratch = rows * n;
    l.pab = rows * index;
    l.row0 = index;
    if (tests.reml || tests.lrt) {
        l.grid_points = (size_t)n_grid;
        l.hi_eval_grid = n * (size_t)n_grid;
        l.inv_sums_grid = (size_t)n_grid * inv;
    }
    if (tests.reml) {
        l.inv_identity_sums = inv;
        l.dpab = rows * index;
    }
    if (tests.score) {
        l.hi_eval_null = n;
        l.null_inv_sums = inv;
    }
    return l;
}

static workspace_bytes_t general_layout_bytes(const general_layout_t *l)
{
    workspace_bytes_t b = {0};
    b.persistent = (l->retained + l->eigenvalues + l->utw_transposed
                    + 2 * l->grid_points + l->inv_sums_grid
                    + l->inv_identity_sums + l->null_inv_sums) * sizeof(double)
        + aligned_double_bytes(l->hi_eval_grid)
        + aligned_double_bytes(l->hi_eval_null)
        + pab_table_bytes(l->n_cvt);
    b.per_thread = (l->scratch + l->pab + l->dpab + l->row0)
        * sizeof(double);
    /* The fresh likelihood and score kernels each hold a MAX_N_INDEX
     * reduction buffer on the worker stack, whatever n_cvt is. */
    b.transient_per_thread = (size_t)2 * MAX_N_INDEX * sizeof(double);
    return b;
}

workspace_bytes_t general_workspace_bytes(int n_cvt, int n_samples, int n_grid,
                                          lmm_tests_t tests)
{
    general_layout_t l = general_layout(n_cvt, n_samples, n_grid, tests);
    return general_layout_bytes(&l);
}

typedef struct {
    /* Fixed params */
    double *eigenvalues;    /* (n_samples,) — owned copy */
    double reml_const;
    int n_samples;
    /* Packed Pab recursion table, built from n_cvt */
    pab_table_t table;
    /* Iab: invariant identity sums (precomputed, reused per-SNP) */
    double *inv_identity_sums;  /* (n_inv,) — sum of each invariant column at identity */
    /* F-distribution */
    double lbeta_ab;
    double beta_a, beta_b;
    /* Invariant SoA (reference, not owned — Python holds the array) */
    const double *uab_inv;
    PyObject *uab_inv_ref;      /* keeps uab_invariant_soa array alive */
    double *utw_transposed;     /* (n_cvt * n_samples) column-major, owned */
    const double *Uty;          /* (n_samples,) borrowed */
    double *scratch_flat;       /* (actual_threads * n_var * n_samples) owned */
    int actual_threads;         /* for scratch deallocation sizing */
    /* Per-thread heap buffers for Pab recursion (replaces stack arrays) */
    double *pab_per_thread;     /* (actual_threads * pab_size) owned */
    double *dpab_per_thread;    /* same shape, REML score derivative */
    double *row0_per_thread;    /* (actual_threads * n_index) owned */
    int pab_size;               /* n_rows * n_index for this workspace */
    PyObject *Uty_ref;          /* keeps Uty array alive */
    lmm_tests_t tests;
    general_layout_t layout;
    /* Sub-blocks: NULL when the owning mode does not use them, so
     * ws->lrt == NULL is the contract rather than a comment. */
    general_grid_t *grid;
    general_null_model_t *null_model;
    general_lrt_t *lrt;
} lmm_workspace_general_t;

/* PyCapsule destructor for general workspace */
static void lmm_workspace_general_free(lmm_workspace_general_t *ws)
{
    if (!ws) return;
    if (ws->grid) {
        free(ws->grid->search.lambda_grid);
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
    free(ws->dpab_per_thread);
    free(ws->row0_per_thread);
    Py_XDECREF(ws->Uty_ref);
    if (ws->null_model) {
        free(ws->null_model->hi_eval_null);
        free(ws->null_model->null_inv_sums);
        free(ws->null_model);
    }
    free(ws->lrt);
    free(ws);
}

int general_capsule_bytes(PyObject *capsule, workspace_bytes_t *out)
{
    if (!PyCapsule_IsValid(capsule, GENERAL_CAPSULE)) return 0;
    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)
        PyCapsule_GetPointer(capsule, GENERAL_CAPSULE);
    *out = general_layout_bytes(&ws->layout);
    return 1;
}

static void lmm_workspace_general_destructor(PyObject *cap)
{
    lmm_workspace_general_free((lmm_workspace_general_t *)
        PyCapsule_GetPointer(cap, GENERAL_CAPSULE));
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
    int n_cvt = ws->table.n_cvt;
    assert(col_0based >= 0 && col_0based <= n_cvt + 1);
    if (col_0based < n_cvt)
        return ws->utw_transposed + (size_t)col_0based * ws->n_samples;
    if (col_0based == n_cvt)
        return x;
    return ws->Uty;  /* col_0based == n_cvt + 1 */
}

/* The lambda grid, its log-determinants and invariant dot products, for the
 * tests that search lambda. 0, or -1 with PyErr set. */
static int init_general_grid(
    lmm_workspace_general_t *ws, double l_min, double l_max, int n_grid,
    int n_refine)
{
    int n_samples = ws->n_samples;
    int n_inv = ws->table.n_inv;
    double log_l_min = log(l_min);
    double log_l_max_v = log(l_max);
    double step = (log_l_max_v - log_l_min) / (double)(n_grid - 1);

    general_grid_t *grid = (general_grid_t *)calloc(1, sizeof(general_grid_t));
    if (!grid) { PyErr_NoMemory(); return -1; }
    double *lambda_grid = (double *)malloc(ws->layout.grid_points * sizeof(double));
    grid->search = (lambda_search_t){
        .lambda_grid = lambda_grid,
        .log_l_min = log_l_min, .step = step,
        .n_grid = n_grid, .n_refine = n_refine,
    };
    grid->hi_eval_grid = alloc_aligned_doubles(ws->layout.hi_eval_grid);
    grid->logdet_h_grid = (double *)malloc(ws->layout.grid_points * sizeof(double));
    grid->inv_sums_grid = (double *)malloc(
        ws->layout.inv_sums_grid * sizeof(double));

    if (!lambda_grid || !grid->hi_eval_grid ||
        !grid->logdet_h_grid || !grid->inv_sums_grid) {
        free(lambda_grid);
        free(grid->hi_eval_grid);
        free(grid->logdet_h_grid);
        free(grid->inv_sums_grid);
        free(grid);
        PyErr_NoMemory();
        return -1;
    }

    for (int g = 0; g < n_grid; g++)
        lambda_grid[g] = exp(log_l_min + g * step);

    /* Precompute hi_eval_grid, logdet_h_grid, and invariant sums */
    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi_row = grid->hi_eval_grid + (size_t)g * n_samples;

        for (int i = 0; i < n_samples; i++)
            hi_row[i] = 1.0 / (lam * ws->eigenvalues[i] + 1.0);
        grid->logdet_h_grid[g] = logdet_h_lambda(ws->eigenvalues, n_samples, lam);

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
    return 0;
}

/* Fill a calloc'd general workspace whose table build_pab_table already built:
 * eigenvalues, uab_inv, UtW (transposed), Uty, per-thread scratch, the
 * lambda grid and invariant sums the layout asks for, and the beta/REML
 * constants. 0, or -1 with PyErr set; the caller frees ws through
 * lmm_workspace_general_free. */
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
    const general_layout_t *layout = &ws->layout;

    ws->n_samples = n_samples;

    /* Copy eigenvalues (owned) */
    ws->eigenvalues = (double *)malloc(layout->eigenvalues * sizeof(double));
    if (!ws->eigenvalues) { PyErr_NoMemory(); return -1; }
    memcpy(ws->eigenvalues, PyArray_DATA(eigenvalues_arr),
           (size_t)n_samples * sizeof(double));

    Py_INCREF(uab_inv_arr);
    ws->uab_inv_ref = (PyObject *)uab_inv_arr;
    ws->uab_inv = (const double *)PyArray_DATA(uab_inv_arr);

    /* Transpose UtW from row-major (n_samples, n_cvt) to column-major
     * (n_cvt, n_samples) for cache-friendly per-column access. */
    ws->utw_transposed = (double *)malloc(
        layout->utw_transposed * sizeof(double));
    if (!ws->utw_transposed) { PyErr_NoMemory(); return -1; }
    {
        const double *src = (const double *)PyArray_DATA(UtW_arr);
        for (int c = 0; c < n_cvt; c++) {
            double *dst = ws->utw_transposed + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                dst[i] = src[(size_t)i * n_cvt + c];
        }
    }

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
        (size_t)actual_threads * layout->scratch * sizeof(double));
    if (!ws->scratch_flat) { PyErr_NoMemory(); return -1; }

    /* Per-thread heap buffers for Pab recursion (avoids stack overflow) */
    int pab_size = n_rows * n_index;
    ws->pab_size = pab_size;
    ws->pab_per_thread = (double *)malloc(
        (size_t)actual_threads * layout->pab * sizeof(double));
    if (!ws->pab_per_thread) { PyErr_NoMemory(); return -1; }
    if (layout->dpab) {
        ws->dpab_per_thread = (double *)malloc(
            (size_t)actual_threads * layout->dpab * sizeof(double));
        if (!ws->dpab_per_thread) { PyErr_NoMemory(); return -1; }
    }
    ws->row0_per_thread = (double *)malloc(
        (size_t)actual_threads * layout->row0 * sizeof(double));
    if (!ws->row0_per_thread) { PyErr_NoMemory(); return -1; }

    /* Compute df, reml_const, beta params */
    int df = ws->table.df;
    ws->beta_a = (double)df / 2.0;
    ws->beta_b = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);
    ws->reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    if (layout->grid_points &&
        init_general_grid(ws, l_min, l_max, n_grid, n_refine) < 0)
        return -1;

    if (!layout->inv_identity_sums) return 0;
    ws->inv_identity_sums = (double *)malloc(
        layout->inv_identity_sums * sizeof(double));
    if (!ws->inv_identity_sums) { PyErr_NoMemory(); return -1; }
    for (int c = 0; c < ws->table.n_inv; c++) {
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
 *       n_cvt,            # number of covariates, including intercept
 *       *, lmm_mode, hi_eval_null=None, logl_H0=None,
 *   ) -> PyCapsule
 *
 * Every index array is derived from n_cvt. lmm_mode is 1 (Wald),
 * 2 (LRT), 3 (Score) or 4 (all three); hi_eval_null is required by 3 and 4,
 * logl_H0 by 2 and 4, matching create_workspace_ncvt1_c's mode contract.
 * ------------------------------------------------------------------------- */
PyObject *create_workspace_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "UtW", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "n_cvt", "lmm_mode", "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *UtW_obj, *Uty_obj;
    PyObject *hi_eval_null_obj = NULL, *logl_H0_obj = NULL;
    int n_samples, n_grid, n_refine, n_threads, n_cvt, lmm_mode = 0;
    double l_min, l_max, logl_H0 = 0.0;
    lmm_tests_t tests;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiiii|$iOO", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &UtW_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &n_cvt, &lmm_mode, &hi_eval_null_obj, &logl_H0_obj)) {
        return NULL;
    }
    if (parse_mode_inputs(lmm_mode, &hi_eval_null_obj, logl_H0_obj,
                          &tests, &logl_H0) < 0)
        return NULL;
    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *UtW_arr = NULL, *Uty_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_general_t *ws = NULL;
    PyObject *capsule = NULL;

    ws = (lmm_workspace_general_t *)calloc(1, sizeof(lmm_workspace_general_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    ws->tests = tests;
    ws->layout = general_layout(n_cvt, n_samples, n_grid, tests);
    if (build_pab_table(n_cvt, &ws->table, n_samples) < 0)
        goto err_ws;

    eigenvalues_arr = take_vector(eigenvalues_obj, n_samples, "eigenvalues");
    if (!eigenvalues_arr) goto err_ws;
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples, l_max) < 0)
        goto err_ws;
    uab_inv_arr = take_matrix(uab_inv_obj, ws->table.n_inv, n_samples, "uab_invariant");
    if (!uab_inv_arr) goto err_ws;
    UtW_arr = take_matrix(UtW_obj, n_samples, n_cvt, "UtW");
    if (!UtW_arr) goto err_ws;
    Uty_arr = take_vector(Uty_obj, n_samples, "Uty");
    if (!Uty_arr) goto err_ws;
    if (tests.score) {
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

    if (tests.score) {
        general_null_model_t *nm =
            (general_null_model_t *)calloc(1, sizeof(general_null_model_t));
        if (!nm) { PyErr_NoMemory(); goto err_ws; }

        nm->hi_eval_null = alloc_aligned_doubles(ws->layout.hi_eval_null);
        if (!nm->hi_eval_null) { free(nm); PyErr_NoMemory(); goto err_ws; }
        memcpy(nm->hi_eval_null,
               (const double *)PyArray_DATA(hi_eval_null_arr),
               (size_t)n_samples * sizeof(double));

        /* Precompute null-model invariant sums */
        int n_inv = ws->table.n_inv;
        nm->null_inv_sums = (double *)malloc(
            ws->layout.null_inv_sums * sizeof(double));
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
    if (tests.lrt) {
        general_lrt_t *lrt = (general_lrt_t *)calloc(1, sizeof(general_lrt_t));
        if (!lrt) { PyErr_NoMemory(); goto err_ws; }

        lrt->logl_H0 = logl_H0;
        lrt->mle_const = 0.5 * (double)n_samples
                         * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);
        ws->lrt = lrt;
    }

    capsule = PyCapsule_New(
        ws, GENERAL_CAPSULE, lmm_workspace_general_destructor);
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

static double general_score_block(
    const lmm_workspace_general_t *ws, const double *scratch,
    double *row0, double *pab, double *beta_out, double *se_out)
{
    const pab_table_t *t = &ws->table;
    int n_samples = ws->n_samples;

    for (int i = 0; i < t->n_index; i++) row0[i] = 0.0;
    for (int c = 0; c < t->n_inv; c++)
        row0[t->invariant_indices[c]] = ws->null_model->null_inv_sums[c];
    for (int c = 0; c < t->n_var; c++) {
        double s = 0.0;
        const double *col = scratch + (size_t)c * n_samples;
        for (int i = 0; i < n_samples; i++)
            s += ws->null_model->hi_eval_null[i] * col[i];
        row0[t->varying_indices[c]] = s;
    }

    calc_pab_general(row0, t, pab);

    double score_f;
    int score_valid = score_from_pab_general(
        pab, t, n_samples, beta_out, se_out, &score_f);
    return f_to_pvalue(score_f, t->df, score_valid,
                       ws->beta_a, ws->beta_b, ws->lbeta_ab);
}

static double general_reml_block(
    const lmm_workspace_general_t *ws, const double *scratch,
    double *row0, double *pab, double *dpab,
    double *logl_out, double *beta_out, double *se_out, double *pwald_out)
{
    const pab_table_t *t = &ws->table;
    const general_grid_t *grid = ws->grid;
    int n_samples = ws->n_samples;

    for (int i = 0; i < t->n_index; i++) row0[i] = 0.0;
    for (int c = 0; c < t->n_inv; c++)
        row0[t->invariant_indices[c]] = ws->inv_identity_sums[c];
    for (int c = 0; c < t->n_var; c++) {
        double s = 0.0;
        const double *col = scratch + (size_t)c * n_samples;
        for (int i = 0; i < n_samples; i++) s += col[i];
        row0[t->varying_indices[c]] = s;
    }

    double logdet_iab = logdet_from_row0(row0, t, pab);

    const general_snp_t snp = {
        .uab_inv = ws->uab_inv, .uab_var = scratch,
        .eigenvalues = ws->eigenvalues, .n_samples = n_samples, .t = t,
        .logdet_iab = logdet_iab, .reml_const = ws->reml_const,
        .row0 = row0, .pab = pab, .dpab = dpab,
    };
    int best_idx = coarse_grid_reml_general(
        &snp, grid->hi_eval_grid, grid->logdet_h_grid, grid->inv_sums_grid,
        grid->search.n_grid);
    double wald_f;
    int wald_valid;
    double lambda_reml = refine_lambda_general(
        &snp, &grid->search, best_idx,
        logl_out, beta_out, se_out, &wald_f, &wald_valid);
    *pwald_out = f_to_pvalue(wald_f, t->df, wald_valid,
                             ws->beta_a, ws->beta_b, ws->lbeta_ab);
    return lambda_reml;
}

static double general_lrt_block(
    const lmm_workspace_general_t *ws, const double *scratch,
    double *row0, double *pab,
    double *logl_H1_out, double *p_lrt_out)
{
    const pab_table_t *t = &ws->table;
    const general_grid_t *grid = ws->grid;
    const general_snp_t snp = {
        .uab_inv = ws->uab_inv, .uab_var = scratch,
        .eigenvalues = ws->eigenvalues, .n_samples = ws->n_samples, .t = t,
        .mle_const = ws->lrt->mle_const,
        .row0 = row0, .pab = pab,
    };
    int best_idx = coarse_grid_mle_general(
        &snp, grid->hi_eval_grid, grid->logdet_h_grid, grid->inv_sums_grid,
        grid->search.n_grid);
    double lambda_mle = refine_lambda_mle_general(
        &snp, &grid->search, best_idx, logl_H1_out);

    double lrt_stat = 2.0 * (*logl_H1_out - ws->lrt->logl_H0);
    if (lrt_stat < 0.0) lrt_stat = 0.0;
    *p_lrt_out = chi2_sf_c(lrt_stat);
    return lambda_mle;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_fused_general_c
 *
 * Per-chunk compute for one general (n_cvt >= 2) workspace, any lmm_mode.
 *
 * Python signature:
 *   compute_lmm_chunk_fused_general_c(
 *       workspace,   # PyCapsule from create_workspace_general_c, any lmm_mode
 *       utg_t,       # (n_snps, n_samples) float64
 *       n_threads,   # int
 *   ) -> dict, keys depending on lmm_mode:
 *        1: lambdas, logls, betas, ses, pwalds
 *        2: logls, lambdas_mle, p_lrts
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

    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)
        PyCapsule_GetPointer(capsule_obj, GENERAL_CAPSULE);
    if (!ws) return NULL;

    const lmm_tests_t tests = ws->tests;
    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_index = ws->table.n_index;
    int n_snps;
    PyArrayObject *utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;

    lmm_output_t out = {0};
    if (alloc_lmm_output(&out, (npy_intp)n_snps, tests) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        Py_DECREF(utg_t_arr);
        return NULL;
    }

    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    double *out_lambdas     = tests.reml ? (double *)PyArray_DATA(out.lambdas) : NULL;
    double *out_logls       = (tests.reml || tests.lrt)
        ? (double *)PyArray_DATA(out.logls) : NULL;
    double *out_betas       = (tests.reml || tests.score)
        ? (double *)PyArray_DATA(out.betas) : NULL;
    double *out_ses         = (tests.reml || tests.score)
        ? (double *)PyArray_DATA(out.ses) : NULL;
    double *out_pwalds      = tests.reml ? (double *)PyArray_DATA(out.pwalds) : NULL;
    double *out_p_scores    = tests.score ? (double *)PyArray_DATA(out.p_scores) : NULL;
    double *out_lambdas_mle = tests.lrt ? (double *)PyArray_DATA(out.lambdas_mle) : NULL;
    double *out_p_lrts      = tests.lrt ? (double *)PyArray_DATA(out.p_lrts) : NULL;

    int actual_threads = clamp_threads(n_threads, n_snps);
    if (actual_threads > ws->actual_threads) actual_threads = ws->actual_threads;

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
        double *my_dpab = ws->dpab_per_thread
            ? ws->dpab_per_thread + (size_t)tid * ws->pab_size : NULL;
        double *my_row0 = ws->row0_per_thread + (size_t)tid * n_index;

        for (int v = 0; v < n_var; v++) {
            double *out_v = scratch + (size_t)v * n_samples;
            const double *a = get_fused_vector(ws, ws->table.var_a_cols[v], x);
            const double *b = get_fused_vector(ws, ws->table.var_b_cols[v], x);
            #pragma omp simd
            for (int i = 0; i < n_samples; i++)
                out_v[i] = a[i] * b[i];
        }

        if (tests.score) {
            double score_beta, score_se;
            out_p_scores[snp] = general_score_block(
                ws, scratch, my_row0, my_pab, &score_beta, &score_se);
            if (!tests.reml) {
                out_betas[snp] = score_beta;
                out_ses[snp]   = score_se;
            }
        }

        if (tests.reml) {
            out_lambdas[snp] = general_reml_block(
                ws, scratch, my_row0, my_pab, my_dpab,
                &out_logls[snp], &out_betas[snp], &out_ses[snp],
                &out_pwalds[snp]);
        }

        if (tests.lrt) {
            /* GEMMA modes 2 and 4 report the LRT alternative-model MLE
             * likelihood in logl_H1, overwriting mode 4's REML logl. */
            out_lambdas_mle[snp] = general_lrt_block(
                ws, scratch, my_row0, my_pab,
                &out_logls[snp], &out_p_lrts[snp]);
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(utg_t_arr);
    return finish_lmm_output(&out, tests, n_snps);
}
