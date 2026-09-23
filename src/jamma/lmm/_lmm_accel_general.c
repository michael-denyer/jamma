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
    if (tests.reml)
        l.inv_identity_sums = inv;
    if (tests.reml || tests.lrt)
        l.dpab = rows * index;
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
    /* The fresh likelihood and score kernels each also hold a MAX_N_INDEX
     * reduction buffer on the worker stack, whatever n_cvt is. */
    b.per_thread = (l->scratch + l->pab + l->dpab + l->row0
                    + (size_t)2 * MAX_N_INDEX) * sizeof(double);
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
    double *dpab_per_thread;    /* same shape, REML and MLE score derivative */
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
    general_grid_t *grid = (general_grid_t *)calloc(1, sizeof(general_grid_t));
    if (!grid) { PyErr_NoMemory(); return -1; }
    double *lambda_grid = (double *)malloc(ws->layout.grid_points * sizeof(double));
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

    build_lambda_grid(&grid->search, l_min, l_max, n_grid, n_refine,
                      ws->eigenvalues, ws->n_samples,
                      ws->uab_inv, ws->table.n_inv, lambda_grid,
                      grid->hi_eval_grid, grid->logdet_h_grid,
                      grid->inv_sums_grid);
    ws->grid = grid;
    return 0;
}

/* Fill a calloc'd general workspace whose table build_pab_table already built:
 * eigenvalues, uab_inv, UtW (transposed), Uty, per-thread scratch, the
 * lambda grid and invariant sums the layout asks for, and the beta/REML
 * constants. 0, or -1 with PyErr set; the caller frees ws through
 * lmm_workspace_general_free. */
static int init_fused_general_workspace(lmm_workspace_general_t *ws,
                                        const workspace_inputs_t *in)
{
    int n_samples = in->n_samples;
    int n_cvt   = ws->table.n_cvt;
    int n_index = ws->table.n_index;
    int n_rows  = ws->table.n_rows;
    const general_layout_t *layout = &ws->layout;

    ws->n_samples = n_samples;

    /* Copy eigenvalues (owned) */
    ws->eigenvalues = (double *)malloc(layout->eigenvalues * sizeof(double));
    if (!ws->eigenvalues) { PyErr_NoMemory(); return -1; }
    memcpy(ws->eigenvalues, PyArray_DATA(in->eigenvalues),
           (size_t)n_samples * sizeof(double));

    Py_INCREF(in->uab_inv);
    ws->uab_inv_ref = (PyObject *)in->uab_inv;
    ws->uab_inv = (const double *)PyArray_DATA(in->uab_inv);

    /* Transpose UtW from row-major (n_samples, n_cvt) to column-major
     * (n_cvt, n_samples) for cache-friendly per-column access. */
    ws->utw_transposed = (double *)malloc(
        layout->utw_transposed * sizeof(double));
    if (!ws->utw_transposed) { PyErr_NoMemory(); return -1; }
    {
        const double *src = (const double *)PyArray_DATA(in->UtW);
        for (int c = 0; c < n_cvt; c++) {
            double *dst = ws->utw_transposed + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                dst[i] = src[(size_t)i * n_cvt + c];
        }
    }

    /* Borrow Uty pointer */
    Py_INCREF(in->Uty);
    ws->Uty_ref = (PyObject *)in->Uty;
    ws->Uty = (const double *)PyArray_DATA(in->Uty);

    /* Allocate per-thread scratch: n_var * n_samples per thread */
    int actual_threads = in->n_threads;
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
        init_general_grid(ws, in->l_min, in->l_max, in->n_grid, in->n_refine) < 0)
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

/* The general workspace for in->tests: builds the Pab table from in->n_cvt,
 * then the blocks the tests need. */
PyObject *general_create_workspace(const workspace_inputs_t *in)
{
    const lmm_tests_t tests = in->tests;
    int n_samples = in->n_samples;
    PyObject *capsule;
    lmm_workspace_general_t *ws =
        (lmm_workspace_general_t *)calloc(1, sizeof(lmm_workspace_general_t));
    if (!ws) return PyErr_NoMemory();
    ws->tests = tests;
    ws->layout = general_layout(in->n_cvt, n_samples, in->n_grid, tests);
    if (build_pab_table(in->n_cvt, &ws->table, n_samples) < 0)
        goto err_ws;

    if (init_fused_general_workspace(ws, in) < 0)
        goto err_ws;

    if (tests.score) {
        general_null_model_t *nm =
            (general_null_model_t *)calloc(1, sizeof(general_null_model_t));
        if (!nm) { PyErr_NoMemory(); goto err_ws; }

        nm->hi_eval_null = alloc_aligned_doubles(ws->layout.hi_eval_null);
        if (!nm->hi_eval_null) { free(nm); PyErr_NoMemory(); goto err_ws; }
        memcpy(nm->hi_eval_null,
               (const double *)PyArray_DATA(in->hi_eval_null),
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

        lrt->logl_H0 = in->logl_H0;
        lrt->mle_const = 0.5 * (double)n_samples
                         * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);
        ws->lrt = lrt;
    }

    capsule = PyCapsule_New(
        ws, GENERAL_CAPSULE, lmm_workspace_general_destructor);
    if (capsule) return capsule;

err_ws:
    lmm_workspace_general_free(ws);
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
    int score_valid = score_stats(pab_terms_general(pab, t), n_samples, t->df,
                                  beta_out, se_out, &score_f);
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
    double *row0, double *pab, double *dpab,
    double *logl_H1_out, double *p_lrt_out)
{
    const pab_table_t *t = &ws->table;
    const general_grid_t *grid = ws->grid;
    const general_snp_t snp = {
        .uab_inv = ws->uab_inv, .uab_var = scratch,
        .eigenvalues = ws->eigenvalues, .n_samples = ws->n_samples, .t = t,
        .mle_const = ws->lrt->mle_const,
        .row0 = row0, .pab = pab, .dpab = dpab,
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

int general_compute_chunk(PyObject *capsule, PyObject *utg_t_obj,
                          int n_threads, PyObject **result)
{
    if (!PyCapsule_IsValid(capsule, GENERAL_CAPSULE)) return 0;
    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)
        PyCapsule_GetPointer(capsule, GENERAL_CAPSULE);
    *result = NULL;

    const lmm_tests_t tests = ws->tests;
    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_index = ws->table.n_index;
    int n_snps;
    PyArrayObject *utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return 1;

    lmm_output_t out = {0};
    if (alloc_lmm_output(&out, (npy_intp)n_snps, tests) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        Py_DECREF(utg_t_arr);
        return 1;
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

    int actual_threads = clamp_threads(n_threads, n_snps, ws->actual_threads);

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
                ws, scratch, my_row0, my_pab, my_dpab,
                &out_logls[snp], &out_p_lrts[snp]);
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(utg_t_arr);
    *result = finish_lmm_output(&out, tests, n_snps);
    return 1;
}
