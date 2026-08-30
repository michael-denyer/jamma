/* n_cvt=1 workspace ownership and Python compute entry points. */

#define NO_IMPORT_ARRAY
#include "_lmm_accel_internal.h"

#include "_lmm_kernels_ncvt1.h"
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
 * n_cvt = 1 workspace: the per-run state every n_cvt=1 test shares.
 *
 * One struct for Wald (mode 1), LRT (2), Score (3) and all three (4). The
 * lambda grid, its per-grid invariant dot products and the Iab scalars are
 * built once per run; the null-model block is filled only for modes that use
 * it and stays NULL otherwise. Python arrays are kept alive via Py_INCREF
 * until the workspace is freed.
 * ========================================================================= */

#define NCVT1_CAPSULE "lmm_workspace_ncvt1"

/* Coarse-grid block: the lambda grid and its per-grid precomputed data.
 * NULL for mode 3 (Score does no lambda search), non-NULL otherwise. */
typedef struct {
    int n_grid;
    int n_refine;
    double log_l_min, step;   /* bracket endpoints as computed at creation */
    double *lambda_grid;      /* (n_grid,) */
    double *hi_eval_grid;     /* (n_grid * n_samples) */
    double *logdet_h_grid;    /* (n_grid,) */
    grid_invariant_t *grid_inv;  /* (n_grid,) */
} ncvt1_grid_t;

/* Null-model block: the null-Hi_eval-weighted invariant dot products every
 * Score or LRT test needs. NULL unless mode is 3 or 4. */
typedef struct {
    double *hi_eval_null;   /* (n_samples,) null-model Hi_eval, owned */
    double s_ww, s_wy, s_yy; /* invariant dot products under null Hi_eval */
} ncvt1_null_model_t;

/* LRT block: the null MLE log-likelihood and its normalizing constant.
 * NULL unless mode is 2 or 4. */
typedef struct {
    double logl_H0;    /* null MLE log-likelihood */
    double mle_const;  /* 0.5 * n * (log(n) - log(2*pi) - 1) */
} ncvt1_lrt_t;

/* Score block: h_null_w/h_null_Uty fold hi_eval_null into w and Uty once per
 * run so the mode-3-only loop sums (h*w)*x per SNP. NULL unless mode is 3. */
typedef struct {
    double *h_null_w;    /* (n_samples,) hi_eval_null * w */
    double *h_null_Uty;  /* (n_samples,) hi_eval_null * Uty */
} ncvt1_score_t;

typedef struct {
    int n_samples;
    int df;
    double reml_const;
    double beta_a, beta_b, lbeta_ab;
    /* Invariant Iab scalars (lambda-independent) */
    double iab_inv_ww;  /* 1/sum(inv_ww) (or 0) */
    double iab_log_ww;  /* log(sum(inv_ww)) (or 0) */
    /* Borrowed pointers — kept alive via Py_INCREF */
    const double *eigenvalues;
    const double *inv_ww;   /* uab_invariant_soa row 0 */
    const double *inv_wy;   /* uab_invariant_soa row 1 */
    const double *inv_yy;   /* uab_invariant_soa row 2 */
    PyObject *eigenvalues_ref;  /* keeps eigenvalues array alive */
    PyObject *uab_inv_ref;      /* keeps uab_invariant_soa array alive */
    /* The lmm_mode the workspace was created for: 1 Wald, 2 LRT, 3 Score,
     * 4 all three. Each compute entry point checks it. */
    int mode;
    /* Sub-blocks: NULL when the owning mode does not use them, so ws->lrt
     * == NULL is the contract rather than a comment. */
    ncvt1_grid_t *grid;
    ncvt1_null_model_t *null_model;
    ncvt1_lrt_t *lrt;
    ncvt1_score_t *score;
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
    if (ws->grid) {
        free(ws->grid->lambda_grid);
        free(ws->grid->hi_eval_grid);
        free(ws->grid->logdet_h_grid);
        free(ws->grid->grid_inv);
        free(ws->grid);
    }
    if (ws->null_model) {
        free(ws->null_model->hi_eval_null);
        free(ws->null_model);
    }
    free(ws->lrt);
    if (ws->score) {
        free(ws->score->h_null_w);
        free(ws->score->h_null_Uty);
        free(ws->score);
    }
    Py_XDECREF(ws->eigenvalues_ref);
    Py_XDECREF(ws->uab_inv_ref);
    Py_XDECREF(ws->w_ref);
    Py_XDECREF(ws->Uty_ref);
    free(ws);
}

static void lmm_workspace_destructor(PyObject *cap)
{
    lmm_workspace_free(
        (lmm_workspace_t *)PyCapsule_GetPointer(cap, NCVT1_CAPSULE));
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

/* Fill a calloc'd n_cvt=1 workspace from validated inputs: the scalar
 * constants, the borrowed array pointers (INCREF'd here, released by
 * lmm_workspace_free), the invariant Iab scalar and, unless with_grid is 0,
 * the lambda grid. Score (mode 3) does no lambda search and skips the grid.
 * 0, or -1 with PyErr set. */
static int init_ncvt1_workspace(
    lmm_workspace_t *ws,
    PyArrayObject *eigenvalues_arr, PyArrayObject *uab_inv_arr,
    PyArrayObject *w_arr, PyArrayObject *Uty_arr,
    int n_samples, double l_min, double l_max, int n_grid, int n_refine,
    int with_grid)
{
    ws->n_samples = n_samples;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);
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
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    if (!with_grid) return 0;

    ncvt1_grid_t *grid = (ncvt1_grid_t *)calloc(1, sizeof(ncvt1_grid_t));
    if (!grid) { PyErr_NoMemory(); return -1; }

    grid->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    grid->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    grid->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));
    if (!grid->lambda_grid || !grid->hi_eval_grid ||
        !grid->logdet_h_grid || !grid->grid_inv) {
        free(grid->lambda_grid);
        free(grid->hi_eval_grid);
        free(grid->logdet_h_grid);
        free(grid->grid_inv);
        free(grid);
        PyErr_NoMemory();
        return -1;
    }

    build_grid_ncvt1(n_grid, n_samples, log_l_min, step,
                     ws->eigenvalues, ws->inv_ww, ws->inv_wy, ws->inv_yy,
                     grid->lambda_grid, grid->hi_eval_grid, grid->logdet_h_grid,
                     grid->grid_inv);
    grid->n_grid = n_grid;
    grid->n_refine = n_refine;
    grid->log_l_min = log_l_min;
    grid->step = step;
    ws->grid = grid;
    return 0;
}

/* The owned copy of the null-model Hi_eval and its invariant dot products,
 * for the Score test (modes 3, 4). 0, or -1 with PyErr set. */
static int init_ncvt1_null_hi(lmm_workspace_t *ws, const double *hi_eval_null)
{
    int n_samples = ws->n_samples;
    ncvt1_null_model_t *nm =
        (ncvt1_null_model_t *)calloc(1, sizeof(ncvt1_null_model_t));
    if (!nm) { PyErr_NoMemory(); return -1; }

    nm->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!nm->hi_eval_null) {
        free(nm);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(nm->hi_eval_null, hi_eval_null, (size_t)n_samples * sizeof(double));

    {
        double ns_ww = 0.0, ns_wy = 0.0, ns_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double h = nm->hi_eval_null[i];
            ns_ww += h * ws->inv_ww[i];
            ns_wy += h * ws->inv_wy[i];
            ns_yy += h * ws->inv_yy[i];
        }
        nm->s_ww = ns_ww;
        nm->s_wy = ns_wy;
        nm->s_yy = ns_yy;
    }
    ws->null_model = nm;
    return 0;
}

/* The null MLE log-likelihood and the MLE constant, for the LRT (modes 2, 4).
 * 0, or -1 with PyErr set. */
static int set_ncvt1_null_logl(lmm_workspace_t *ws, double logl_H0)
{
    int n_samples = ws->n_samples;
    ncvt1_lrt_t *lrt = (ncvt1_lrt_t *)calloc(1, sizeof(ncvt1_lrt_t));
    if (!lrt) { PyErr_NoMemory(); return -1; }
    lrt->logl_H0 = logl_H0;
    lrt->mle_const = 0.5 * (double)n_samples
                     * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);
    ws->lrt = lrt;
    return 0;
}

/* Score (mode 3) folds hi_eval_null into w and Uty once per run. The kernel
 * then sums (h*w)*x per SNP; mode 4 sums h*(w*x) instead, and the two
 * associations are not bit-identical, so this stays a mode-3 block.
 * Requires init_ncvt1_null_hi first. 0, or -1 with PyErr set. */
static int init_ncvt1_score_vectors(lmm_workspace_t *ws)
{
    int n_samples = ws->n_samples;
    ncvt1_score_t *sc = (ncvt1_score_t *)calloc(1, sizeof(ncvt1_score_t));
    if (!sc) { PyErr_NoMemory(); return -1; }

    sc->h_null_w = alloc_aligned_doubles((size_t)n_samples);
    sc->h_null_Uty = alloc_aligned_doubles((size_t)n_samples);
    if (!sc->h_null_w || !sc->h_null_Uty) {
        free(sc->h_null_w);
        free(sc->h_null_Uty);
        free(sc);
        PyErr_NoMemory();
        return -1;
    }
    const double *hi = ws->null_model->hi_eval_null;
    for (int i = 0; i < n_samples; i++) {
        sc->h_null_w[i]   = hi[i] * ws->w[i];
        sc->h_null_Uty[i] = hi[i] * ws->Uty[i];
    }
    ws->score = sc;
    return 0;
}

/* -------------------------------------------------------------------------
 * create_workspace_ncvt1_c
 *
 * Python signature:
 *   create_workspace_ncvt1_c(
 *       eigenvalues, uab_invariant, w, Uty,
 *       n_samples, l_min, l_max, n_grid, n_refine,
 *       *, lmm_mode, hi_eval_null=None, logl_H0=None,
 *   ) -> PyCapsule
 *
 * lmm_mode picks the null-model inputs: 2 (LRT) needs logl_H0, 3 (Score)
 * needs hi_eval_null, 4 needs both, 1 (Wald) takes neither. An input the
 * mode does not use is rejected rather than ignored.
 * ------------------------------------------------------------------------- */
PyObject *create_workspace_ncvt1_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "w", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine",
        "lmm_mode", "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *w_obj, *Uty_obj;
    PyObject *hi_eval_null_obj = NULL, *logl_H0_obj = NULL;
    int n_samples, n_grid, n_refine, lmm_mode = 0;
    double l_min, l_max, logl_H0 = 0.0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddii|$iOO", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj, &w_obj, &Uty_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine,
            &lmm_mode, &hi_eval_null_obj, &logl_H0_obj)) {
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

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
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
    if (wants_hi) {
        hi_eval_null_arr = take_vector(hi_eval_null_obj, n_samples, "hi_eval_null");
        if (!hi_eval_null_arr) goto err_input;
    }
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;
    if (wants_hi && validate_hi_eval_null(
            (const double *)PyArray_DATA(hi_eval_null_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }
    ws->mode = lmm_mode;
    if (init_ncvt1_workspace(ws, eigenvalues_arr, uab_inv_arr, w_arr, Uty_arr,
                             n_samples, l_min, l_max, n_grid, n_refine,
                             lmm_mode != 3) < 0)
        goto err_ws;
    if (wants_hi && init_ncvt1_null_hi(
            ws, (const double *)PyArray_DATA(hi_eval_null_arr)) < 0)
        goto err_ws;
    if (wants_logl && set_ncvt1_null_logl(ws, logl_H0) < 0)
        goto err_ws;
    if (lmm_mode == 3 && init_ncvt1_score_vectors(ws) < 0)
        goto err_ws;

    capsule = PyCapsule_New(ws, NCVT1_CAPSULE, lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(w_arr);
    Py_DECREF(Uty_arr);
    Py_XDECREF(hi_eval_null_arr);
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

/* Clamp a caller-supplied thread count to the workspace semantics every
 * ncvt1 compute now shares: at most n_snps, at least 1. Thread count never
 * enters the per-SNP arithmetic (each iteration is independent under
 * schedule(static)), so this choice cannot move a result bit. */
static int clamp_threads(int n_threads, int n_snps)
{
    int actual = n_threads;
    if (actual > n_snps) actual = n_snps;
    if (actual < 1) actual = 1;
    return actual;
}

/* -------------------------------------------------------------------------
 * ncvt1_wald_loop
 *
 * Wald (lmm_mode 1) or Wald + Score + LRT (lmm_mode 4) over one chunk of an
 * n_cvt=1 workspace. Hoisted verbatim from the per-mode entry points this
 * replaces: computes wx/xx/xy on-the-fly from the w/Uty stored in the
 * workspace rather than taking them prebuilt, which does not change the
 * arithmetic or its order. The REML lambda search is the same coarse grid
 * plus golden-section refinement in both modes; mode 4 additionally reads
 * the MLE bracket that same grid pass produces.
 * ------------------------------------------------------------------------- */
static PyObject *ncvt1_wald_loop(
    lmm_workspace_t *ws, const double *utg_t_data, int n_snps,
    int actual_threads, int mode4)
{
    lmm_output_t out = {0};

    if (alloc_lmm_output(&out, (npy_intp)n_snps, ws->mode) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        return NULL;
    }

    int n_samples = ws->n_samples;
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
    double *out_p_scores    = mode4 ? (double *)PyArray_DATA(out.p_scores) : NULL;
    double *out_lambdas_mle = mode4 ? (double *)PyArray_DATA(out.lambdas_mle) : NULL;
    double *out_p_lrts      = mode4 ? (double *)PyArray_DATA(out.p_lrts) : NULL;

    const ncvt1_grid_t *grid = ws->grid;
    int n_grid    = grid->n_grid;
    int n_refine  = grid->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Per-thread scratch buffers:
     * - 3 for wx/xx/xy on-the-fly computation
     * - 1 for MLE golden section refinement (hi_eval_local), mode 4 only */
    double **scratch_wx = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **scratch_xx = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **scratch_xy = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **thread_bufs = mode4
        ? alloc_thread_scratch(actual_threads, (size_t)n_samples)
        : NULL;
    if (!scratch_wx || !scratch_xx || !scratch_xy || (mode4 && !thread_bufs)) {
        free_thread_scratch(scratch_wx, actual_threads);
        free_thread_scratch(scratch_xx, actual_threads);
        free_thread_scratch(scratch_xy, actual_threads);
        free_thread_scratch(thread_bufs, actual_threads);
        decref_lmm_output(&out);
        PyErr_NoMemory();
        return NULL;
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

        /* Compute wx/xx/xy on-the-fly */
        for (int i = 0; i < n_samples; i++) {
            vwx[i] = w_ptr[i] * x[i];
            vxx[i] = x[i] * x[i];
            vxy[i] = x[i] * Uty_ptr[i];
        }

        /* ---- (a) Score: null-model Pab ---- */
        if (mode4) {
            const ncvt1_null_model_t *nm = ws->null_model;
            double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
            #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
            for (int i = 0; i < n_samples; i++) {
                double h = nm->hi_eval_null[i];
                s_wx += h * vwx[i];
                s_xx += h * vxx[i];
                s_xy += h * vxy[i];
            }

            double pab_null[3][6];
            calc_pab_ncvt1_split(nm->s_ww, s_wx, nm->s_wy,
                                  s_xx, s_xy, nm->s_yy, pab_null);

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
            grid->hi_eval_grid, grid->logdet_h_grid, grid->grid_inv, n_grid,
            logdet_iab, df, reml_const, mode4 ? ws->lrt->mle_const : 0.0,
            &best_reml_idx, &best_mle_idx
        );

        /* ---- (c) Wald: REML refinement from the shared coarse grid ---- */
        double logl_reml, wald_beta, wald_se, wald_f;
        int wald_valid;
        double lambda_reml = refine_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, grid->lambda_grid, grid->log_l_min, grid->step,
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
        if (mode4) {
            double *hi_eval_local = thread_bufs[tid];

            double logl_H1;
            double lambda_mle = refine_lambda_mle_ncvt1_split(
                vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
                ws->eigenvalues, n_samples, grid->lambda_grid,
                grid->log_l_min, grid->step, n_grid, n_refine,
                best_mle_idx, ws->lrt->mle_const, hi_eval_local, &logl_H1
            );

            out_lambdas_mle[snp] = lambda_mle;

            double lrt_stat = 2.0 * (logl_H1 - ws->lrt->logl_H0);
            if (lrt_stat < 0.0) lrt_stat = 0.0;
            out_p_lrts[snp] = chi2_sf_c(lrt_stat);
        }
    }

    Py_END_ALLOW_THREADS

    /* Free per-thread scratch buffers */
    free_thread_scratch(scratch_wx, actual_threads);
    free_thread_scratch(scratch_xx, actual_threads);
    free_thread_scratch(scratch_xy, actual_threads);
    free_thread_scratch(thread_bufs, actual_threads);

    if (warn_betainc_convergence(out_betas, out_pwalds, n_snps) < 0) {
        decref_lmm_output(&out);
        return NULL;
    }

    return build_lmm_result_dict(&out);
}

/* -------------------------------------------------------------------------
 * ncvt1_score_loop
 *
 * Score (lmm_mode 3) over one chunk of an n_cvt=1 workspace, hoisted
 * verbatim from the entry point this replaces. Sums (h*w)*x per SNP from
 * the mode-3-only h_null_w/h_null_Uty the workspace precomputed; mode 4's
 * Score block instead sums h*(w*x) from scratch it builds per SNP, and the
 * two associations are not bit-identical, so this body stays mode-3 only.
 * ------------------------------------------------------------------------- */
static PyObject *ncvt1_score_loop(
    lmm_workspace_t *ws, const double *utg_t_data, int n_snps,
    int actual_threads)
{
    int n_samples = ws->n_samples;

    score_output_t out;
    if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    double *out_betas    = (double *)PyArray_DATA(out.betas);
    double *out_ses      = (double *)PyArray_DATA(out.ses);
    double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

    /* Read precomputed invariants from workspace */
    const double *h_null_w   = ws->score->h_null_w;
    const double *h_null_Uty = ws->score->h_null_Uty;
    const double *hi_eval_null = ws->null_model->hi_eval_null;
    double null_s_ww = ws->null_model->s_ww;
    double null_s_wy = ws->null_model->s_wy;
    double null_s_yy = ws->null_model->s_yy;
    int df       = ws->df;
    double a     = ws->beta_a;
    double b_val = ws->beta_b;
    double lbeta_ab = ws->lbeta_ab;

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
        return NULL;
    }

    return build_score_result_dict(&out);
}

/* -------------------------------------------------------------------------
 * ncvt1_lrt_loop
 *
 * LRT (lmm_mode 2) over one chunk of an n_cvt=1 workspace, hoisted verbatim
 * from the entry point this replaces. Per-thread scratch is allocated per
 * call so the thread count can be retuned between chunks.
 * ------------------------------------------------------------------------- */
static PyObject *ncvt1_lrt_loop(
    lmm_workspace_t *ws, const double *utg_t_data, int n_snps,
    int actual_threads)
{
    int n_samples = ws->n_samples;

    lrt_output_t out;
    if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    /* Allocate per-thread scratch buffers (thread-safe, adapts to retuned n_threads) */
    double **thread_bufs = alloc_thread_scratch(actual_threads, (size_t)n_samples);
    double **thread_scratch =
        alloc_thread_scratch(actual_threads, (size_t)3 * n_samples);
    if (!thread_bufs || !thread_scratch) {
        free_thread_scratch(thread_bufs, actual_threads);
        free_thread_scratch(thread_scratch, actual_threads);
        decref_lrt_output(&out);
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
            vwx_local[i] = ws->w[i] * x[i];
            vxx_local[i] = x[i] * x[i];
            vxy_local[i] = ws->Uty[i] * x[i];
        }

        double logl_H1;
        double lam_mle = golden_section_lambda_mle_ncvt1_split(
            vwx_local, vxx_local, vxy_local,
            ws->inv_ww, ws->inv_wy, ws->inv_yy,
            ws->eigenvalues, n_samples,
            ws->grid->lambda_grid, ws->grid->hi_eval_grid, ws->grid->logdet_h_grid,
            ws->grid->grid_inv, ws->grid->log_l_min, ws->grid->step,
            ws->grid->n_grid, ws->grid->n_refine,
            ws->lrt->mle_const, hi_eval_local, &logl_H1
        );
        out_lambdas_mle[s] = lam_mle;

        double lrt_stat = 2.0 * (logl_H1 - ws->lrt->logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[s] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    /* Free per-call scratch */
    free_thread_scratch(thread_bufs, actual_threads);
    free_thread_scratch(thread_scratch, actual_threads);

    return build_lrt_result_dict(&out);
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_ncvt1_c
 *
 * One compute entry point for every n_cvt=1 workspace. Intake, clamp_threads,
 * then dispatch on ws->mode to the loop function that mode built its
 * workspace for. This is a dispatcher selecting a whole loop body, not
 * interleaved gating: each loop function above is one of the three former
 * entry points, hoisted verbatim.
 *
 * Python signature:
 *   compute_lmm_chunk_ncvt1_c(workspace, utg_t, n_threads)
 * Returns:
 *   mode 1: dict with lambdas, logls, betas, ses, pwalds
 *   mode 2: dict with lambdas_mle, p_lrts
 *   mode 3: dict with betas, ses, p_scores
 *   mode 4: mode 1's keys plus p_scores, lambdas_mle, p_lrts
 *   each value (n_snps,) float64.
 * ------------------------------------------------------------------------- */
PyObject *compute_lmm_chunk_ncvt1_c_py(
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

    lmm_workspace_t *ws = (lmm_workspace_t *)
        PyCapsule_GetPointer(capsule_obj, NCVT1_CAPSULE);
    if (!ws) return NULL;

    int n_samples = ws->n_samples;
    int n_snps;
    PyArrayObject *utg_t_arr = take_chunk(utg_t_obj, n_samples, &n_snps);
    if (!utg_t_arr) return NULL;

    int actual_threads = clamp_threads(n_threads, n_snps);
    const double *utg_t_data = (const double *)PyArray_DATA(utg_t_arr);

    PyObject *result;
    switch (ws->mode) {
        case 1:
        case 4:
            result = ncvt1_wald_loop(
                ws, utg_t_data, n_snps, actual_threads, ws->mode == 4);
            break;
        case 2:
            result = ncvt1_lrt_loop(ws, utg_t_data, n_snps, actual_threads);
            break;
        case 3:
            result = ncvt1_score_loop(ws, utg_t_data, n_snps, actual_threads);
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                "compute_lmm_chunk_ncvt1_c: workspace has invalid lmm_mode=%d",
                ws->mode);
            result = NULL;
    }

    Py_DECREF(utg_t_arr);
    return result;
}
