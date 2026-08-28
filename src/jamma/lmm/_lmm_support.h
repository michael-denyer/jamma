/*
 * _lmm_support.h — allocation, validation, and Python marshalling shared by
 * every kernel family in _lmm_accel.c.
 *
 * These are the functions that talk to CPython and the allocator rather than
 * to the numerics: scratch buffers, argument validation, the alloc/decref/build
 * triples for each result shape, and the pab-table parser. Every kernel family
 * needs them, none of them owns any floating-point pipeline, so they are the
 * one seam in _lmm_accel.c that can move without a numerical argument.
 *
 * NumPy C-API across two translation units
 * ----------------------------------------
 * NumPy reaches its C API through a per-translation-unit `PyArray_API` pointer.
 * With one .c file that pointer is filled in by `import_array()` and everything
 * works. With two, the second unit's copy stays NULL and the first
 * `PyArray_SimpleNew` segfaults. `PY_ARRAY_UNIQUE_SYMBOL` makes the pointer a
 * single shared extern instead, so:
 *
 *   - _lmm_accel.c includes this header directly. It owns `import_array()`.
 *   - every other unit defines NO_IMPORT_ARRAY before including this header.
 *
 * A new .c file that forgets NO_IMPORT_ARRAY fails to link rather than
 * crashing at runtime, which is the failure mode we want.
 */

#ifndef JAMMA_LMM_SUPPORT_H
#define JAMMA_LMM_SUPPORT_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL JAMMA_LMM_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>

/* Pab bounds and the table layout live in _lmm_types.h: the numerical
 * kernels read them and must not need <Python.h> to do it. */
#include "_lmm_types.h"

/* ---------------------------------------------------------------------------
 * Result shapes. One struct per set of output arrays a kernel family returns,
 * each with an alloc / decref / build triple below.
 * ------------------------------------------------------------------------- */

/* Wald: 5 arrays. */
typedef struct {
    PyArrayObject *lambdas;
    PyArrayObject *logls;
    PyArrayObject *betas;
    PyArrayObject *ses;
    PyArrayObject *pwalds;
} output_arrays_t;

/* Score: 3 arrays. */
typedef struct {
    PyArrayObject *betas;
    PyArrayObject *ses;
    PyArrayObject *p_scores;
} score_output_t;

/* LRT: 2 arrays. */
typedef struct {
    PyArrayObject *lambdas_mle;
    PyArrayObject *p_lrts;
} lrt_output_t;

/* Mode 4: Wald + Score + LRT in one pass, 8 arrays. */
typedef struct {
    PyArrayObject *lambdas;      /* REML lambda */
    PyArrayObject *logls;        /* REML log-likelihood */
    PyArrayObject *betas;        /* Wald beta (REML-optimized) */
    PyArrayObject *ses;          /* Wald SE (REML-optimized) */
    PyArrayObject *pwalds;       /* Wald p-value */
    PyArrayObject *p_scores;     /* Score p-value */
    PyArrayObject *lambdas_mle;  /* MLE lambda */
    PyArrayObject *p_lrts;       /* LRT p-value */
} mode4_output_t;

/* ---------------------------------------------------------------------------
 * Pab recursion table, parsed from the dict build_pab_table_for_c() returns.
 * ------------------------------------------------------------------------- */


/* ---------------------------------------------------------------------------
 * Allocation
 * ------------------------------------------------------------------------- */

/* n doubles, 32-byte aligned (AVX2). NULL on failure or n == 0. */
double *alloc_aligned_doubles(size_t n);

/* One n-double aligned buffer per thread, so the per-SNP loop never calls
 * malloc (heap-lock contention at high thread counts). NULL on any failure,
 * having freed whatever it had already taken. */
double **alloc_thread_scratch(int n_threads, size_t n);

/* Symmetric teardown, NULL-safe so it can run on every cleanup path. */
void free_thread_scratch(double **bufs, int n_threads);

/* ---------------------------------------------------------------------------
 * Validation
 * ------------------------------------------------------------------------- */

/* Reject non-finite eigenvalues before the compute loop; without this they
 * propagate silently through the whole REML pipeline and produce garbage with
 * no error. O(n_samples), negligible against O(n*m*k). 0, or -1 with PyErr. */
int validate_eigenvalues(const double *data, int n_samples);

/* 0, or -1 with PyErr set. */
int validate_batch_params(int n_samples, double l_min, double l_max,
                          int n_grid, int n_refine);

/* Post-compute scan for NaN p-values where beta is finite, i.e. the stats came
 * out fine but the betainc continued fraction did not converge. Call after
 * Py_END_ALLOW_THREADS. Returns -1 if the warning was promoted to an exception
 * (simplefilter("error")). */
int warn_betainc_convergence(const double *betas, const double *pvalues,
                             int n_snps);

/* ---------------------------------------------------------------------------
 * Result marshalling. Each build_* consumes the caller's references: on
 * success the dict holds its own, on failure everything is released.
 * ------------------------------------------------------------------------- */

int alloc_output_arrays(output_arrays_t *out, npy_intp n_snps);
void decref_output_arrays(output_arrays_t *out);
PyObject *build_result_dict(output_arrays_t *out);

int alloc_score_output(score_output_t *out, npy_intp n_snps);
void decref_score_output(score_output_t *out);
PyObject *build_score_result_dict(score_output_t *out);

int alloc_lrt_output(lrt_output_t *out, npy_intp n_snps);
void decref_lrt_output(lrt_output_t *out);
PyObject *build_lrt_result_dict(lrt_output_t *out);

int alloc_mode4_output(mode4_output_t *out, npy_intp n_snps);
void decref_mode4_output(mode4_output_t *out);
PyObject *build_mode4_result_dict(mode4_output_t *out);

/* ---------------------------------------------------------------------------
 * Argument parsing
 * ------------------------------------------------------------------------- */

/* Malloc'd copy of a length-checked int32 array; caller frees. */
int *parse_int32_array(PyObject *obj, int expected_len, const char *name);

/* Parse and fully validate the dict PabCTable._asdict() produces: every
 * index in range, the level table consistent with entries. 0 on success, -1
 * with PyErr set. On success the caller must free_pab_table. On failure
 * everything already taken is released and the struct is zeroed. */
int parse_pab_table_from_dict(PyObject *dict, pab_table_t *t, int n_samples);

/* Release the owned fields and zero the struct, so a second call is a no-op.
 * Does NOT free the struct itself. */
void free_pab_table(pab_table_t *t);

/* ---------------------------------------------------------------------------
 * Array intake. Each returns a new reference to a C-contiguous, aligned
 * float64 view of obj, or NULL with PyErr set.
 * ------------------------------------------------------------------------- */

/* Any shape; the caller checks dims it can only know later. */
PyArrayObject *take_array(PyObject *obj);

/* Shape (n,). */
PyArrayObject *take_vector(PyObject *obj, int n, const char *name);

/* Shape (rows, cols). */
PyArrayObject *take_matrix(PyObject *obj, int rows, int cols,
                           const char *name);

/* A genotype chunk utg_t, shape (n_snps, n_samples) with n_snps <= INT_MAX.
 * Writes n_snps to *n_snps_out. Whether n_snps == 0 is an error is the
 * caller's decision: the n_cvt=1 Wald paths accept it. */
PyArrayObject *take_chunk(PyObject *obj, int n_samples, int *n_snps_out);

/* Each 0 on success, -1 with PyErr set. */
int validate_n_cvt(int n_cvt);
int validate_logl_H0(double logl_H0);
int validate_hi_eval_null(const double *hi, int n_samples);

/* ---------------------------------------------------------------------------
 * n_cvt = 1 lambda grid. Fills the caller-allocated lambda_grid (n_grid,),
 * hi_eval_grid (n_grid * n_samples), logdet_h_grid (n_grid,) and the per-grid
 * invariant dot products grid_inv (n_grid,).
 * ------------------------------------------------------------------------- */
void build_grid_ncvt1(int n_grid, int n_samples, double log_l_min, double step,
                      const double *eigenvalues, const double *inv_ww,
                      const double *inv_wy, const double *inv_yy,
                      double *lambda_grid, double *hi_eval_grid,
                      double *logdet_h_grid, grid_invariant_t *grid_inv);

#endif /* JAMMA_LMM_SUPPORT_H */
