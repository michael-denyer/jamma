/*
 * _lmm_support.h — allocation, validation, and Python marshalling shared by
 * every LMM accelerator translation unit.
 *
 * These are the functions that talk to CPython and the allocator rather than
 * to the numerics: scratch buffers, argument validation, the alloc/decref/build
 * triples for each result shape, and the packed Pab table. Every kernel family
 * needs them, none of them owns any floating-point pipeline, so they are the
 * one shared seam that can move without a numerical argument.
 *
 * NumPy C-API across multiple translation units
 * ---------------------------------------------
 * NumPy reaches its C API through a per-translation-unit `PyArray_API` pointer.
 * With one .c file that pointer is filled in by `import_array()` and everything
 * works. With more than one, another unit's copy stays NULL and the first
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

typedef struct {
    PyArrayObject *lambdas;      /* REML lambda */
    PyArrayObject *logls;
    PyArrayObject *betas;
    PyArrayObject *ses;
    PyArrayObject *pwalds;       /* Wald p-value */
    PyArrayObject *p_scores;
    PyArrayObject *lambdas_mle;
    PyArrayObject *p_lrts;
} lmm_output_t;

/* ---------------------------------------------------------------------------
 * Allocation
 * ------------------------------------------------------------------------- */

/* n doubles, 32-byte aligned (AVX2). NULL on failure or n == 0. */
double *alloc_aligned_doubles(size_t n);
size_t aligned_double_bytes(size_t n);
size_t pab_entry_count(int n_rows);
size_t pab_table_bytes(int n_cvt);

/* One n-double aligned buffer per thread, so the per-SNP loop never calls
 * malloc (heap-lock contention at high thread counts). NULL on any failure,
 * having freed whatever it had already taken. */
double **alloc_thread_scratch(int n_threads, size_t n);

/* Symmetric teardown, NULL-safe so it can run on every cleanup path. */
void free_thread_scratch(double **bufs, int n_threads);

/* ---------------------------------------------------------------------------
 * Validation
 * ------------------------------------------------------------------------- */

/* Reject non-finite eigenvalues, and any that would make l_max * ev + 1
 * non-positive, before the compute loop; without this they propagate silently
 * through the whole REML pipeline and produce garbage with no error.
 * O(n_samples), negligible against O(n*m*k). 0, or -1 with PyErr. */
int validate_eigenvalues(const double *data, int n_samples, double l_max);

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

int alloc_lmm_output(lmm_output_t *out, npy_intp n_snps, lmm_tests_t tests);
void decref_lmm_output(lmm_output_t *out);
PyObject *build_lmm_result_dict(lmm_output_t *out);

PyObject *finish_lmm_output(lmm_output_t *out, lmm_tests_t tests, int n_snps);

/* ---------------------------------------------------------------------------
 * Pab recursion table, built from n_cvt at workspace creation. The typedef
 * lives in _lmm_types.h.
 * ------------------------------------------------------------------------- */

/* Construct the canonical packed table from n_cvt. Free on every failure. */
int build_pab_table(int n_cvt, pab_table_t *t, int n_samples);

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
int validate_logl_H0(double logl_H0);
int validate_hi_eval_null(const double *hi, int n_samples);

int parse_mode_inputs(int lmm_mode, PyObject **hi_obj, PyObject *logl_obj,
                      lmm_tests_t *tests, double *logl_H0);

/* Threads one chunk runs on: at most one per SNP and at most the
 * workspace's thread capacity, and at least one. */
int clamp_threads(int n_threads, int n_snps, int capacity);

#endif /* JAMMA_LMM_SUPPORT_H */
