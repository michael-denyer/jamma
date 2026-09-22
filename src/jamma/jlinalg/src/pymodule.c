/**
 * pymodule.c -- Python C extension module _jlinalg.
 *
 * Exposes vendor-dispatch BLAS/LAPACK operations (dgemm, dsyrk, eigh),
 * SNP statistics, and introspection functions to Python via the NumPy buffer
 * protocol.
 *
 * Module-level constants:
 *   jlinalg_isa   -- compile-time ISA string ("AVX2", "NEON", or "generic")
 *   HAS_OPENMP    -- True if compiled with OpenMP (-fopenmp)
 *   ABI_VERSION   -- integer (JLINALG_ABI_VERSION from jlinalg.h)
 *
 * Exported functions: dgemm, dsyrk, eigh, compute_snp_stats_chunk,
 *                     set_n_threads, get_n_threads
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include "jlinalg.h"

/* numpy.linalg.LinAlgError -- cached at module init for eigh convergence errors.
 * Falls back to PyExc_RuntimeError if numpy.linalg cannot be imported. */
static PyObject *LinAlgError = NULL;

/* ---------------------------------------------------------------------------
 * py_dgemm -- matrix-matrix product C = op(A) @ op(B)
 *
 * Signature: dgemm(A, B, transa='N', transb='N', out=None) -> ndarray
 * A and B are 2-D float64 arrays.  transa/transb: 'N' (no transpose) or
 * 'T' (transpose).  out: optional preallocated output array (M x N, float64,
 * C-contiguous).  If None, a new array is allocated.  Returns the output array.
 *
 * jamma.jlinalg.dgemm validates the full public contract (transa/transb
 * values, dimension match, out shape) once in Python before calling this
 * entry point on either backend, so the semantic checks live there. This
 * function keeps only what memory safety needs: a wired vendor dgemm
 * (jlinalg_dgemm_ext aborts without one), dtype, contiguity,
 * alignment, and writeability of out, none of which Python re-derives from
 * flags a caller could still get wrong when calling this entry point
 * directly (e.g. from C tests).
 * ---------------------------------------------------------------------------
 */
static PyObject *py_dgemm(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"A", "B", "transa", "transb", "out", NULL};
    PyObject *oA, *oB;
    PyObject *oOut = Py_None;
    const char *transa_str = "N";
    const char *transb_str = "N";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ssO", kwlist, &oA, &oB, &transa_str,
                                     &transb_str, &oOut))
        return NULL;

    if (!blas_has_external()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "dgemm: vendor dgemm is not wired (blas_has_dgemm == 0); "
                        "jamma.jlinalg.dgemm falls back to NumPy in this state");
        return NULL;
    }

    int transa = (transa_str[0] == 'T' || transa_str[0] == 't') ? 1 : 0;
    int transb = (transb_str[0] == 'T' || transb_str[0] == 't') ? 1 : 0;

    PyArrayObject *aA = NULL, *aB = NULL, *aC = NULL;
    PyObject *result = NULL;

    /* Coerce to C-contiguous float64. */
    aA = (PyArrayObject *)PyArray_FROM_OTF(oA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aA) goto cleanup;
    aB = (PyArrayObject *)PyArray_FROM_OTF(oB, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aB) goto cleanup;

    if (PyArray_NDIM(aA) != 2 || PyArray_NDIM(aB) != 2) {
        PyErr_SetString(PyExc_ValueError, "dgemm: A and B must be 2-D arrays");
        goto cleanup;
    }

    /* Effective dimensions after transpose */
    npy_intp M = transa ? PyArray_DIM(aA, 1) : PyArray_DIM(aA, 0);
    npy_intp K_a = transa ? PyArray_DIM(aA, 0) : PyArray_DIM(aA, 1);
    npy_intp K_b = transb ? PyArray_DIM(aB, 1) : PyArray_DIM(aB, 0);
    npy_intp N = transb ? PyArray_DIM(aB, 0) : PyArray_DIM(aB, 1);

    if (K_a != K_b) {
        PyErr_Format(PyExc_ValueError,
                     "dgemm: inner dimensions mismatch: op(A) is %ldx%ld, op(B) is %ldx%ld",
                     (long)M, (long)K_a, (long)K_b, (long)N);
        goto cleanup;
    }

    /* Output C (M x N): use caller-provided buffer or allocate fresh */
    if (oOut != Py_None) {
        if (!PyArray_Check(oOut)) {
            PyErr_SetString(PyExc_TypeError, "dgemm: out must be a numpy array");
            goto cleanup;
        }
        PyArrayObject *out = (PyArrayObject *)oOut;
        if (PyArray_TYPE(out) != NPY_DOUBLE) {
            PyErr_Format(PyExc_ValueError, "dgemm: out must be float64, got dtype %d",
                         PyArray_TYPE(out));
            goto cleanup;
        }
        if (!PyArray_IS_C_CONTIGUOUS(out) || !PyArray_ISWRITEABLE(out)) {
            PyErr_SetString(PyExc_ValueError, "dgemm: out must be C-contiguous and writeable");
            goto cleanup;
        }
        if (!PyArray_ISALIGNED(out)) {
            PyErr_SetString(PyExc_ValueError, "dgemm: out must be aligned");
            goto cleanup;
        }
        if (PyArray_NDIM(out) != 2 || PyArray_DIM(out, 0) != M || PyArray_DIM(out, 1) != N) {
            PyErr_Format(PyExc_ValueError,
                         "dgemm: out shape (%zd, %zd) doesn't match result shape (%zd, %zd)",
                         (Py_ssize_t)PyArray_DIM(out, 0), (Py_ssize_t)PyArray_DIM(out, 1),
                         (Py_ssize_t)M, (Py_ssize_t)N);
            goto cleanup;
        }
        Py_INCREF(oOut);
        aC = out;
    } else {
        npy_intp dims[2] = {M, N};
        aC = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!aC) goto cleanup;
    }

    const double *pA = (const double *)PyArray_DATA(aA);
    const double *pB = (const double *)PyArray_DATA(aB);
    double *pC = (double *)PyArray_DATA(aC);

    /* Leading dimensions are the physical column counts (row-major storage) */
    npy_intp lda = PyArray_DIM(aA, 1);
    npy_intp ldb = PyArray_DIM(aB, 1);

    Py_BEGIN_ALLOW_THREADS jlinalg_dgemm_ext(M, N, K_a, pA, lda, pB, ldb, pC, N, transa, transb);
    Py_END_ALLOW_THREADS

        result = (PyObject *)aC;

cleanup:
    Py_XDECREF(aA);
    Py_XDECREF(aB);
    return result;
}

/* ---------------------------------------------------------------------------
 * py_dsyrk -- symmetric rank-k update K = X @ X.T
 *
 * Signature: dsyrk(X: ndarray, *, out=None, beta=0.0) -> ndarray
 * X must be 2-D C-contiguous float64 of shape (N, K).
 * out, when provided, must be writable, aligned, C-contiguous float64 of shape (N, N).
 * Computes out = X @ X.T + beta*out and returns the output, bitwise symmetric.
 *
 * jamma.jlinalg.dsyrk validates the full public contract (beta requires out,
 * out shape) once in Python before calling this entry point on either
 * backend, so the semantic checks live there. This function keeps only what
 * memory safety needs: a wired vendor dsyrk (jlinalg_dsyrk_ext aborts
 * without one), dtype, contiguity, alignment, and writeability of
 * out, none of which Python re-derives from flags a caller could still get
 * wrong when calling this entry point directly (e.g. from C tests).
 * ---------------------------------------------------------------------------
 */
static PyObject *py_dsyrk(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"X", "out", "beta", NULL};
    PyObject *oX;
    PyObject *oOut = Py_None;
    double beta = 0.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$Od", kwlist, &oX, &oOut, &beta)) return NULL;

    if (!blas_has_dsyrk()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "dsyrk: vendor dsyrk is not wired (blas_has_dsyrk == 0); "
                        "jamma.jlinalg.dsyrk falls back to NumPy in this state");
        return NULL;
    }

    PyArrayObject *aC = NULL;
    PyObject *result = NULL;
    PyArrayObject *aX = (PyArrayObject *)PyArray_FROM_OTF(oX, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aX) goto cleanup;

    if (PyArray_NDIM(aX) != 2) {
        PyErr_SetString(PyExc_ValueError, "dsyrk: X must be a 2-D array");
        goto cleanup;
    }

    npy_intp N = PyArray_DIM(aX, 0);
    npy_intp K = PyArray_DIM(aX, 1);

    if (oOut != Py_None) {
        if (!PyArray_Check(oOut)) {
            PyErr_SetString(PyExc_TypeError, "dsyrk: out must be a numpy array");
            goto cleanup;
        }
        PyArrayObject *out = (PyArrayObject *)oOut;
        if (PyArray_TYPE(out) != NPY_DOUBLE) {
            PyErr_SetString(PyExc_ValueError, "dsyrk: out must be float64");
            goto cleanup;
        }
        if (!PyArray_IS_C_CONTIGUOUS(out)) {
            PyErr_SetString(PyExc_ValueError, "dsyrk: out must be C-contiguous");
            goto cleanup;
        }
        if (!PyArray_ISALIGNED(out)) {
            PyErr_SetString(PyExc_ValueError, "dsyrk: out must be aligned");
            goto cleanup;
        }
        if (!PyArray_ISWRITEABLE(out)) {
            PyErr_SetString(PyExc_ValueError, "dsyrk: out must be writeable");
            goto cleanup;
        }
        if (PyArray_NDIM(out) != 2 || PyArray_DIM(out, 0) != N || PyArray_DIM(out, 1) != N) {
            PyErr_Format(PyExc_ValueError, "dsyrk: out shape doesn't match result shape (%zd, %zd)",
                         (Py_ssize_t)N, (Py_ssize_t)N);
            goto cleanup;
        }
        Py_INCREF(oOut);
        aC = out;
    } else {
        npy_intp dims[2] = {N, N};
        aC = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!aC) goto cleanup;
    }

    const double *pX = (const double *)PyArray_DATA(aX);
    double *pC = (double *)PyArray_DATA(aC);

    Py_BEGIN_ALLOW_THREADS jlinalg_dsyrk_ext(N, K, pX, K, pC, N, beta);
    Py_END_ALLOW_THREADS

        result = (PyObject *)aC;

cleanup:
    Py_XDECREF(aX);
    return result;
}

/* ---------------------------------------------------------------------------
 * py_eigh -- compute eigenvalues and eigenvectors of symmetric matrix
 *
 * Signature: eigh(K: ndarray, inplace: bool = False, driver: str = "auto")
 *   -> tuple[ndarray, ndarray, int]
 * K must be 2-D C-contiguous float64 of shape (N, N).
 * driver: "auto" (DSYEVD, falling through to DSYEVR on alloc failure),
 *         "dsyevd" (same as "auto" -- DSYEVD is always tried first when not
 *         skipped), or "dsyevr" (skip the DSYEVD attempt and require DSYEVR).
 *
 * When inplace=False (default): K is used as scratch; a fresh N*N eigenvector
 * array is allocated and returned.  Backward compatible with existing callers.
 *
 * When inplace=True: K is overwritten in-place with eigenvectors.  No separate
 * N*N allocation is made (only the N eigenvalues).  The returned eigenvector
 * array IS K.  This saves N^2*8 bytes at 125k scale (~125 GB).
 *
 * The third return value is the driver that actually ran (1 = DSYEVD,
 * 2 = DSYEVR, 0 = neither, e.g. N == 1). jamma.jlinalg.eigh wraps it in an
 * EighStatus with a string driver_used for callers.
 * ---------------------------------------------------------------------------
 */
static PyObject *py_eigh(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *oK;
    int inplace = 0;
    const char *driver_str = "auto";
    static char *kwlist[] = {"K", "inplace", "driver", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p$s", kwlist, &oK, &inplace, &driver_str))
        return NULL;

    int require_dsyevr;
    if (strcmp(driver_str, "auto") == 0 || strcmp(driver_str, "dsyevd") == 0) {
        require_dsyevr = 0;
    } else if (strcmp(driver_str, "dsyevr") == 0) {
        require_dsyevr = 1;
    } else {
        PyErr_Format(PyExc_ValueError,
                     "eigh: driver must be 'auto', 'dsyevd', or 'dsyevr', got '%s'", driver_str);
        return NULL;
    }

    PyArrayObject *aK = NULL, *aW = NULL, *aU = NULL;
    PyObject *result = NULL;

    aK = (PyArrayObject *)PyArray_FROM_OTF(oK, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (!aK) goto cleanup;

    if (PyArray_NDIM(aK) != 2 || PyArray_DIM(aK, 0) != PyArray_DIM(aK, 1)) {
        PyErr_SetString(PyExc_ValueError, "eigh: K must be 2-D square float64");
        goto cleanup;
    }

    npy_intp N = PyArray_DIM(aK, 0);
    double *pK = (double *)PyArray_DATA(aK);

    /* Allocate eigenvalues (N,) -- always needed */
    aW = (PyArrayObject *)PyArray_SimpleNew(1, &N, NPY_DOUBLE);
    if (!aW) goto cleanup;

    /* Reject inplace when FROM_OTF created a temporary copy */
    if (inplace && (PyArray_FLAGS(aK) & NPY_ARRAY_WRITEBACKIFCOPY)) {
        PyErr_SetString(PyExc_ValueError,
                        "eigh: inplace=True requires a C-contiguous, writeable, float64 array. "
                        "The input was converted to a temporary copy.");
        goto cleanup;
    }

    /* Eigenvector buffer: when inplace=True, reuse K directly (no N*N alloc). */
    double *pU;
    if (inplace) {
        pU = pK; /* K and eigenvectors share the same buffer */
    } else {
        npy_intp dims2[2] = {N, N};
        aU = (PyArrayObject *)PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
        if (!aU) goto cleanup;
        pU = (double *)PyArray_DATA(aU);
    }

    double *pW = (double *)PyArray_DATA(aW);

    /* Initialize status struct */
    jlinalg_eigh_status_t eigh_status;
    memset(&eigh_status, 0, sizeof(eigh_status));

    int ret;
    Py_BEGIN_ALLOW_THREADS ret = jlinalg_eigh_c(N, pK, N, pW, pU, N, require_dsyevr, &eigh_status);
    Py_END_ALLOW_THREADS

        if (ret != 0) {
        if (ret == JLINALG_EXT_UNAVAILABLE) {
            PyErr_Format(PyExc_RuntimeError,
                         require_dsyevr
                             ? "jlinalg eigh: driver='dsyevr' requested but vendor DSYEVR is "
                               "not available. Use numpy.linalg.eigh instead."
                             : "jlinalg eigh: no vendor LAPACK available "
                               "(DSYEVD and DSYEVR both unavailable). "
                               "Use numpy.linalg.eigh instead.");
        } else if (ret == JLINALG_EXT_ALLOC_FAIL) {
            PyErr_Format(PyExc_MemoryError, "jlinalg eigh: workspace allocation failed -- "
                                            "matrix too large for available memory");
        } else if (ret == JLINALG_EXT_COUNT_MISMATCH) {
            PyErr_Format(PyExc_RuntimeError,
                         "jlinalg eigh: vendor LAPACK DSYEVR returned fewer eigenvalues "
                         "than expected -- this indicates an ABI mismatch or vendor bug");
        } else if (ret == JLINALG_EXT_BAD_STRIDE) {
            PyErr_Format(PyExc_RuntimeError,
                         "jlinalg eigh: internal error -- padded stride passed to "
                         "jlinalg_eigh_c, this is a jlinalg bug, please report it");
        } else if (ret < 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "jlinalg eigh: illegal argument to vendor LAPACK (info=%d) -- "
                         "this is a jlinalg bug, please report it",
                         ret);
        } else {
            /* Convergence failure -- raise numpy.linalg.LinAlgError */
            PyErr_Format(LinAlgError, "jlinalg eigh: convergence failure (returned %d)", ret);
        }
        goto cleanup;
    }

    /* The performance-fallback warning is non-fatal. When a warnings filter
     * turns it into an error, an inplace call has already overwritten K, so
     * it returns the result rather than raise over a clobbered input. */
    if (eigh_status.vendor_lapack_skipped &&
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "jlinalg eigh: vendor LAPACK work buffer allocation failed -- "
                     "eigendecomposition may have used a slower path. "
                     "Free memory or reduce matrix size.",
                     1) < 0) {
        if (!inplace) goto cleanup;
        PyErr_Clear();
    }

    PyArray_ResolveWritebackIfCopy(aK);
    result = Py_BuildValue("(OOi)", aW, inplace ? (PyObject *)aK : (PyObject *)aU,
                           eigh_status.driver_used);

cleanup:
    if (!result && aK) PyArray_DiscardWritebackIfCopy(aK);
    Py_XDECREF(aK);
    Py_XDECREF(aW);
    Py_XDECREF(aU);
    return result;
}

/* ---------------------------------------------------------------------------
 * py_set_n_threads -- Set jlinalg thread count.
 *
 * Signature: set_n_threads(n: int) -> int
 * Returns the previous thread count.
 * Raises ValueError if n < 1.
 * ---------------------------------------------------------------------------
 */
static PyObject *py_set_n_threads(PyObject *self, PyObject *args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    int old = jlinalg_set_n_threads(n);
    if (old < 0) {
        PyErr_SetString(PyExc_ValueError, "set_n_threads: n must be >= 1");
        return NULL;
    }
    return PyLong_FromLong(old);
}

/* ---------------------------------------------------------------------------
 * py_get_n_threads -- Get current jlinalg thread count.
 *
 * Signature: get_n_threads() -> int
 * ---------------------------------------------------------------------------
 */
static PyObject *py_get_n_threads(PyObject *self, PyObject *args) {
    (void)args; /* unused */
    return PyLong_FromLong(jlinalg_get_n_threads());
}

/* ---------------------------------------------------------------------------
 * py_compute_snp_stats_chunk -- Single-pass per-SNP statistics.
 *
 * Signature: compute_snp_stats_chunk(data, means, miss_counts, vars
 *                                    [, n_aa, n_ab, n_bb])
 * ---------------------------------------------------------------------------
 */
static PyObject *py_compute_snp_stats_chunk(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *o_data, *o_means, *o_miss, *o_vars;
    PyObject *o_naa = Py_None, *o_nab = Py_None, *o_nbb = Py_None;

    if (!PyArg_ParseTuple(args, "OOOO|OOO", &o_data, &o_means, &o_miss, &o_vars, &o_naa, &o_nab,
                          &o_nbb))
        return NULL;

    /* Every owned reference is declared and NULL-initialised before the first
     * goto so the single cleanup label can release them unconditionally.
     * a_data is aligned, native-endian and contiguous (no writeback);
     * a_means/miss/vars and the
     * three HWE arrays are INOUT (writeback resolved on success, discarded on
     * error). The `ok` flag selects resolve-vs-discard at the label. */
    PyArrayObject *a_data = NULL, *a_means = NULL, *a_miss = NULL, *a_vars = NULL;
    PyArrayObject *a_naa = NULL, *a_nab = NULL, *a_nbb = NULL;
    PyObject *result = NULL;
    int ok = 0;

    /* Preserve either contiguous layout. Normalize alignment and byte order
     * before passing typed pointers to the kernel. */
    a_data = (PyArrayObject *)PyArray_CheckFromAny(
        o_data, NULL, 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY, NULL);
    if (!a_data) goto cleanup;

    int dtype = PyArray_TYPE(a_data);
    if (dtype != NPY_FLOAT32 && dtype != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError,
                        "compute_snp_stats_chunk: data must be float32 or float64");
        goto cleanup;
    }
    if (PyArray_NDIM(a_data) != 2) {
        PyErr_SetString(PyExc_ValueError, "compute_snp_stats_chunk: data must be 2-D");
        goto cleanup;
    }

    if (!PyArray_ISONESEGMENT(a_data)) {
        PyArrayObject *contiguous = (PyArrayObject *)PyArray_NewCopy(a_data, NPY_KEEPORDER);
        if (!contiguous) goto cleanup;
        Py_SETREF(a_data, contiguous);
    }

    npy_intp n_samples = PyArray_DIM(a_data, 0);
    npy_intp n_snps = PyArray_DIM(a_data, 1);
    int is_fortran = PyArray_IS_F_CONTIGUOUS(a_data);

    /* Validate HWE args: all-None or all-array, not a mix */
    int naa_none = (o_naa == Py_None);
    int nab_none = (o_nab == Py_None);
    int nbb_none = (o_nbb == Py_None);
    int n_hwe_none = naa_none + nab_none + nbb_none;
    if (n_hwe_none != 0 && n_hwe_none != 3) {
        PyErr_SetString(PyExc_ValueError,
                        "compute_snp_stats_chunk: n_aa, n_ab, n_bb must all be arrays "
                        "or all None");
        goto cleanup;
    }

    /* Extract output arrays with INOUT for writeable access */
    a_means = (PyArrayObject *)PyArray_FROM_OTF(o_means, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    a_miss = (PyArrayObject *)PyArray_FROM_OTF(o_miss, NPY_INTP, NPY_ARRAY_INOUT_ARRAY2);
    a_vars = (PyArrayObject *)PyArray_FROM_OTF(o_vars, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!a_means || !a_miss || !a_vars) goto cleanup;

    /* Validate output array sizes match data columns */
    if (PyArray_SIZE(a_means) < n_snps || PyArray_SIZE(a_miss) < n_snps ||
        PyArray_SIZE(a_vars) < n_snps) {
        PyErr_Format(PyExc_ValueError,
                     "compute_snp_stats_chunk: output arrays must have at least %zd "
                     "elements (data has %zd columns), got means=%zd, miss=%zd, "
                     "vars=%zd",
                     (Py_ssize_t)n_snps, (Py_ssize_t)n_snps, (Py_ssize_t)PyArray_SIZE(a_means),
                     (Py_ssize_t)PyArray_SIZE(a_miss), (Py_ssize_t)PyArray_SIZE(a_vars));
        goto cleanup;
    }

    /* HWE arrays (optional -- None means no HWE) */
    int compute_hwe = 0;
    int64_t *naa_ptr = NULL, *nab_ptr = NULL, *nbb_ptr = NULL;
    if (n_hwe_none == 0) {
        compute_hwe = 1;
        a_naa = (PyArrayObject *)PyArray_FROM_OTF(o_naa, NPY_INT64, NPY_ARRAY_INOUT_ARRAY2);
        a_nab = (PyArrayObject *)PyArray_FROM_OTF(o_nab, NPY_INT64, NPY_ARRAY_INOUT_ARRAY2);
        a_nbb = (PyArrayObject *)PyArray_FROM_OTF(o_nbb, NPY_INT64, NPY_ARRAY_INOUT_ARRAY2);
        if (!a_naa || !a_nab || !a_nbb) goto cleanup;
        if (PyArray_SIZE(a_naa) < n_snps || PyArray_SIZE(a_nab) < n_snps ||
            PyArray_SIZE(a_nbb) < n_snps) {
            PyErr_Format(PyExc_ValueError,
                         "compute_snp_stats_chunk: HWE arrays must have at least %zd "
                         "elements, got n_aa=%zd, n_ab=%zd, n_bb=%zd",
                         (Py_ssize_t)n_snps, (Py_ssize_t)PyArray_SIZE(a_naa),
                         (Py_ssize_t)PyArray_SIZE(a_nab), (Py_ssize_t)PyArray_SIZE(a_nbb));
            goto cleanup;
        }
        naa_ptr = (int64_t *)PyArray_DATA(a_naa);
        nab_ptr = (int64_t *)PyArray_DATA(a_nab);
        nbb_ptr = (int64_t *)PyArray_DATA(a_nbb);
    }

    /* Dispatch based on dtype -- release GIL for the C kernel */
    Py_BEGIN_ALLOW_THREADS if (dtype == NPY_FLOAT32) {
        snp_stats_chunk_f32((const float *)PyArray_DATA(a_data), n_samples, n_snps,
                            (double *)PyArray_DATA(a_means), (npy_intp *)PyArray_DATA(a_miss),
                            (double *)PyArray_DATA(a_vars), naa_ptr, nab_ptr, nbb_ptr, compute_hwe,
                            is_fortran);
    }
    else {
        snp_stats_chunk_f64((const double *)PyArray_DATA(a_data), n_samples, n_snps,
                            (double *)PyArray_DATA(a_means), (npy_intp *)PyArray_DATA(a_miss),
                            (double *)PyArray_DATA(a_vars), naa_ptr, nab_ptr, nbb_ptr, compute_hwe,
                            is_fortran);
    }
    Py_END_ALLOW_THREADS

        /* Success: commit writeback for every INOUT array, then return None. */
        PyArray_ResolveWritebackIfCopy(a_means);
    PyArray_ResolveWritebackIfCopy(a_miss);
    PyArray_ResolveWritebackIfCopy(a_vars);
    if (a_naa) PyArray_ResolveWritebackIfCopy(a_naa);
    if (a_nab) PyArray_ResolveWritebackIfCopy(a_nab);
    if (a_nbb) PyArray_ResolveWritebackIfCopy(a_nbb);
    ok = 1;
    Py_INCREF(Py_None);
    result = Py_None;

cleanup:
    /* On error, discard uncommitted writeback copies for the INOUT arrays so
     * numpy does not copy scratch back into the caller's buffers.  a_data has
     * no writeback.  All decrefs are NULL-safe. */
    if (!ok) {
        if (a_means) PyArray_DiscardWritebackIfCopy(a_means);
        if (a_miss) PyArray_DiscardWritebackIfCopy(a_miss);
        if (a_vars) PyArray_DiscardWritebackIfCopy(a_vars);
        if (a_naa) PyArray_DiscardWritebackIfCopy(a_naa);
        if (a_nab) PyArray_DiscardWritebackIfCopy(a_nab);
        if (a_nbb) PyArray_DiscardWritebackIfCopy(a_nbb);
    }
    Py_XDECREF(a_data);
    Py_XDECREF(a_means);
    Py_XDECREF(a_miss);
    Py_XDECREF(a_vars);
    Py_XDECREF(a_naa);
    Py_XDECREF(a_nab);
    Py_XDECREF(a_nbb);
    return result;
}

/* ---------------------------------------------------------------------------
 * Method table
 * ---------------------------------------------------------------------------
 */
static PyMethodDef JlinalgMethods[] = {
    {"dgemm", (PyCFunction)py_dgemm, METH_VARARGS | METH_KEYWORDS,
     "dgemm(A, B, transa='N', transb='N') -> ndarray\n"
     "Matrix-matrix product C = op(A) @ op(B) via vendor BLAS."},
    {"dsyrk", (PyCFunction)py_dsyrk, METH_VARARGS | METH_KEYWORDS,
     "dsyrk(X, *, out=None, beta=0.0) -> ndarray\n"
     "Symmetric rank-k update: K = X @ X.T + beta*K via vendor BLAS."},
    {"eigh", (PyCFunction)py_eigh, METH_VARARGS | METH_KEYWORDS,
     "eigh(K, inplace=False, driver='auto') -> (eigenvalues, eigenvectors, driver_used)\n"
     "Compute all eigenvalues and eigenvectors of symmetric K.\n"
     "When inplace=False (default), K is scratch and a fresh eigenvector\n"
     "array is returned.  When inplace=True, K is overwritten with\n"
     "eigenvectors in-place (no separate N*N allocation).\n"
     "driver: 'auto' (DSYEVD, falling through to DSYEVR), or 'dsyevr' to\n"
     "require DSYEVR directly. driver_used reports which one ran (1 or 2)."},
    {"set_n_threads", py_set_n_threads, METH_VARARGS,
     "set_n_threads(n) -> int\n"
     "Set jlinalg thread count. Returns old count."},
    {"get_n_threads", py_get_n_threads, METH_NOARGS,
     "get_n_threads() -> int\n"
     "Get current jlinalg thread count."},
    {"compute_snp_stats_chunk", py_compute_snp_stats_chunk, METH_VARARGS,
     "compute_snp_stats_chunk(data, means, miss_counts, vars[, n_aa, n_ab, n_bb])\n"
     "Single-pass per-SNP statistics into pre-allocated output arrays."},
    {NULL, NULL, 0, NULL}};

/* ---------------------------------------------------------------------------
 * Module definition
 * ---------------------------------------------------------------------------
 */
static struct PyModuleDef jlinalgmodule = {PyModuleDef_HEAD_INIT, "_jlinalg", /* module name */
                                           NULL,                              /* module docstring */
                                           -1, /* global state, no sub-interpreter support */
                                           JlinalgMethods};

/* ---------------------------------------------------------------------------
 * PyInit__jlinalg -- module initialiser
 * ---------------------------------------------------------------------------
 */
PyMODINIT_FUNC PyInit__jlinalg(void) {
    import_array();

    /* Cache numpy.linalg.LinAlgError for eigh convergence errors. */
    {
        PyObject *linalg_mod = PyImport_ImportModule("numpy.linalg");
        if (linalg_mod) {
            LinAlgError = PyObject_GetAttrString(linalg_mod, "LinAlgError");
            Py_DECREF(linalg_mod);
        }
        if (!LinAlgError) {
            PyErr_Clear();
            LinAlgError = PyExc_RuntimeError;
        }
    }

    if (jlinalg_init() != 0) {
        PyErr_SetString(PyExc_ImportError,
                        "_jlinalg: initialisation failed (vendor BLAS dispatch -- try "
                        "reducing OMP_NUM_THREADS if memory-constrained)");
        return NULL;
    }

    PyObject *m = PyModule_Create(&jlinalgmodule);
    if (!m) return NULL;

#ifdef _OPENMP
    PyObject *has_openmp = Py_True;
#else
    PyObject *has_openmp = Py_False;
#endif
    if (PyModule_AddStringConstant(m, "jlinalg_isa", jlinalg_isa_name()) < 0 ||
        PyModule_AddStringConstant(m, "blas_backend", blas_backend_name()) < 0 ||
        PyModule_AddObjectRef(m, "HAS_OPENMP", has_openmp) < 0 ||
        PyModule_AddIntConstant(m, "ABI_VERSION", JLINALG_ABI_VERSION) < 0)
        goto fail;

    static const struct {
        const char *name;
        int (*get)(void);
    } BLAS_FLAGS[] = {
        {"blas_is_ilp64", blas_is_ilp64},
        {"blas_has_dgemm", blas_has_external},
        {"blas_has_dsyrk", blas_has_dsyrk},
        {"blas_has_dsyevd", blas_has_dsyevd},
        {"blas_has_lapacke_dsyevd", blas_has_lapacke_dsyevd},
        {"blas_has_dsyevr", blas_has_dsyevr},
    };
    for (size_t i = 0; i < sizeof BLAS_FLAGS / sizeof *BLAS_FLAGS; i++) {
        if (PyModule_AddIntConstant(m, BLAS_FLAGS[i].name, BLAS_FLAGS[i].get()) < 0) goto fail;
    }
    return m;

fail:
    Py_DECREF(m);
    return NULL;
}
