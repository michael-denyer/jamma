/**
 * pymodule.c — Python C extension module _jlinalg.
 *
 * Exposes the jlinalg BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv,
 * dgemm, dsyrk, dsyr2k) and LAPACK eigh to Python via the NumPy buffer
 * protocol.  Arrays are
 * accessed via PyArray_FROM_OTF for contiguous double* extraction (copies
 * non-contiguous or non-float64 inputs as needed).
 *
 * Module-level constants:
 *   jlinalg_isa   — active ISA string ("AVX2", "NEON", or "generic")
 *   HAS_OPENMP  — True if compiled with OpenMP (-fopenmp)
 *   ABI_VERSION — integer (JLINALG_ABI_VERSION from jlinalg.h)
 *   JLINALG_MR    — microkernel row tile size (set by platform.c after ISA detection)
 *   JLINALG_NR    — microkernel column tile size
 *   JLINALG_KC    — KC blocking depth
 *   JLINALG_MC    — MC row panel size
 *   JLINALG_NC    — NC column panel size
 *
 * Exported functions: ddot, dnrm2, daxpy, dscal, dgemv, dgemm, dsyrk, dsyr2k, eigh
 *
 * Patterns follow _lmm_accel.c: PyArray_FROM_OTF with NPY_ARRAY_IN_ARRAY for
 * read-only inputs, NPY_ARRAY_INOUT_ARRAY2 for in-place writeable outputs.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include "jlinalg.h"

/* numpy.linalg.LinAlgError — cached at module init for eigh convergence errors.
 * Falls back to PyExc_RuntimeError if numpy.linalg cannot be imported. */
static PyObject *LinAlgError = NULL;

/* ---------------------------------------------------------------------------
 * py_ddot — dot product of two 1-D float64 arrays
 *
 * Signature: ddot(x: ndarray, y: ndarray) -> float
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_ddot(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;
    if (!PyArg_ParseTuple(args, "OO", &ox, &oy))
        return NULL;

    PyArrayObject *ax = (PyArrayObject *)PyArray_FROM_OTF(
        ox, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ay = (PyArrayObject *)PyArray_FROM_OTF(
        oy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!ax || !ay) {
        Py_XDECREF(ax);
        Py_XDECREF(ay);
        return NULL;
    }

    if (PyArray_NDIM(ax) != 1 || PyArray_NDIM(ay) != 1) {
        PyErr_SetString(PyExc_ValueError, "ddot: x and y must be 1-D arrays");
        Py_DECREF(ax); Py_DECREF(ay);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(ax);
    if (PyArray_SIZE(ay) != n) {
        PyErr_SetString(PyExc_ValueError, "ddot: x and y must have the same length");
        Py_DECREF(ax); Py_DECREF(ay);
        return NULL;
    }

    const double *px = (const double *)PyArray_DATA(ax);
    const double *py = (const double *)PyArray_DATA(ay);
    double result = jlinalg_dispatch.ddot(n, px, 1, py, 1);

    Py_DECREF(ax);
    Py_DECREF(ay);
    return PyFloat_FromDouble(result);
}

/* ---------------------------------------------------------------------------
 * py_dnrm2 — Euclidean norm of a 1-D float64 array
 *
 * Signature: dnrm2(x: ndarray) -> float
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dnrm2(PyObject *self, PyObject *args)
{
    PyObject *ox;
    if (!PyArg_ParseTuple(args, "O", &ox))
        return NULL;

    PyArrayObject *ax = (PyArrayObject *)PyArray_FROM_OTF(
        ox, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!ax)
        return NULL;

    if (PyArray_NDIM(ax) != 1) {
        PyErr_SetString(PyExc_ValueError, "dnrm2: x must be a 1-D array");
        Py_DECREF(ax);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(ax);
    const double *px = (const double *)PyArray_DATA(ax);
    double result = jlinalg_dispatch.dnrm2(n, px, 1);

    Py_DECREF(ax);
    return PyFloat_FromDouble(result);
}

/* ---------------------------------------------------------------------------
 * py_daxpy — y += alpha * x (in-place)
 *
 * Signature: daxpy(alpha: float, x: ndarray, y: ndarray) -> None
 * y is modified in-place.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_daxpy(PyObject *self, PyObject *args)
{
    double alpha;
    PyObject *ox, *oy;
    if (!PyArg_ParseTuple(args, "dOO", &alpha, &ox, &oy))
        return NULL;

    PyArrayObject *ax = (PyArrayObject *)PyArray_FROM_OTF(
        ox, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ay = (PyArrayObject *)PyArray_FROM_OTF(
        oy, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (!ax || !ay) {
        Py_XDECREF(ax);
        if (ay) PyArray_DiscardWritebackIfCopy(ay);
        Py_XDECREF(ay);
        return NULL;
    }

    if (PyArray_NDIM(ax) != 1 || PyArray_NDIM(ay) != 1) {
        PyErr_SetString(PyExc_ValueError, "daxpy: x and y must be 1-D arrays");
        PyArray_DiscardWritebackIfCopy(ay);
        Py_DECREF(ax); Py_DECREF(ay);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(ax);
    if (PyArray_SIZE(ay) != n) {
        PyErr_SetString(PyExc_ValueError, "daxpy: x and y must have the same length");
        PyArray_DiscardWritebackIfCopy(ay);
        Py_DECREF(ax); Py_DECREF(ay);
        return NULL;
    }

    const double *px = (const double *)PyArray_DATA(ax);
    double *py = (double *)PyArray_DATA(ay);
    jlinalg_dispatch.daxpy(n, alpha, px, 1, py, 1);

    PyArray_ResolveWritebackIfCopy(ay);
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_RETURN_NONE;
}

/* ---------------------------------------------------------------------------
 * py_dscal — x *= alpha (in-place)
 *
 * Signature: dscal(alpha: float, x: ndarray) -> None
 * x is modified in-place.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dscal(PyObject *self, PyObject *args)
{
    double alpha;
    PyObject *ox;
    if (!PyArg_ParseTuple(args, "dO", &alpha, &ox))
        return NULL;

    PyArrayObject *ax = (PyArrayObject *)PyArray_FROM_OTF(
        ox, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (!ax)
        return NULL;

    if (PyArray_NDIM(ax) != 1) {
        PyErr_SetString(PyExc_ValueError, "dscal: x must be a 1-D array");
        PyArray_DiscardWritebackIfCopy(ax);
        Py_DECREF(ax);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(ax);
    double *px = (double *)PyArray_DATA(ax);
    jlinalg_dispatch.dscal(n, alpha, px, 1);

    PyArray_ResolveWritebackIfCopy(ax);
    Py_DECREF(ax);
    Py_RETURN_NONE;
}

/* ---------------------------------------------------------------------------
 * py_dgemv — matrix-vector product y = A @ x
 *
 * Signature: dgemv(A: ndarray, x: ndarray) -> ndarray
 * A must be 2-D C-contiguous float64, x must be 1-D float64.
 * Returns a new 1-D float64 array of length m = A.shape[0].
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dgemv(PyObject *self, PyObject *args)
{
    PyObject *oA, *ox;
    if (!PyArg_ParseTuple(args, "OO", &oA, &ox))
        return NULL;

    PyArrayObject *aA = (PyArrayObject *)PyArray_FROM_OTF(
        oA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ax = (PyArrayObject *)PyArray_FROM_OTF(
        ox, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aA || !ax) {
        Py_XDECREF(aA);
        Py_XDECREF(ax);
        return NULL;
    }

    if (PyArray_NDIM(aA) != 2) {
        PyErr_SetString(PyExc_ValueError, "dgemv: A must be a 2-D array");
        Py_DECREF(aA); Py_DECREF(ax);
        return NULL;
    }
    if (PyArray_NDIM(ax) != 1) {
        PyErr_SetString(PyExc_ValueError, "dgemv: x must be a 1-D array");
        Py_DECREF(aA); Py_DECREF(ax);
        return NULL;
    }

    npy_intp m = PyArray_DIM(aA, 0);
    npy_intp n = PyArray_DIM(aA, 1);

    if (PyArray_SIZE(ax) != n) {
        PyErr_Format(PyExc_ValueError,
            "dgemv: A has %ld columns but x has %ld elements",
            (long)n, (long)PyArray_SIZE(ax));
        Py_DECREF(aA); Py_DECREF(ax);
        return NULL;
    }

    /* Allocate output y (length m, float64) */
    PyArrayObject *ay = (PyArrayObject *)PyArray_SimpleNew(1, &m, NPY_DOUBLE);
    if (!ay) {
        Py_DECREF(aA); Py_DECREF(ax);
        return NULL;
    }

    const double *pA = (const double *)PyArray_DATA(aA);
    const double *px = (const double *)PyArray_DATA(ax);
    double *py = (double *)PyArray_DATA(ay);

    jlinalg_dispatch.dgemv(m, n, pA, px, py);

    Py_DECREF(aA);
    Py_DECREF(ax);
    return (PyObject *)ay;
}

/* ---------------------------------------------------------------------------
 * py_dgemm — matrix-matrix product C = op(A) @ op(B)
 *
 * Signature: dgemm(A, B, transa='N', transb='N', out=None) -> ndarray
 * A and B are 2-D float64 arrays.  transa/transb: 'N' (no transpose) or
 * 'T' (transpose).  out: optional preallocated output array (M x N, float64,
 * C-contiguous).  If None, a new array is allocated.  Returns the output array.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dgemm(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"A", "B", "transa", "transb", "out", NULL};
    PyObject *oA, *oB;
    PyObject *oOut = Py_None;
    const char *transa_str = "N";
    const char *transb_str = "N";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ssO", kwlist,
            &oA, &oB, &transa_str, &transb_str, &oOut))
        return NULL;

    /* Validate transpose flags: exactly one char, 'N'/'n' or 'T'/'t'. */
    if (transa_str[0] == '\0' || transa_str[1] != '\0') {
        PyErr_Format(PyExc_ValueError,
            "dgemm: transa must be 'N' or 'T', got '%s'", transa_str);
        return NULL;
    }
    if (transb_str[0] == '\0' || transb_str[1] != '\0') {
        PyErr_Format(PyExc_ValueError,
            "dgemm: transb must be 'N' or 'T', got '%s'", transb_str);
        return NULL;
    }
    char ta = transa_str[0];
    char tb = transb_str[0];
    if (ta != 'N' && ta != 'n' && ta != 'T' && ta != 't') {
        PyErr_Format(PyExc_ValueError,
            "dgemm: transa must be 'N' or 'T', got '%s'", transa_str);
        return NULL;
    }
    if (tb != 'N' && tb != 'n' && tb != 'T' && tb != 't') {
        PyErr_Format(PyExc_ValueError,
            "dgemm: transb must be 'N' or 'T', got '%s'", transb_str);
        return NULL;
    }
    int transa = (ta == 'T' || ta == 't') ? 1 : 0;
    int transb = (tb == 'T' || tb == 't') ? 1 : 0;

    /* Coerce to C-contiguous float64.  Fortran-order / non-contiguous inputs
     * are copied — O(N^2) cost dominated by O(N^3) dgemm for non-trivial M,N,K.
     * Stride-aware packing would avoid the copy but complicates pack_A/pack_B. */
    PyArrayObject *aA = (PyArrayObject *)PyArray_FROM_OTF(
        oA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *aB = (PyArrayObject *)PyArray_FROM_OTF(
        oB, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aA || !aB) {
        Py_XDECREF(aA);
        Py_XDECREF(aB);
        return NULL;
    }

    if (PyArray_NDIM(aA) != 2) {
        PyErr_SetString(PyExc_ValueError, "dgemm: A must be a 2-D array");
        Py_DECREF(aA); Py_DECREF(aB);
        return NULL;
    }
    if (PyArray_NDIM(aB) != 2) {
        PyErr_SetString(PyExc_ValueError, "dgemm: B must be a 2-D array");
        Py_DECREF(aA); Py_DECREF(aB);
        return NULL;
    }

    /* Effective dimensions after transpose:
     *   transa=0: op(A) is A.shape[0] x A.shape[1]  → M=A.shape[0], K=A.shape[1]
     *   transa=1: op(A) is A.shape[1] x A.shape[0]  → M=A.shape[1], K=A.shape[0]
     * Similarly for B. */
    npy_intp M   = transa ? PyArray_DIM(aA, 1) : PyArray_DIM(aA, 0);
    npy_intp K_a = transa ? PyArray_DIM(aA, 0) : PyArray_DIM(aA, 1);
    npy_intp K_b = transb ? PyArray_DIM(aB, 1) : PyArray_DIM(aB, 0);
    npy_intp N   = transb ? PyArray_DIM(aB, 0) : PyArray_DIM(aB, 1);

    if (K_a != K_b) {
        PyErr_Format(PyExc_ValueError,
            "dgemm: inner dimensions mismatch: op(A) is %ldx%ld, op(B) is %ldx%ld",
            (long)M, (long)K_a, (long)K_b, (long)N);
        Py_DECREF(aA); Py_DECREF(aB);
        return NULL;
    }

    /* Output C (M x N): use caller-provided buffer or allocate fresh */
    PyArrayObject *aC;
    if (oOut != Py_None) {
        aC = (PyArrayObject *)PyArray_FROM_OTF(
            oOut, NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED);
        if (!aC) {
            Py_DECREF(aA); Py_DECREF(aB);
            return NULL;
        }
        if (PyArray_NDIM(aC) != 2 ||
            PyArray_DIM(aC, 0) != M || PyArray_DIM(aC, 1) != N) {
            PyErr_Format(PyExc_ValueError,
                "dgemm: out shape (%zd, %zd) doesn't match result shape (%zd, %zd)",
                (Py_ssize_t)PyArray_DIM(aC, 0), (Py_ssize_t)PyArray_DIM(aC, 1),
                (Py_ssize_t)M, (Py_ssize_t)N);
            Py_DECREF(aC); Py_DECREF(aA); Py_DECREF(aB);
            return NULL;
        }
    } else {
        npy_intp dims[2] = {M, N};
        aC = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!aC) {
            Py_DECREF(aA); Py_DECREF(aB);
            return NULL;
        }
    }

    const double *pA = (const double *)PyArray_DATA(aA);
    const double *pB = (const double *)PyArray_DATA(aB);
    double       *pC = (double *)PyArray_DATA(aC);

    /* Guard: if no external BLAS and dgemm workspace allocation failed during
     * init, the packed buffers are NULL and jlinalg_dgemm_c would segfault. */
    if (!blas_has_external() && (!jlinalg_packed_A || !jlinalg_packed_B)) {
        PyErr_SetString(PyExc_RuntimeError,
            "dgemm: workspace allocation failed during jlinalg init; "
            "reduce OMP_NUM_THREADS or use the numpy fallback");
        Py_DECREF(aA); Py_DECREF(aB); Py_DECREF(aC);
        return NULL;
    }

    /* Leading dimensions are the physical column counts (row-major storage) */
    npy_intp lda = PyArray_DIM(aA, 1);
    npy_intp ldb = PyArray_DIM(aB, 1);

    /* Release the GIL for the O(N^3) C/OpenMP computation.  Safe because
     * jlinalg_dgemm_ext operates purely on C double arrays; the PyArray refs
     * (aA, aB, aC) keep the buffers alive for the duration.
     * jlinalg_dgemm_ext routes to system BLAS when available, else jlinalg own. */
    Py_BEGIN_ALLOW_THREADS
    jlinalg_dgemm_ext(M, N, K_a, pA, lda, pB, ldb, pC, N, transa, transb);
    Py_END_ALLOW_THREADS

    Py_DECREF(aA);
    Py_DECREF(aB);
    return (PyObject *)aC;
}

/* ---------------------------------------------------------------------------
 * py_dsyrk — symmetric rank-k update K = X @ X.T
 *
 * Signature: dsyrk(X: ndarray) -> ndarray
 * X must be 2-D C-contiguous float64 of shape (N, K).
 * Returns a new 2-D float64 array of shape (N, N), bitwise symmetric.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dsyrk(PyObject *self, PyObject *args)
{
    PyObject *oX;
    if (!PyArg_ParseTuple(args, "O", &oX))
        return NULL;

    PyArrayObject *aX = (PyArrayObject *)PyArray_FROM_OTF(
        oX, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aX)
        return NULL;

    if (PyArray_NDIM(aX) != 2) {
        PyErr_SetString(PyExc_ValueError,
            "dsyrk: X must be a 2-D array");
        Py_DECREF(aX);
        return NULL;
    }

    npy_intp N = PyArray_DIM(aX, 0);
    npy_intp K = PyArray_DIM(aX, 1);

    /* Guard: workspace must be allocated (jlinalg_init succeeded).
     * Check BEFORE allocating the N×N output to avoid a large allocation
     * that would be immediately freed on failure.
     * Skip guard if vendor dsyrk is available (doesn't need jlinalg workspace). */
    if (!blas_has_dsyrk() && (!jlinalg_packed_A || !jlinalg_packed_B)) {
        PyErr_SetString(PyExc_RuntimeError,
            "dsyrk: workspace allocation failed during jlinalg init; "
            "reduce OMP_NUM_THREADS or use the numpy fallback");
        Py_DECREF(aX);
        return NULL;
    }

    npy_intp dims[2] = {N, N};
    /* Use SimpleNew (uninitialized) — jlinalg_dsyrk_c zeroes C before accumulating. */
    PyArrayObject *aC = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!aC) {
        Py_DECREF(aX);
        return NULL;
    }

    const double *pX = (const double *)PyArray_DATA(aX);
    double       *pC = (double *)PyArray_DATA(aC);

    Py_BEGIN_ALLOW_THREADS
    /* ldx = K, ldc = N: safe because PyArray_FROM_OTF guarantees C-contiguous layout.
     * jlinalg_dsyrk_ext routes to vendor cblas_dsyrk when available, else jlinalg_dsyrk_c. */
    jlinalg_dsyrk_ext(N, K, pX, K, pC, N);
    Py_END_ALLOW_THREADS

    Py_DECREF(aX);
    return (PyObject *)aC;
}

/* ---------------------------------------------------------------------------
 * py_dsyr2k — symmetric rank-2k update: result = C - A @ B.T - B @ A.T
 *
 * Signature: dsyr2k(C: ndarray, A: ndarray, B: ndarray) -> ndarray
 * C must be 2-D float64 of shape (N, N); A and B must be (N, K).
 * Returns a new 2-D float64 array of shape (N, N).
 * The input C is not modified.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dsyr2k(PyObject *self, PyObject *args)
{
    PyObject *oC, *oA, *oB;
    if (!PyArg_ParseTuple(args, "OOO", &oC, &oA, &oB))
        return NULL;

    PyArrayObject *aC_in = (PyArrayObject *)PyArray_FROM_OTF(
        oC, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *aA = (PyArrayObject *)PyArray_FROM_OTF(
        oA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *aB = (PyArrayObject *)PyArray_FROM_OTF(
        oB, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aC_in || !aA || !aB) {
        Py_XDECREF(aC_in);
        Py_XDECREF(aA);
        Py_XDECREF(aB);
        return NULL;
    }

    if (PyArray_NDIM(aC_in) != 2) {
        PyErr_SetString(PyExc_ValueError, "dsyr2k: C must be a 2-D array");
        goto err_dsyr2k;
    }
    if (PyArray_DIM(aC_in, 0) != PyArray_DIM(aC_in, 1)) {
        PyErr_Format(PyExc_ValueError,
            "dsyr2k: C must be square, got shape (%ld, %ld)",
            (long)PyArray_DIM(aC_in, 0), (long)PyArray_DIM(aC_in, 1));
        goto err_dsyr2k;
    }
    if (PyArray_NDIM(aA) != 2) {
        PyErr_SetString(PyExc_ValueError, "dsyr2k: A must be a 2-D array");
        goto err_dsyr2k;
    }
    if (PyArray_NDIM(aB) != 2) {
        PyErr_SetString(PyExc_ValueError, "dsyr2k: B must be a 2-D array");
        goto err_dsyr2k;
    }

    {
        npy_intp N = PyArray_DIM(aC_in, 0);
        npy_intp K = PyArray_DIM(aA, 1);

        if (PyArray_DIM(aA, 0) != N) {
            PyErr_Format(PyExc_ValueError,
                "dsyr2k: A rows (%ld) must match C dimension (%ld)",
                (long)PyArray_DIM(aA, 0), (long)N);
            goto err_dsyr2k;
        }
        if (PyArray_DIM(aB, 0) != N) {
            PyErr_Format(PyExc_ValueError,
                "dsyr2k: B rows (%ld) must match C dimension (%ld)",
                (long)PyArray_DIM(aB, 0), (long)N);
            goto err_dsyr2k;
        }
        if (PyArray_DIM(aB, 1) != K) {
            PyErr_Format(PyExc_ValueError,
                "dsyr2k: A columns (%ld) must match B columns (%ld)",
                (long)K, (long)PyArray_DIM(aB, 1));
            goto err_dsyr2k;
        }

        /* Create output: copy C into a new C-contiguous array */
        PyArrayObject *aC_out = (PyArrayObject *)PyArray_NewCopy(aC_in, NPY_CORDER);
        if (!aC_out)
            goto err_dsyr2k;

        /* Guard: workspace must be allocated */
        if (!jlinalg_packed_A || !jlinalg_packed_B) {
            PyErr_SetString(PyExc_RuntimeError,
                "dsyr2k: workspace allocation failed during jlinalg init; "
                "reduce OMP_NUM_THREADS or use the numpy fallback");
            Py_DECREF(aC_out);
            goto err_dsyr2k;
        }

        const double *pA = (const double *)PyArray_DATA(aA);
        const double *pB = (const double *)PyArray_DATA(aB);
        double       *pC = (double *)PyArray_DATA(aC_out);

        Py_BEGIN_ALLOW_THREADS
        /* jlinalg_dsyr2k_c subtracts A @ B.T + B @ A.T from all elements of pC
         * (full-matrix update, no mirror step).
         * lda = ldb = K, ldc = N: safe because PyArray_FROM_OTF guarantees
         * C-contiguous layout. */
        jlinalg_dsyr2k_c(N, K, pA, K, pB, K, pC, N);
        Py_END_ALLOW_THREADS

        Py_DECREF(aC_in);
        Py_DECREF(aA);
        Py_DECREF(aB);
        return (PyObject *)aC_out;
    }

err_dsyr2k:
    Py_XDECREF(aC_in);
    Py_XDECREF(aA);
    Py_XDECREF(aB);
    return NULL;
}

/* ---------------------------------------------------------------------------
 * py_eigh — compute eigenvalues and eigenvectors of symmetric matrix
 *
 * Signature: eigh(K: ndarray, inplace: bool = False) -> tuple[ndarray, ndarray]
 * K must be 2-D C-contiguous float64 of shape (N, N).
 *
 * When inplace=False (default): K is used as scratch; a fresh N×N eigenvector
 * array is allocated and returned.  Backward compatible with existing callers.
 *
 * When inplace=True: K is overwritten in-place with eigenvectors.  No separate
 * N×N allocation is made (only the N eigenvalues).  The returned eigenvector
 * array IS K.  This saves N²×8 bytes at 125k scale (~125 GB).
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_eigh(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *oK;
    int inplace = 0;
    static char *kwlist[] = {"K", "inplace", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", kwlist, &oK, &inplace))
        return NULL;

    PyArrayObject *aK = (PyArrayObject *)PyArray_FROM_OTF(
        oK, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (!aK) return NULL;

    if (PyArray_NDIM(aK) != 2 || PyArray_DIM(aK,0) != PyArray_DIM(aK,1)) {
        PyErr_SetString(PyExc_ValueError, "eigh: K must be 2-D square float64");
        PyArray_DiscardWritebackIfCopy(aK);
        Py_DECREF(aK);
        return NULL;
    }

    npy_intp N = PyArray_DIM(aK, 0);
    double *pK = (double *)PyArray_DATA(aK);

    /* Allocate eigenvalues (N,) — always needed */
    PyArrayObject *aW = (PyArrayObject *)PyArray_SimpleNew(1, &N, NPY_DOUBLE);
    if (!aW) {
        PyArray_DiscardWritebackIfCopy(aK);
        Py_DECREF(aK);
        return NULL;
    }

    /* Reject inplace when FROM_OTF created a temporary copy (non-float64,
     * non-C-contiguous, or read-only input).  WRITEBACKIFCOPY means aK is a
     * copy that would be written back on resolve — the returned eigenvector
     * array would NOT be the caller's original K, violating the buffer
     * identity contract and defeating the memory-saving goal. */
    if (inplace && (PyArray_FLAGS(aK) & NPY_ARRAY_WRITEBACKIFCOPY)) {
        PyErr_SetString(PyExc_ValueError,
            "eigh: inplace=True requires a C-contiguous, writeable, float64 array. "
            "The input was converted to a temporary copy.");
        Py_DECREF(aW);
        PyArray_DiscardWritebackIfCopy(aK);
        Py_DECREF(aK);
        return NULL;
    }

    /* Eigenvector buffer: when inplace=True, reuse K directly (no N×N alloc).
     * When inplace=False (default), allocate a separate N×N array. */
    PyArrayObject *aU = NULL;
    double *pU;
    if (inplace) {
        pU = pK;  /* K and eigenvectors share the same buffer */
    } else {
        npy_intp dims2[2] = {N, N};
        aU = (PyArrayObject *)PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
        if (!aU) {
            Py_DECREF(aW);
            PyArray_DiscardWritebackIfCopy(aK);
            Py_DECREF(aK);
            return NULL;
        }
        pU = (double *)PyArray_DATA(aU);
    }

    double *pW = (double *)PyArray_DATA(aW);

    /* Initialize status struct and reset LP64 overflow counter */
    jlinalg_eigh_status_t eigh_status;
    memset(&eigh_status, 0, sizeof(eigh_status));
    blas_dispatch_reset_lp64_overflow();

    int ret;
    Py_BEGIN_ALLOW_THREADS
    /* ldk = ldz = N: safe because PyArray_FROM_OTF/SimpleNew guarantee C-contiguous.
     * When inplace=True, pK == pU so jlinalg_eigh_c receives K==eigenvectors.
     * This is safe on the vendor DSYEVD/DSYEVR paths.  The D&C pipeline
     * (dsytrd+dstedc+dormtr) rejects K==eigenvectors with an error because
     * dstedc would overwrite the Householder reflectors that dormtr needs. */
    ret = jlinalg_eigh_c(N, pK, N, pW, pU, N, &eigh_status);
    Py_END_ALLOW_THREADS

    if (ret != 0) {
        if (ret == JLINALG_EXT_INPLACE_UNSUPPORTED) {
            PyErr_Format(PyExc_RuntimeError,
                "jlinalg eigh: inplace=True requires vendor LAPACK "
                "(DSYEVD or DSYEVR); neither is available. "
                "Use inplace=False or install ILP64 numpy.");
        } else if (ret == JLINALG_EXT_ALLOC_FAIL) {
            PyErr_Format(PyExc_MemoryError,
                "jlinalg eigh: workspace allocation failed — "
                "matrix too large for available memory");
        } else if (ret == JLINALG_EXT_COUNT_MISMATCH) {
            PyErr_Format(PyExc_RuntimeError,
                "jlinalg eigh: vendor LAPACK DSYEVR returned fewer eigenvalues "
                "than expected — this indicates an ABI mismatch or vendor bug");
        } else if (ret == JLINALG_EXT_INTERNAL_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                "jlinalg eigh: internal error in dstedc (unsupported GEMM parameters) — "
                "this is a jlinalg bug, please report it");
        } else if (ret < 0) {
            PyErr_Format(PyExc_RuntimeError,
                "jlinalg eigh: illegal argument to vendor LAPACK (info=%d) — "
                "this is a jlinalg bug, please report it", ret);
        } else {
            /* Convergence failure — raise numpy.linalg.LinAlgError for
             * consistency with numpy.linalg.eigh callers. */
            PyErr_Format(LinAlgError,
                "jlinalg eigh: convergence failure (returned %d)", ret);
        }
        Py_DECREF(aW);
        Py_XDECREF(aU);
        PyArray_DiscardWritebackIfCopy(aK);
        Py_DECREF(aK);
        return NULL;
    }

    /* Surface performance fallback warnings to Python (non-fatal).
     * When inplace=True, K is already overwritten — we cannot undo it, so
     * promoted-to-error warnings are cleared and results returned anyway.
     * When inplace=False, promoted warnings abort via goto warn_error. */
#define EMIT_STATUS_WARNING(msg)                                 \
    do {                                                         \
        if (PyErr_WarnEx(PyExc_RuntimeWarning, (msg), 1) < 0) { \
            if (inplace) { PyErr_Clear(); }                      \
            else          { goto warn_error; }                   \
        }                                                        \
    } while (0)

    if (eigh_status.dstedc_ws_fallback) {
        EMIT_STATUS_WARNING(
                "jlinalg eigh: dstedc workspace allocation failed — "
                "using global mutex path (significantly slower for large matrices). "
                "Reduce matrix size or free memory.");
    }
    if (eigh_status.dsytrd_mirror_fallback) {
        EMIT_STATUS_WARNING(
                "jlinalg eigh: dsytrd mirror buffer allocation failed — "
                "using scalar dsymv (slower tridiagonalization). "
                "Reduce matrix size or free memory.");
    }
    if (eigh_status.secular_failures > 0) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "jlinalg eigh: %d secular equation(s) failed to converge — "
            "eigenvalues may have reduced accuracy for near-degenerate modes",
            eigh_status.secular_failures);
        EMIT_STATUS_WARNING(msg);
    }
    if (eigh_status.qr_fallback) {
        EMIT_STATUS_WARNING(
                "jlinalg eigh: D&C eigensolver failed residual check — "
                "fell back to QR iteration (much slower for large matrices)");
    }
    if (eigh_status.vendor_lapack_skipped) {
        EMIT_STATUS_WARNING(
                "jlinalg eigh: vendor LAPACK work buffer allocation failed — "
                "using jlinalg D&C pipeline instead. "
                "Free memory or reduce matrix size.");
    }
    if (blas_dispatch_lp64_overflow_count() > 0) {
        EMIT_STATUS_WARNING(
                "jlinalg eigh: LP64 overflow guard triggered during GEMM — "
                "fell back to jlinalg own dgemm (much slower). "
                "Install ILP64 numpy for large matrices.");
    }

#undef EMIT_STATUS_WARNING

    /* Commit writeback after warnings pass (no-op when inplace=True since
     * WRITEBACKIFCOPY is rejected at entry). */
    PyArray_ResolveWritebackIfCopy(aK);

    /* Build result tuple.  Py_BuildValue("(NN)") steals references. */
    PyObject *result;
    if (inplace) {
        /* Eigenvectors live in K — return aK as the eigenvector array. */
        result = Py_BuildValue("(NN)", aW, (PyObject *)aK);
    } else {
        Py_DECREF(aK);
        result = Py_BuildValue("(NN)", aW, aU);
    }
    return result;

warn_error:
    Py_DECREF(aW);
    Py_XDECREF(aU);
    PyArray_DiscardWritebackIfCopy(aK);
    Py_DECREF(aK);
    return NULL;
}

/* ---------------------------------------------------------------------------
 * py_qr — reduced QR factorization via vendor LAPACK dgeqrf + dorgqr
 *
 * Signature: qr(A: ndarray) -> tuple[ndarray, ndarray]
 * A must be 2-D float64 of shape (m, n) with m >= n.
 * Returns (Q, R) where Q is (m, n) and R is (n, n) upper triangular.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_qr(PyObject *self, PyObject *args)
{
    PyObject *oA;
    if (!PyArg_ParseTuple(args, "O", &oA))
        return NULL;

    PyArrayObject *aA = (PyArrayObject *)PyArray_FROM_OTF(
        oA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aA) return NULL;

    if (PyArray_NDIM(aA) != 2) {
        PyErr_SetString(PyExc_ValueError, "qr: A must be a 2-D array");
        Py_DECREF(aA);
        return NULL;
    }

    npy_intp m = PyArray_DIM(aA, 0);
    npy_intp n = PyArray_DIM(aA, 1);
    if (m < 1 || n < 1) {
        PyErr_SetString(PyExc_ValueError, "qr: A must have positive dimensions");
        Py_DECREF(aA);
        return NULL;
    }
    if (m < n) {
        PyErr_Format(PyExc_ValueError,
            "qr: requires m >= n (tall-skinny), got shape (%ld, %ld)",
            (long)m, (long)n);
        Py_DECREF(aA);
        return NULL;
    }

    npy_intp minmn = m < n ? m : n;
    const double *pA = (const double *)PyArray_DATA(aA);

    /* Allocate column-major work buffer and transpose row-major A into it */
    double *A_col = (double *)malloc((size_t)m * (size_t)n * sizeof(double));
    double *tau = (double *)malloc((size_t)minmn * sizeof(double));
    if (!A_col || !tau) {
        free(A_col); free(tau);
        Py_DECREF(aA);
        PyErr_NoMemory();
        return NULL;
    }

    /* Row-major -> column-major transpose */
    for (npy_intp i = 0; i < m; i++)
        for (npy_intp j = 0; j < n; j++)
            A_col[j * m + i] = pA[i * n + j];

    int ret;
    Py_BEGIN_ALLOW_THREADS
    ret = jlinalg_dgeqrf_ext(m, n, A_col, m, tau);
    Py_END_ALLOW_THREADS

    if (ret != JLINALG_EXT_SUCCESS) {
        free(A_col); free(tau);
        Py_DECREF(aA);
        if (ret == JLINALG_EXT_UNAVAILABLE)
            PyErr_SetString(PyExc_ValueError, "qr: vendor LAPACK not available");
        else if (ret == JLINALG_EXT_ALLOC_FAIL)
            PyErr_NoMemory();
        else
            PyErr_Format(LinAlgError, "qr: dgeqrf failed (info=%d)", ret);
        return NULL;
    }

    /* Extract R from upper triangle of A_col BEFORE dorgqr overwrites it.
     * R is n x n upper triangular (row-major output). */
    npy_intp rdims[2] = {n, n};
    PyArrayObject *aR = (PyArrayObject *)PyArray_ZEROS(2, rdims, NPY_DOUBLE, 0);
    if (!aR) {
        free(A_col); free(tau);
        Py_DECREF(aA);
        return NULL;
    }
    double *pR = (double *)PyArray_DATA(aR);
    /* A_col is col-major: A_col[i + j*m] = element (i,j).
     * R[i][j] for i <= j, i,j in [0,n) */
    for (npy_intp j = 0; j < n; j++)
        for (npy_intp i = 0; i <= j && i < n; i++)
            pR[i * n + j] = A_col[i + j * m];

    /* Now generate Q from Householder vectors */
    int ret2;
    Py_BEGIN_ALLOW_THREADS
    ret2 = jlinalg_dorgqr_ext(m, n, A_col, m, tau);
    Py_END_ALLOW_THREADS

    free(tau);

    if (ret2 != JLINALG_EXT_SUCCESS) {
        free(A_col);
        Py_DECREF(aA); Py_DECREF(aR);
        if (ret2 == JLINALG_EXT_UNAVAILABLE)
            PyErr_SetString(PyExc_ValueError, "qr: vendor LAPACK not available for dorgqr");
        else if (ret2 == JLINALG_EXT_ALLOC_FAIL)
            PyErr_NoMemory();
        else
            PyErr_Format(LinAlgError, "qr: dorgqr failed (info=%d)", ret2);
        return NULL;
    }

    /* Transpose Q from col-major A_col to row-major output (m x n) */
    npy_intp qdims[2] = {m, n};
    PyArrayObject *aQ = (PyArrayObject *)PyArray_SimpleNew(2, qdims, NPY_DOUBLE);
    if (!aQ) {
        free(A_col);
        Py_DECREF(aA); Py_DECREF(aR);
        return NULL;
    }
    double *pQ = (double *)PyArray_DATA(aQ);
    for (npy_intp i = 0; i < m; i++)
        for (npy_intp j = 0; j < n; j++)
            pQ[i * n + j] = A_col[j * m + i];

    free(A_col);
    Py_DECREF(aA);

    PyObject *result = PyTuple_Pack(2, (PyObject *)aQ, (PyObject *)aR);
    Py_DECREF(aQ);
    Py_DECREF(aR);
    return result;
}

/* ---------------------------------------------------------------------------
 * py_svd — reduced SVD via vendor LAPACK dgesvd
 *
 * Signature: svd(A, compute_uv=True) -> (U, s, Vh) or s
 * A must be 2-D float64 of shape (m, n) with m >= n.
 * Returns (U, s, Vh) where U is (m, n), s is (n,), Vh is (n, n).
 * If compute_uv=False, returns s only.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_svd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"A", "compute_uv", NULL};
    PyObject *oA;
    int compute_uv = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist,
                                      &oA, &compute_uv))
        return NULL;

    PyArrayObject *aA = (PyArrayObject *)PyArray_FROM_OTF(
        oA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!aA) return NULL;

    if (PyArray_NDIM(aA) != 2) {
        PyErr_SetString(PyExc_ValueError, "svd: A must be a 2-D array");
        Py_DECREF(aA);
        return NULL;
    }

    npy_intp m = PyArray_DIM(aA, 0);
    npy_intp n = PyArray_DIM(aA, 1);
    if (m < n) {
        PyErr_Format(PyExc_ValueError,
            "svd: requires m >= n (tall-skinny), got shape (%ld, %ld)",
            (long)m, (long)n);
        Py_DECREF(aA);
        return NULL;
    }
    if (m < 1 || n < 1) {
        PyErr_SetString(PyExc_ValueError, "svd: A must have positive dimensions");
        Py_DECREF(aA);
        return NULL;
    }

    const double *pA = (const double *)PyArray_DATA(aA);

    /* Allocate column-major work buffer and transpose */
    double *A_col = (double *)malloc((size_t)m * (size_t)n * sizeof(double));
    if (!A_col) {
        Py_DECREF(aA);
        PyErr_NoMemory();
        return NULL;
    }
    for (npy_intp i = 0; i < m; i++)
        for (npy_intp j = 0; j < n; j++)
            A_col[j * m + i] = pA[i * n + j];

    /* Allocate singular values array */
    PyArrayObject *aS = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    if (!aS) {
        free(A_col); Py_DECREF(aA);
        return NULL;
    }
    double *pS = (double *)PyArray_DATA(aS);

    if (compute_uv) {
        double *U_col = (double *)malloc((size_t)m * (size_t)n * sizeof(double));
        double *Vt_col = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        if (!U_col || !Vt_col) {
            free(U_col); free(Vt_col); free(A_col);
            Py_DECREF(aA); Py_DECREF(aS);
            PyErr_NoMemory();
            return NULL;
        }

        int ret;
        Py_BEGIN_ALLOW_THREADS
        ret = jlinalg_dgesvd_ext(m, n, A_col, m, pS, U_col, m, Vt_col, n, 1);
        Py_END_ALLOW_THREADS

        free(A_col);

        if (ret != JLINALG_EXT_SUCCESS) {
            free(U_col); free(Vt_col);
            Py_DECREF(aA); Py_DECREF(aS);
            if (ret == JLINALG_EXT_UNAVAILABLE)
                PyErr_SetString(PyExc_ValueError, "svd: vendor LAPACK not available");
            else if (ret == JLINALG_EXT_ALLOC_FAIL)
                PyErr_NoMemory();
            else
                PyErr_Format(LinAlgError, "svd: dgesvd failed (info=%d)", ret);
            return NULL;
        }

        /* Transpose U_col (col-major m x n) -> row-major U (m x n) */
        npy_intp udims[2] = {m, n};
        PyArrayObject *aU = (PyArrayObject *)PyArray_SimpleNew(2, udims, NPY_DOUBLE);
        /* Transpose Vt_col (col-major n x n) -> row-major Vh (n x n) */
        npy_intp vdims[2] = {n, n};
        PyArrayObject *aVh = (PyArrayObject *)PyArray_SimpleNew(2, vdims, NPY_DOUBLE);
        if (!aU || !aVh) {
            Py_XDECREF(aU); Py_XDECREF(aVh);
            free(U_col); free(Vt_col);
            Py_DECREF(aA); Py_DECREF(aS);
            return NULL;
        }

        double *pU = (double *)PyArray_DATA(aU);
        for (npy_intp i = 0; i < m; i++)
            for (npy_intp j = 0; j < n; j++)
                pU[i * n + j] = U_col[j * m + i];

        double *pVh = (double *)PyArray_DATA(aVh);
        for (npy_intp i = 0; i < n; i++)
            for (npy_intp j = 0; j < n; j++)
                pVh[i * n + j] = Vt_col[j * n + i];

        free(U_col); free(Vt_col);
        Py_DECREF(aA);

        PyObject *result = PyTuple_Pack(3, (PyObject *)aU, (PyObject *)aS, (PyObject *)aVh);
        Py_DECREF(aU); Py_DECREF(aS); Py_DECREF(aVh);
        return result;
    } else {
        /* compute_uv=False: singular values only */
        int ret;
        Py_BEGIN_ALLOW_THREADS
        ret = jlinalg_dgesvd_ext(m, n, A_col, m, pS, NULL, 1, NULL, 1, 0);
        Py_END_ALLOW_THREADS

        free(A_col);
        Py_DECREF(aA);

        if (ret != JLINALG_EXT_SUCCESS) {
            Py_DECREF(aS);
            if (ret == JLINALG_EXT_UNAVAILABLE)
                PyErr_SetString(PyExc_ValueError, "svd: vendor LAPACK not available");
            else if (ret == JLINALG_EXT_ALLOC_FAIL)
                PyErr_NoMemory();
            else
                PyErr_Format(LinAlgError, "svd: dgesvd failed (info=%d)", ret);
            return NULL;
        }
        return (PyObject *)aS;
    }
}

/* ---------------------------------------------------------------------------
 * py_set_n_threads — Set jlinalg thread count (clamped to init-time max).
 *
 * Signature: set_n_threads(n: int) -> int
 * Returns the previous thread count.
 * Raises ValueError if n < 1.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_set_n_threads(PyObject *self, PyObject *args)
{
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    int old = jlinalg_set_n_threads(n);
    if (old < 0) {
        PyErr_SetString(PyExc_ValueError, "set_n_threads: n must be >= 1");
        return NULL;
    }
    return PyLong_FromLong(old);
}

/* ---------------------------------------------------------------------------
 * py_get_n_threads — Get current jlinalg thread count.
 *
 * Signature: get_n_threads() -> int
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_get_n_threads(PyObject *self, PyObject *args)
{
    (void)args;  /* unused */
    return PyLong_FromLong(jlinalg_get_n_threads());
}

/* ---------------------------------------------------------------------------
 * py_compute_snp_stats_chunk — Single-pass per-SNP statistics.
 *
 * Signature: compute_snp_stats_chunk(data, means, miss_counts, vars
 *                                    [, n_aa, n_ab, n_bb])
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_compute_snp_stats_chunk(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *o_data, *o_means, *o_miss, *o_vars;
    PyObject *o_naa = Py_None, *o_nab = Py_None, *o_nbb = Py_None;

    if (!PyArg_ParseTuple(args, "OOOO|OOO",
            &o_data, &o_means, &o_miss, &o_vars,
            &o_naa, &o_nab, &o_nbb))
        return NULL;

    /* Extract data array — accept both float32 and float64.
     * NPY_NOTYPE preserves original dtype; NPY_ARRAY_C_CONTIGUOUS ensures
     * contiguity and native byte order. Single conversion, no re-probe. */
    PyArrayObject *a_data = (PyArrayObject *)PyArray_FROM_OTF(
        o_data, NPY_NOTYPE, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_data) return NULL;

    int dtype = PyArray_TYPE(a_data);
    if (dtype != NPY_FLOAT32 && dtype != NPY_FLOAT64) {
        Py_DECREF(a_data);
        PyErr_SetString(PyExc_TypeError,
            "compute_snp_stats_chunk: data must be float32 or float64");
        return NULL;
    }
    if (PyArray_NDIM(a_data) != 2) {
        Py_DECREF(a_data);
        PyErr_SetString(PyExc_ValueError,
            "compute_snp_stats_chunk: data must be 2-D");
        return NULL;
    }

    npy_intp n_samples = PyArray_DIM(a_data, 0);
    npy_intp n_snps = PyArray_DIM(a_data, 1);

    /* Validate HWE args: all-None or all-array, not a mix */
    int naa_none = (o_naa == Py_None);
    int nab_none = (o_nab == Py_None);
    int nbb_none = (o_nbb == Py_None);
    int n_hwe_none = naa_none + nab_none + nbb_none;
    if (n_hwe_none != 0 && n_hwe_none != 3) {
        Py_DECREF(a_data);
        PyErr_SetString(PyExc_ValueError,
            "compute_snp_stats_chunk: n_aa, n_ab, n_bb must all be arrays "
            "or all None");
        return NULL;
    }

    /* Extract output arrays with INOUT for writeable access */
    PyArrayObject *a_means = (PyArrayObject *)PyArray_FROM_OTF(
        o_means, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject *a_miss = (PyArrayObject *)PyArray_FROM_OTF(
        o_miss, NPY_INTP, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject *a_vars = (PyArrayObject *)PyArray_FROM_OTF(
        o_vars, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!a_means || !a_miss || !a_vars) {
        Py_DECREF(a_data);
        if (a_means) PyArray_DiscardWritebackIfCopy(a_means);
        if (a_miss)  PyArray_DiscardWritebackIfCopy(a_miss);
        if (a_vars)  PyArray_DiscardWritebackIfCopy(a_vars);
        Py_XDECREF(a_means); Py_XDECREF(a_miss); Py_XDECREF(a_vars);
        return NULL;
    }

    /* Validate output array sizes match data columns */
    if (PyArray_SIZE(a_means) < n_snps ||
        PyArray_SIZE(a_miss) < n_snps ||
        PyArray_SIZE(a_vars) < n_snps) {
        PyErr_Format(PyExc_ValueError,
            "compute_snp_stats_chunk: output arrays must have at least %zd "
            "elements (data has %zd columns), got means=%zd, miss=%zd, "
            "vars=%zd",
            (Py_ssize_t)n_snps, (Py_ssize_t)n_snps,
            (Py_ssize_t)PyArray_SIZE(a_means),
            (Py_ssize_t)PyArray_SIZE(a_miss),
            (Py_ssize_t)PyArray_SIZE(a_vars));
        Py_DECREF(a_data);
        PyArray_DiscardWritebackIfCopy(a_means);
        PyArray_DiscardWritebackIfCopy(a_miss);
        PyArray_DiscardWritebackIfCopy(a_vars);
        Py_DECREF(a_means); Py_DECREF(a_miss); Py_DECREF(a_vars);
        return NULL;
    }

    /* HWE arrays (optional — None means no HWE) */
    int compute_hwe = 0;
    PyArrayObject *a_naa = NULL, *a_nab = NULL, *a_nbb = NULL;
    int64_t *naa_ptr = NULL, *nab_ptr = NULL, *nbb_ptr = NULL;
    if (n_hwe_none == 0) {
        compute_hwe = 1;
        a_naa = (PyArrayObject *)PyArray_FROM_OTF(o_naa, NPY_INT64, NPY_ARRAY_INOUT_ARRAY2);
        a_nab = (PyArrayObject *)PyArray_FROM_OTF(o_nab, NPY_INT64, NPY_ARRAY_INOUT_ARRAY2);
        a_nbb = (PyArrayObject *)PyArray_FROM_OTF(o_nbb, NPY_INT64, NPY_ARRAY_INOUT_ARRAY2);
        if (!a_naa || !a_nab || !a_nbb) {
            Py_DECREF(a_data);
            PyArray_DiscardWritebackIfCopy(a_means);
            PyArray_DiscardWritebackIfCopy(a_miss);
            PyArray_DiscardWritebackIfCopy(a_vars);
            Py_DECREF(a_means); Py_DECREF(a_miss); Py_DECREF(a_vars);
            if (a_naa) PyArray_DiscardWritebackIfCopy(a_naa);
            if (a_nab) PyArray_DiscardWritebackIfCopy(a_nab);
            if (a_nbb) PyArray_DiscardWritebackIfCopy(a_nbb);
            Py_XDECREF(a_naa); Py_XDECREF(a_nab); Py_XDECREF(a_nbb);
            return NULL;
        }
        if (PyArray_SIZE(a_naa) < n_snps ||
            PyArray_SIZE(a_nab) < n_snps ||
            PyArray_SIZE(a_nbb) < n_snps) {
            PyErr_Format(PyExc_ValueError,
                "compute_snp_stats_chunk: HWE arrays must have at least %zd "
                "elements, got n_aa=%zd, n_ab=%zd, n_bb=%zd",
                (Py_ssize_t)n_snps,
                (Py_ssize_t)PyArray_SIZE(a_naa),
                (Py_ssize_t)PyArray_SIZE(a_nab),
                (Py_ssize_t)PyArray_SIZE(a_nbb));
            Py_DECREF(a_data);
            PyArray_DiscardWritebackIfCopy(a_means);
            PyArray_DiscardWritebackIfCopy(a_miss);
            PyArray_DiscardWritebackIfCopy(a_vars);
            Py_DECREF(a_means); Py_DECREF(a_miss); Py_DECREF(a_vars);
            PyArray_DiscardWritebackIfCopy(a_naa);
            PyArray_DiscardWritebackIfCopy(a_nab);
            PyArray_DiscardWritebackIfCopy(a_nbb);
            Py_DECREF(a_naa); Py_DECREF(a_nab); Py_DECREF(a_nbb);
            return NULL;
        }
        naa_ptr = (int64_t *)PyArray_DATA(a_naa);
        nab_ptr = (int64_t *)PyArray_DATA(a_nab);
        nbb_ptr = (int64_t *)PyArray_DATA(a_nbb);
    }

    /* Dispatch based on dtype — release GIL for the C kernel */
    Py_BEGIN_ALLOW_THREADS
    if (dtype == NPY_FLOAT32) {
        snp_stats_chunk_f32(
            (const float *)PyArray_DATA(a_data), n_samples, n_snps,
            (double *)PyArray_DATA(a_means), (npy_intp *)PyArray_DATA(a_miss),
            (double *)PyArray_DATA(a_vars),
            naa_ptr, nab_ptr, nbb_ptr, compute_hwe);
    } else {
        snp_stats_chunk_f64(
            (const double *)PyArray_DATA(a_data), n_samples, n_snps,
            (double *)PyArray_DATA(a_means), (npy_intp *)PyArray_DATA(a_miss),
            (double *)PyArray_DATA(a_vars),
            naa_ptr, nab_ptr, nbb_ptr, compute_hwe);
    }
    Py_END_ALLOW_THREADS

    /* Resolve INOUT arrays */
    PyArray_ResolveWritebackIfCopy(a_means);
    PyArray_ResolveWritebackIfCopy(a_miss);
    PyArray_ResolveWritebackIfCopy(a_vars);
    if (a_naa) PyArray_ResolveWritebackIfCopy(a_naa);
    if (a_nab) PyArray_ResolveWritebackIfCopy(a_nab);
    if (a_nbb) PyArray_ResolveWritebackIfCopy(a_nbb);

    Py_DECREF(a_data);
    Py_DECREF(a_means); Py_DECREF(a_miss); Py_DECREF(a_vars);
    Py_XDECREF(a_naa); Py_XDECREF(a_nab); Py_XDECREF(a_nbb);
    Py_RETURN_NONE;
}

/* ---------------------------------------------------------------------------
 * Method table
 * ---------------------------------------------------------------------------
 */
static PyMethodDef JlinalgMethods[] = {
    {"ddot",  py_ddot,  METH_VARARGS,
        "ddot(x, y) -> float\n"
        "Double-precision dot product of two 1-D float64 arrays."},
    {"dnrm2", py_dnrm2, METH_VARARGS,
        "dnrm2(x) -> float\n"
        "Euclidean norm of a 1-D float64 array (Blue algorithm)."},
    {"daxpy", py_daxpy, METH_VARARGS,
        "daxpy(alpha, x, y) -> None\n"
        "y += alpha * x in-place."},
    {"dscal", py_dscal, METH_VARARGS,
        "dscal(alpha, x) -> None\n"
        "x *= alpha in-place."},
    {"dgemv", py_dgemv, METH_VARARGS,
        "dgemv(A, x) -> ndarray\n"
        "Matrix-vector product y = A @ x (row-major A)."},
    {"dgemm", (PyCFunction)py_dgemm, METH_VARARGS | METH_KEYWORDS,
        "dgemm(A, B, transa='N', transb='N') -> ndarray\n"
        "Matrix-matrix product C = op(A) @ op(B)."},
    {"dsyrk", py_dsyrk, METH_VARARGS,
        "dsyrk(X) -> ndarray\n"
        "Symmetric rank-k update: K = X @ X.T (float64)."},
    {"dsyr2k", py_dsyr2k, METH_VARARGS,
        "dsyr2k(C, A, B) -> ndarray\n"
        "Symmetric rank-2k update: C - A @ B.T - B @ A.T (float64)."},
    {"eigh", (PyCFunction)py_eigh, METH_VARARGS | METH_KEYWORDS,
        "eigh(K, inplace=False) -> (eigenvalues, eigenvectors)\n"
        "Compute all eigenvalues and eigenvectors of symmetric K.\n"
        "When inplace=False (default), K is scratch and a fresh eigenvector\n"
        "array is returned.  When inplace=True, K is overwritten with\n"
        "eigenvectors in-place (no separate N*N allocation)."},
    {"qr", py_qr, METH_VARARGS,
        "qr(A) -> (Q, R)\n"
        "Reduced QR factorization via vendor LAPACK dgeqrf + dorgqr.\n"
        "Q is (m, n), R is (n, n) upper triangular."},
    {"svd", (PyCFunction)py_svd, METH_VARARGS | METH_KEYWORDS,
        "svd(A, compute_uv=True) -> (U, s, Vh) or s\n"
        "Reduced SVD via vendor LAPACK dgesvd. Tall-skinny only (m >= n).\n"
        "U is (m, n), s is (n,), Vh is (n, n)."},
    {"set_n_threads", py_set_n_threads, METH_VARARGS,
        "set_n_threads(n) -> int\n"
        "Set jlinalg thread count (clamped to init max). Returns old count."},
    {"get_n_threads", py_get_n_threads, METH_NOARGS,
        "get_n_threads() -> int\n"
        "Get current jlinalg thread count."},
    {"compute_snp_stats_chunk", py_compute_snp_stats_chunk, METH_VARARGS,
        "compute_snp_stats_chunk(data, means, miss_counts, vars[, n_aa, n_ab, n_bb])\n"
        "Single-pass per-SNP statistics into pre-allocated output arrays."},
    {NULL, NULL, 0, NULL}
};

/* ---------------------------------------------------------------------------
 * Module definition
 * ---------------------------------------------------------------------------
 */
static struct PyModuleDef jlinalgmodule = {
    PyModuleDef_HEAD_INIT,
    "_jlinalg",   /* module name */
    NULL,       /* module docstring (brief description in __init__.py) */
    -1,         /* global state, no sub-interpreter support */
    JlinalgMethods
};

/* ---------------------------------------------------------------------------
 * PyInit__jlinalg — module initialiser
 * ---------------------------------------------------------------------------
 */
PyMODINIT_FUNC
PyInit__jlinalg(void)
{
    /* import_array() returns NULL on failure in numpy 2.x + Python 3.
     * _preflight_c_build() in hatch_build.py refuses to compile against
     * numpy 1.x, so the macro's return-on-failure is guaranteed. */
    import_array();

    /* Cache numpy.linalg.LinAlgError for eigh convergence errors.
     * Falls back to RuntimeError if numpy.linalg is unavailable. */
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

    /* Detect ISA and populate dispatch table */
    if (jlinalg_init() != 0) {
        PyErr_SetString(PyExc_ImportError,
            "_jlinalg: initialisation failed (ISA detection or dgemm workspace "
            "allocation — try reducing OMP_NUM_THREADS if memory-constrained)");
        return NULL;
    }

    PyObject *m = PyModule_Create(&jlinalgmodule);
    if (!m)
        return NULL;

    /* jlinalg_isa: active ISA string constant */
    PyObject *isa = PyUnicode_FromString(jlinalg_isa_name());
    if (!isa || PyModule_AddObject(m, "jlinalg_isa", isa) < 0) {
        Py_XDECREF(isa);
        Py_DECREF(m);
        return NULL;
    }

    /* blas_backend: identifies which dgemm backend is active */
    if (PyModule_AddStringConstant(m, "blas_backend", blas_backend_name()) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    /* HAS_OPENMP: True if compiled with OpenMP. Level 1/2 kernels are
     * single-threaded; dgemm (Level 3) uses OpenMP parallel-for. */
#ifdef _OPENMP
    int has_openmp = 1;
#else
    int has_openmp = 0;
#endif
    PyObject *openmp = PyBool_FromLong(has_openmp);
    if (!openmp || PyModule_AddObject(m, "HAS_OPENMP", openmp) < 0) {
        Py_XDECREF(openmp);
        Py_DECREF(m);
        return NULL;
    }

    /* ABI_VERSION: integer from jlinalg.h */
    PyObject *abi = PyLong_FromLong(JLINALG_ABI_VERSION);
    if (!abi || PyModule_AddObject(m, "ABI_VERSION", abi) < 0) {
        Py_XDECREF(abi);
        Py_DECREF(m);
        return NULL;
    }

    /* Blocking parameters: set by platform.c during jlinalg_init().
     * Exposed so Python tests can verify tile-skip counts analytically
     * without C instrumentation. */
    if (PyModule_AddIntConstant(m, "JLINALG_MR", JLINALG_MR) < 0) {
        Py_DECREF(m); return NULL;
    }
    if (PyModule_AddIntConstant(m, "JLINALG_NR", JLINALG_NR) < 0) {
        Py_DECREF(m); return NULL;
    }
    if (PyModule_AddIntConstant(m, "JLINALG_KC", JLINALG_KC) < 0) {
        Py_DECREF(m); return NULL;
    }
    if (PyModule_AddIntConstant(m, "JLINALG_MC", JLINALG_MC) < 0) {
        Py_DECREF(m); return NULL;
    }
    if (PyModule_AddIntConstant(m, "JLINALG_NC", JLINALG_NC) < 0) {
        Py_DECREF(m); return NULL;
    }

    /* blas_is_ilp64: 1 if external dgemm uses ILP64 (64-bit) integers, 0 otherwise */
    if (PyModule_AddIntConstant(m, "blas_is_ilp64", blas_is_ilp64()) < 0) {
        Py_DECREF(m); return NULL;
    }

    /* blas_has_dsyrk: 1 if vendor cblas_dsyrk is available, 0 otherwise */
    if (PyModule_AddIntConstant(m, "blas_has_dsyrk", blas_has_dsyrk()) < 0) {
        Py_DECREF(m); return NULL;
    }

    /* blas_has_dsyevd: 1 if vendor LAPACK dsyevd is available, 0 otherwise */
    if (PyModule_AddIntConstant(m, "blas_has_dsyevd", blas_has_dsyevd()) < 0) {
        Py_DECREF(m); return NULL;
    }

    /* blas_has_lapacke_dsyevd: 1 if LAPACKE C interface for dsyevd is available (MKL).
     * When true, dsyevd uses row-major LAPACKE — no eigenvector transpose needed. */
    if (PyModule_AddIntConstant(m, "blas_has_lapacke_dsyevd", blas_has_lapacke_dsyevd()) < 0) {
        Py_DECREF(m); return NULL;
    }

    /* blas_has_dsyevr: 1 if vendor LAPACK dsyevr is available (memory-pressure fallback). */
    if (PyModule_AddIntConstant(m, "blas_has_dsyevr", blas_has_dsyevr()) < 0) {
        Py_DECREF(m); return NULL;
    }
    /* blas_has_dgeqrf: 1 if vendor LAPACK dgeqrf + dorgqr available for QR. */
    if (PyModule_AddIntConstant(m, "blas_has_dgeqrf", blas_has_dgeqrf()) < 0) {
        Py_DECREF(m); return NULL;
    }
    /* blas_has_dgesvd: 1 if vendor LAPACK dgesvd available for SVD. */
    if (PyModule_AddIntConstant(m, "blas_has_dgesvd", blas_has_dgesvd()) < 0) {
        Py_DECREF(m); return NULL;
    }

    return m;
}
