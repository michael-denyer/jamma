/**
 * pymodule.c — Python C extension module _jblas.
 *
 * Exposes the six jblas BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv,
 * dgemm) to Python via the NumPy buffer protocol.  Arrays are accessed via
 * PyArray_FROM_OTF for contiguous double* extraction (copies non-contiguous
 * or non-float64 inputs as needed).
 *
 * Module-level constants:
 *   jblas_isa   — active ISA string ("AVX2", "NEON", or "generic")
 *   HAS_OPENMP  — True if compiled with OpenMP (-fopenmp)
 *   ABI_VERSION — integer (JBLAS_ABI_VERSION from jblas.h)
 *
 * Exported functions: ddot, dnrm2, daxpy, dscal, dgemv, dgemm
 *
 * Patterns follow _lmm_accel.c: PyArray_FROM_OTF with NPY_ARRAY_IN_ARRAY for
 * read-only inputs, NPY_ARRAY_INOUT_ARRAY2 for in-place writeable outputs.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

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
    double result = jblas_dispatch.ddot(n, px, 1, py, 1);

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
    double result = jblas_dispatch.dnrm2(n, px, 1);

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
    jblas_dispatch.daxpy(n, alpha, px, 1, py, 1);

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
    jblas_dispatch.dscal(n, alpha, px, 1);

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

    jblas_dispatch.dgemv(m, n, pA, px, py);

    Py_DECREF(aA);
    Py_DECREF(ax);
    return (PyObject *)ay;
}

/* ---------------------------------------------------------------------------
 * py_dgemm — matrix-matrix product C = op(A) @ op(B)
 *
 * Signature: dgemm(A, B, transa='N', transb='N') -> ndarray
 * A and B are 2-D float64 arrays.  transa/transb: 'N' (no transpose) or
 * 'T' (transpose).  Returns a new 2-D float64 array.
 * ---------------------------------------------------------------------------
 */
static PyObject *
py_dgemm(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"A", "B", "transa", "transb", NULL};
    PyObject *oA, *oB;
    const char *transa_str = "N";
    const char *transb_str = "N";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ss", kwlist,
            &oA, &oB, &transa_str, &transb_str))
        return NULL;

    /* Validate transpose flags: only 'N'/'n' and 'T'/'t' are accepted. */
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

    /* Allocate output C (M x N), zero-initialised by jblas_dgemm_c */
    npy_intp dims[2] = {M, N};
    PyArrayObject *aC = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!aC) {
        Py_DECREF(aA); Py_DECREF(aB);
        return NULL;
    }

    const double *pA = (const double *)PyArray_DATA(aA);
    const double *pB = (const double *)PyArray_DATA(aB);
    double       *pC = (double *)PyArray_DATA(aC);

    /* Guard: if dgemm workspace allocation failed during init, the packed
     * buffers are NULL and jblas_dgemm_c would segfault.  Raise instead. */
    if (!jblas_packed_A || !jblas_packed_B) {
        PyErr_SetString(PyExc_RuntimeError,
            "dgemm: workspace allocation failed during jblas init; "
            "reduce OMP_NUM_THREADS or use the numpy fallback");
        Py_DECREF(aA); Py_DECREF(aB); Py_DECREF(aC);
        return NULL;
    }

    /* Leading dimensions are the physical column counts (row-major storage) */
    npy_intp lda = PyArray_DIM(aA, 1);
    npy_intp ldb = PyArray_DIM(aB, 1);

    /* Release the GIL for the O(N^3) C/OpenMP computation.  Safe because
     * jblas_dgemm_c operates purely on C double arrays; the PyArray refs
     * (aA, aB, aC) keep the buffers alive for the duration. */
    Py_BEGIN_ALLOW_THREADS
    jblas_dgemm_c(M, N, K_a, pA, lda, pB, ldb, pC, N, transa, transb);
    Py_END_ALLOW_THREADS

    Py_DECREF(aA);
    Py_DECREF(aB);
    return (PyObject *)aC;
}

/* ---------------------------------------------------------------------------
 * Method table
 * ---------------------------------------------------------------------------
 */
static PyMethodDef JblasMethods[] = {
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
    {NULL, NULL, 0, NULL}
};

/* ---------------------------------------------------------------------------
 * Module definition
 * ---------------------------------------------------------------------------
 */
static struct PyModuleDef jblasmodule = {
    PyModuleDef_HEAD_INIT,
    "_jblas",   /* module name */
    NULL,       /* module docstring (brief description in __init__.py) */
    -1,         /* global state, no sub-interpreter support */
    JblasMethods
};

/* ---------------------------------------------------------------------------
 * PyInit__jblas — module initialiser
 * ---------------------------------------------------------------------------
 */
PyMODINIT_FUNC
PyInit__jblas(void)
{
    /* import_array() returns NULL on failure in numpy 2.x + Python 3.
     * _preflight_c_build() in hatch_build.py refuses to compile against
     * numpy 1.x, so the macro's return-on-failure is guaranteed. */
    import_array();

    /* Detect ISA and populate dispatch table */
    if (jblas_init() != 0) {
        PyErr_SetString(PyExc_ImportError,
            "_jblas: initialisation failed (ISA detection or dgemm workspace "
            "allocation — try reducing OMP_NUM_THREADS if memory-constrained)");
        return NULL;
    }

    PyObject *m = PyModule_Create(&jblasmodule);
    if (!m)
        return NULL;

    /* jblas_isa: active ISA string constant */
    PyObject *isa = PyUnicode_FromString(jblas_isa_name());
    if (!isa || PyModule_AddObject(m, "jblas_isa", isa) < 0) {
        Py_XDECREF(isa);
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

    /* ABI_VERSION: integer from jblas.h */
    PyObject *abi = PyLong_FromLong(JBLAS_ABI_VERSION);
    if (!abi || PyModule_AddObject(m, "ABI_VERSION", abi) < 0) {
        Py_XDECREF(abi);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
