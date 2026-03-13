/**
 * pymodule.c — Python C extension module _jblas.
 *
 * Exposes the five jblas BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv)
 * to Python via the NumPy buffer protocol.  Arrays are accessed via
 * PyArray_FROM_OTF for zero-copy double* extraction where possible.
 *
 * Module-level constants:
 *   jblas_isa   — active ISA string ("AVX2", "NEON", or "generic")
 *   HAS_OPENMP  — True if compiled with OpenMP (-fopenmp)
 *   ABI_VERSION — integer (JBLAS_ABI_VERSION from jblas.h)
 *
 * Exported functions: ddot, dnrm2, daxpy, dscal, dgemv
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
    -1,         /* per-interpreter module state (-1 = global state) */
    JblasMethods
};

/* ---------------------------------------------------------------------------
 * PyInit__jblas — module initialiser
 * ---------------------------------------------------------------------------
 */
PyMODINIT_FUNC
PyInit__jblas(void)
{
    /* Must be called before any NumPy C API usage */
    import_array();

    /* Detect ISA and populate dispatch table */
    jblas_init();

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

    /* HAS_OPENMP: True if compiled with OpenMP */
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
