/*
 * _eigen_accel.c — C extension calling LAPACK DSYEVR for eigendecomposition.
 *
 * Exported function: eigh_dsyevr(K, uplo='L') -> (eigenvalues, eigenvectors)
 *
 * Uses DSYEVR (MRRR algorithm) instead of numpy's hardcoded DSYEVD
 * (divide-and-conquer). DSYEVR has O(N) workspace vs O(N^2) for DSYEVD,
 * saving ~232 GB at 125k samples.
 *
 * ILP64 / LP64 selection:
 *   JAMMA_ILP64 defined at compile time for ILP64 MKL builds.
 *   ILP64: lapack_int = long long, DSYEVR symbol = dsyevr_64_
 *   LP64:  lapack_int = int,       DSYEVR symbol = dsyevr_
 *
 * Memory layout:
 *   Input K is a symmetric matrix (destroyed on return — DSYEVR uses it as
 *   workspace). C-contiguous (row-major) input is passed with UPLO='U',
 *   because a C-row-major upper triangle is the same as Fortran-column-major
 *   lower triangle. F-contiguous input is passed with UPLO='L'.
 *   Eigenvectors Z are returned as Fortran-order (column-major) arrays.
 *
 * ABI version: bump when function signatures or array layout expectations change.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

/* ABI version: bump when function signatures or array layout expectations change. */
#define ABI_VERSION 1

/* ILP64 / LP64 selection */
#ifdef JAMMA_ILP64
  typedef long long lapack_int;
  #define DSYEVR_FUNC dsyevr_64_
#else
  typedef int lapack_int;
  #define DSYEVR_FUNC dsyevr_
#endif

/* ---------------------------------------------------------------------------
 * DSYEVR extern declaration — link-time resolution against numpy's LAPACK.
 *
 * DSYEVR computes eigenvalues and eigenvectors of a real symmetric matrix
 * using the MRRR (Multiple Relatively Robust Representations) algorithm.
 *
 * Arguments (see LAPACK documentation for DSYEVR):
 *   jobz:   'N' (eigenvalues only) or 'V' (eigenvalues + eigenvectors)
 *   range:  'A' (all), 'V' (value range [vl,vu]), 'I' (index range [il,iu])
 *   uplo:   'U' (upper triangle stored) or 'L' (lower triangle stored)
 *   n:      matrix order
 *   a:      input matrix (n x n), overwritten on exit
 *   lda:    leading dimension of a
 *   vl, vu: value range (only for range='V')
 *   il, iu: index range (only for range='I')
 *   abstol: absolute tolerance (0.0 = LAPACK default)
 *   m:      number of eigenvalues found (output)
 *   w:      eigenvalues array (m elements, ascending order)
 *   z:      eigenvectors matrix (ldz x m)
 *   ldz:    leading dimension of z
 *   isuppz: support array (2*m elements)
 *   work:   workspace (lwork elements); lwork=-1 → workspace query
 *   lwork:  workspace size; -1 → query
 *   iwork:  integer workspace (liwork elements)
 *   liwork: integer workspace size; -1 → query
 *   info:   output status (0=success, <0=parameter error, >0=convergence fail)
 * ------------------------------------------------------------------------- */
extern void DSYEVR_FUNC(
    const char *jobz, const char *range, const char *uplo,
    const lapack_int *n, double *a, const lapack_int *lda,
    const double *vl, const double *vu,
    const lapack_int *il, const lapack_int *iu,
    const double *abstol, lapack_int *m,
    double *w, double *z, const lapack_int *ldz,
    lapack_int *isuppz,
    double *work, const lapack_int *lwork,
    lapack_int *iwork, const lapack_int *liwork,
    lapack_int *info);

/* ---------------------------------------------------------------------------
 * py_eigh_dsyevr — Python-callable wrapper for DSYEVR.
 *
 * Args:
 *   K:    float64 ndarray, 2D, square, writeable (destroyed on return)
 *   uplo: str, 'U' or 'L' (default 'L')
 *
 * Returns:
 *   (eigenvalues, eigenvectors) — both float64 ndarrays.
 *   eigenvalues shape: (n,), ascending order.
 *   eigenvectors shape: (n, n), Fortran-contiguous (column-major).
 * ------------------------------------------------------------------------- */
static PyObject *py_eigh_dsyevr(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"K", "uplo", NULL};

    PyArrayObject *K_arr = NULL;
    const char *uplo_arg = "L";

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!|s", kwlist,
            &PyArray_Type, &K_arr, &uplo_arg)) {
        return NULL;
    }

    /* ---- Validate input -------------------------------------------------- */
    if (PyArray_NDIM(K_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "K must be a 2D array");
        return NULL;
    }

    npy_intp n = PyArray_DIM(K_arr, 0);
    if (PyArray_DIM(K_arr, 1) != n) {
        PyErr_Format(PyExc_ValueError,
            "K must be square, got (%ld, %ld)",
            (long)n, (long)PyArray_DIM(K_arr, 1));
        return NULL;
    }

    if (PyArray_TYPE(K_arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "K must be float64");
        return NULL;
    }

    if (!PyArray_ISWRITEABLE(K_arr)) {
        PyErr_SetString(PyExc_ValueError, "K must be writeable (it is overwritten by DSYEVR)");
        return NULL;
    }

    /* Require contiguous — either C or F order */
    int is_c = PyArray_IS_C_CONTIGUOUS(K_arr);
    int is_f = PyArray_IS_F_CONTIGUOUS(K_arr);
    if (!is_c && !is_f) {
        PyErr_SetString(PyExc_ValueError,
            "K must be contiguous (C or Fortran order). "
            "Use np.ascontiguousarray(K) or np.asfortranarray(K).");
        return NULL;
    }

    if (n == 0) {
        /* Empty matrix: return empty arrays */
        npy_intp zero = 0;
        PyObject *w_empty = PyArray_SimpleNew(1, &zero, NPY_FLOAT64);
        npy_intp dims2[2] = {0, 0};
        PyObject *z_empty = PyArray_SimpleNew(2, dims2, NPY_FLOAT64);
        if (!w_empty || !z_empty) {
            Py_XDECREF(w_empty);
            Py_XDECREF(z_empty);
            return NULL;
        }
        PyObject *result = Py_BuildValue("(OO)", w_empty, z_empty);
        Py_DECREF(w_empty);
        Py_DECREF(z_empty);
        return result;
    }

    /* LAPACK uses Fortran conventions (column-major).
     * A C-contiguous (row-major) symmetric matrix passed as-is looks like
     * its own transpose to LAPACK. For a symmetric matrix A = A^T, the
     * upper triangle in C row-major == lower triangle in Fortran column-major.
     * So: C-contiguous input → UPLO='U' to DSYEVR.
     *     F-contiguous input → use the uplo_arg as given (default 'L'). */
    char uplo_char;
    if (is_c) {
        /* Transpose the uplo convention for Fortran-column-major interpretation */
        uplo_char = (*uplo_arg == 'L' || *uplo_arg == 'l') ? 'U' : 'L';
    } else {
        uplo_char = (*uplo_arg == 'L' || *uplo_arg == 'l') ? 'L' : 'U';
    }

    double *a_data = (double *)PyArray_DATA(K_arr);

    lapack_int ln = (lapack_int)n;
    lapack_int lda = ln;  /* leading dimension == n (square, contiguous) */

    /* ---- Allocate output arrays ------------------------------------------ */
    /* eigenvalues: 1D shape (n,) */
    npy_intp n_dim = n;
    PyArrayObject *eigenvalues = (PyArrayObject *)PyArray_SimpleNew(1, &n_dim, NPY_FLOAT64);
    if (!eigenvalues) return NULL;

    /* eigenvectors: 2D shape (n, n), Fortran-contiguous (column-major).
     * DSYEVR writes eigenvectors column-by-column, so F-order is natural. */
    npy_intp ev_dims[2] = {n, n};
    PyArrayObject *eigenvectors = (PyArrayObject *)PyArray_EMPTY(2, ev_dims, NPY_FLOAT64, 1 /* fortran */);
    if (!eigenvectors) {
        Py_DECREF(eigenvalues);
        return NULL;
    }

    double *w_data = (double *)PyArray_DATA(eigenvalues);
    double *z_data = (double *)PyArray_DATA(eigenvectors);

    /* isuppz: support array — at least 2*n elements */
    lapack_int *isuppz = (lapack_int *)malloc(2 * (size_t)n * sizeof(lapack_int));
    if (!isuppz) {
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_NoMemory();
        return NULL;
    }

    /* ---- Workspace query -------------------------------------------------- */
    const char jobz = 'V';    /* compute eigenvalues and eigenvectors */
    const char range = 'A';   /* full spectrum */
    const double vl = 0.0, vu = 0.0;  /* unused (range='A') */
    const lapack_int il = 0, iu = 0;  /* unused (range='A') */
    const double abstol = 0.0;  /* LAPACK default tolerance */
    lapack_int m = 0;   /* number of eigenvalues found (output) */
    lapack_int info = 0;

    /* Workspace query: LWORK=-1 causes DSYEVR to return optimal LWORK in work[0] */
    double work_query = 0.0;
    lapack_int lwork_q = -1;
    lapack_int iwork_query = 0;
    lapack_int liwork_q = -1;

    Py_BEGIN_ALLOW_THREADS
    DSYEVR_FUNC(
        &jobz, &range, &uplo_char,
        &ln, a_data, &lda,
        &vl, &vu, &il, &iu,
        &abstol, &m,
        w_data, z_data, &ln,
        isuppz,
        &work_query, &lwork_q,
        &iwork_query, &liwork_q,
        &info);
    Py_END_ALLOW_THREADS

    if (info != 0) {
        free(isuppz);
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        if (info < 0) {
            PyErr_Format(PyExc_ValueError,
                "DSYEVR workspace query: parameter %ld is invalid", (long)(-info));
        } else {
            PyErr_SetString(PyExc_RuntimeError,
                "DSYEVR workspace query failed unexpectedly");
        }
        return NULL;
    }

    lapack_int lwork = (lapack_int)work_query;
    lapack_int liwork = iwork_query;

    /* ---- Allocate workspace ---------------------------------------------- */
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    lapack_int *iwork = (lapack_int *)malloc((size_t)liwork * sizeof(lapack_int));

    if (!work || !iwork) {
        free(work);
        free(iwork);
        free(isuppz);
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_NoMemory();
        return NULL;
    }

    /* ---- Compute eigendecomposition -------------------------------------- */
    info = 0;

    Py_BEGIN_ALLOW_THREADS
    DSYEVR_FUNC(
        &jobz, &range, &uplo_char,
        &ln, a_data, &lda,
        &vl, &vu, &il, &iu,
        &abstol, &m,
        w_data, z_data, &ln,
        isuppz,
        work, &lwork,
        iwork, &liwork,
        &info);
    Py_END_ALLOW_THREADS

    /* ---- Cleanup workspace ----------------------------------------------- */
    free(work);
    free(iwork);
    free(isuppz);

    /* ---- Check results ---------------------------------------------------- */
    if (info < 0) {
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_Format(PyExc_ValueError,
            "DSYEVR parameter %ld is invalid", (long)(-info));
        return NULL;
    }
    if (info > 0) {
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_SetString(PyExc_RuntimeError,
            "DSYEVR failed to converge");
        return NULL;
    }
    if (m != ln) {
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_Format(PyExc_RuntimeError,
            "DSYEVR returned %ld eigenvalues, expected %ld",
            (long)m, (long)ln);
        return NULL;
    }

    /* ---- Return (eigenvalues, eigenvectors) ------------------------------ */
    PyObject *result = Py_BuildValue("(OO)",
        (PyObject *)eigenvalues, (PyObject *)eigenvectors);
    Py_DECREF(eigenvalues);
    Py_DECREF(eigenvectors);
    return result;
}

/* ---------------------------------------------------------------------------
 * Module definition
 * ------------------------------------------------------------------------- */
static PyMethodDef methods[] = {
    {
        "eigh_dsyevr",
        (PyCFunction)py_eigh_dsyevr,
        METH_VARARGS | METH_KEYWORDS,
        "eigh_dsyevr(K, uplo='L') -> (eigenvalues, eigenvectors)\n"
        "\n"
        "Eigendecompose a real symmetric matrix using LAPACK DSYEVR (MRRR algorithm).\n"
        "\n"
        "DSYEVR has O(N) workspace vs O(N^2) for numpy.linalg.eigh (DSYEVD),\n"
        "saving ~232 GB at 125k samples.\n"
        "\n"
        "Args:\n"
        "    K:    float64 ndarray, 2D, square, writeable, C or Fortran contiguous.\n"
        "          WARNING: K is overwritten and destroyed on return.\n"
        "    uplo: 'U' (upper triangle) or 'L' (lower triangle). Default 'L'.\n"
        "          Note: for C-contiguous K, uplo convention is automatically\n"
        "          transposed to account for Fortran column-major layout.\n"
        "\n"
        "Returns:\n"
        "    (eigenvalues, eigenvectors) tuple:\n"
        "    eigenvalues:   float64 ndarray, shape (n,), ascending order.\n"
        "    eigenvectors:  float64 ndarray, shape (n, n), Fortran-contiguous.\n"
        "                   Column j is the eigenvector for eigenvalues[j].\n"
        "\n"
        "Raises:\n"
        "    ValueError:   K is not square, not float64, not writeable,\n"
        "                  or DSYEVR detects invalid parameters.\n"
        "    RuntimeError: DSYEVR convergence failure, or unexpected partial spectrum.\n"
        "    MemoryError:  workspace allocation failure.\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_eigen_accel",
    "C extension: LAPACK DSYEVR eigendecomposition with O(N) workspace.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__eigen_accel(void)
{
    import_array();  /* returns NULL on failure (NumPy Python 3 macro) */
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    /* ABI version — Python side checks this to detect stale .so files */
    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
