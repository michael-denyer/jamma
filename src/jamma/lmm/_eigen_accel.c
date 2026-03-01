/*
 * _eigen_accel.c — C extension calling LAPACK DSYEVR for eigendecomposition.
 *
 * Exported function: eigh_dsyevr(K, uplo='L') -> (eigenvalues, eigenvectors)
 *
 * Uses DSYEVR (MRRR algorithm) instead of numpy's hardcoded DSYEVD
 * (divide-and-conquer). DSYEVR has O(N) workspace vs O(N^2) for DSYEVD,
 * saving ~250 GB at 125k samples.
 *
 * LAPACK resolution (runtime dlopen):
 *   At module init, discovers numpy's bundled LAPACK library via dlopen and
 *   resolves DSYEVR with dlsym. Tries dsyevr_64_ first (ILP64), then dsyevr_
 *   (LP64). This avoids link-time LAPACK dependencies, making the compiled
 *   extension portable across numpy builds (OpenBLAS, MKL, Accelerate).
 *
 *   ILP64: lapack_int = long long, symbol = dsyevr_64_
 *   LP64:  lapack_int = int,       symbol = dsyevr_
 *
 * Memory layout:
 *   Input K is a symmetric matrix (destroyed on return — DSYEVR uses it as
 *   workspace). C-contiguous (row-major) input has its UPLO convention
 *   transposed before passing to DSYEVR, because a C-row-major matrix is the
 *   transpose of a Fortran-column-major matrix (upper <-> lower).
 *   F-contiguous input uses the caller's UPLO as-is.
 *   Eigenvectors Z are returned as Fortran-order (column-major) arrays.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  /* Windows: not supported */
  #error "_eigen_accel does not support Windows"
#else
  #include <dlfcn.h>    /* dlopen, dlsym, dlclose */
  #include <dirent.h>   /* opendir, readdir */
#endif

/* ABI version: bump when function signatures or array layout expectations change. */
#define ABI_VERSION 2

/* ---------------------------------------------------------------------------
 * DSYEVR function pointer type.
 *
 * We define two variants: one for LP64 (int) and one for ILP64 (long long).
 * At runtime, we resolve the correct symbol and store the function pointer.
 * The actual call dispatches through the appropriate pointer based on
 * the detected integer width.
 * ------------------------------------------------------------------------- */
typedef void (*dsyevr_lp64_fn)(
    const char *jobz, const char *range, const char *uplo,
    const int *n, double *a, const int *lda,
    const double *vl, const double *vu,
    const int *il, const int *iu,
    const double *abstol, int *m,
    double *w, double *z, const int *ldz,
    int *isuppz,
    double *work, const int *lwork,
    int *iwork, const int *liwork,
    int *info);

typedef void (*dsyevr_ilp64_fn)(
    const char *jobz, const char *range, const char *uplo,
    const long long *n, double *a, const long long *lda,
    const double *vl, const double *vu,
    const long long *il, const long long *iu,
    const double *abstol, long long *m,
    double *w, double *z, const long long *ldz,
    long long *isuppz,
    double *work, const long long *lwork,
    long long *iwork, const long long *liwork,
    long long *info);

/* Global state: resolved at module init */
static int g_is_ilp64 = 0;
static dsyevr_lp64_fn g_dsyevr_lp64 = NULL;
static dsyevr_ilp64_fn g_dsyevr_ilp64 = NULL;
static void *g_lapack_handle = NULL;  /* dlopen handle */

/* ---------------------------------------------------------------------------
 * LAPACK library discovery — find and dlopen numpy's bundled LAPACK.
 *
 * Search strategy:
 *   1. macOS: try RTLD_DEFAULT first (Accelerate is globally available)
 *   2. Linux: scan numpy.libs/ and numpy/_core/.libs/ for openblas/mkl
 *   3. Fallback: try RTLD_DEFAULT (system LAPACK)
 * ------------------------------------------------------------------------- */

/* Try to resolve dsyevr from a dlopen handle (or RTLD_DEFAULT).
 * Returns 1 if found, 0 if not. Sets g_is_ilp64 and the function pointer. */
static int try_resolve_dsyevr(void *handle) {
    void *sym;

    /* Try ILP64 first (dsyevr_64_) */
    sym = dlsym(handle, "dsyevr_64_");
    if (sym) {
        g_dsyevr_ilp64 = (dsyevr_ilp64_fn)sym;
        g_is_ilp64 = 1;
        return 1;
    }

    /* Try LP64 (dsyevr_) */
    sym = dlsym(handle, "dsyevr_");
    if (sym) {
        g_dsyevr_lp64 = (dsyevr_lp64_fn)sym;
        g_is_ilp64 = 0;
        return 1;
    }

    return 0;
}

/* Scan a directory for LAPACK-providing shared libraries and try to dlopen them.
 * Returns 1 if DSYEVR was resolved, 0 if not. */
static int scan_dir_for_lapack(const char *dirpath) {
    DIR *dir = opendir(dirpath);
    if (!dir) return 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Look for openblas or mkl shared libraries */
        if (strstr(entry->d_name, "openblas") || strstr(entry->d_name, "libmkl")) {
            /* Must be a .so or .dylib */
            if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib"))
                continue;

            char fullpath[4096];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);

            void *handle = dlopen(fullpath, RTLD_LAZY | RTLD_LOCAL);
            if (!handle) continue;

            if (try_resolve_dsyevr(handle)) {
                g_lapack_handle = handle;
                closedir(dir);
                return 1;
            }
            dlclose(handle);
        }
    }
    closedir(dir);
    return 0;
}

/* Main LAPACK discovery function. Called once at module init.
 * Returns 1 if DSYEVR was resolved, 0 if not. */
static int discover_lapack(void) {
    /* 1. Try process-global symbols first (macOS Accelerate, LD_PRELOAD, etc.) */
    if (try_resolve_dsyevr(RTLD_DEFAULT)) {
        return 1;
    }

    /* 2. Find numpy's package directory and scan its bundled libs */
    PyObject *numpy_mod = PyImport_ImportModule("numpy");
    if (numpy_mod) {
        PyObject *numpy_file = PyObject_GetAttrString(numpy_mod, "__file__");
        if (numpy_file) {
            const char *numpy_path = PyUnicode_AsUTF8(numpy_file);
            if (numpy_path) {
                /* numpy_path is like /path/to/numpy/__init__.py
                 * We need /path/to/numpy/.libs/ and /path/to/numpy.libs/ */
                char basedir[4096];
                strncpy(basedir, numpy_path, sizeof(basedir) - 1);
                basedir[sizeof(basedir) - 1] = '\0';

                /* Strip __init__.py to get numpy package dir */
                char *slash = strrchr(basedir, '/');
                if (slash) {
                    *slash = '\0';  /* basedir = /path/to/numpy */

                    /* Try numpy/.libs/ (older numpy layout) */
                    char libdir[4096];
                    snprintf(libdir, sizeof(libdir), "%s/.libs", basedir);
                    if (scan_dir_for_lapack(libdir)) {
                        Py_DECREF(numpy_file);
                        Py_DECREF(numpy_mod);
                        return 1;
                    }

                    /* Try numpy/_core/.libs/ */
                    snprintf(libdir, sizeof(libdir), "%s/_core/.libs", basedir);
                    if (scan_dir_for_lapack(libdir)) {
                        Py_DECREF(numpy_file);
                        Py_DECREF(numpy_mod);
                        return 1;
                    }

                    /* Try sibling numpy.libs/ (pip wheel layout) */
                    char *parent_slash = strrchr(basedir, '/');
                    if (parent_slash) {
                        /* basedir was /path/to/numpy, need /path/to/numpy.libs */
                        snprintf(libdir, sizeof(libdir), "%s/numpy.libs",
                                 basedir);  /* This gives /path/to/numpy/numpy.libs — wrong */
                        /* Actually need parent + numpy.libs */
                        char parentdir[4096];
                        strncpy(parentdir, basedir, sizeof(parentdir) - 1);
                        parentdir[sizeof(parentdir) - 1] = '\0';
                        char *p = strrchr(parentdir, '/');
                        if (p) {
                            *p = '\0';  /* parentdir = /path/to */
                            snprintf(libdir, sizeof(libdir), "%s/numpy.libs", parentdir);
                            if (scan_dir_for_lapack(libdir)) {
                                Py_DECREF(numpy_file);
                                Py_DECREF(numpy_mod);
                                return 1;
                            }
                        }
                    }
                }
            }
            Py_DECREF(numpy_file);
        }
        Py_DECREF(numpy_mod);
    } else {
        PyErr_Clear();
    }

    /* 3. Last resort: try RTLD_DEFAULT again (in case numpy loaded BLAS
     * between our first attempt and now — shouldn't happen but costs nothing) */
    return 0;
}


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

    /* Check that DSYEVR was resolved at init time */
    if (!g_dsyevr_lp64 && !g_dsyevr_ilp64) {
        PyErr_SetString(PyExc_RuntimeError,
            "DSYEVR symbol not resolved. numpy's LAPACK library could not be found. "
            "Ensure numpy is installed and its bundled BLAS/LAPACK is accessible.");
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

    /* Validate uplo argument */
    if (*uplo_arg != 'L' && *uplo_arg != 'l' && *uplo_arg != 'U' && *uplo_arg != 'u') {
        PyErr_Format(PyExc_ValueError,
            "uplo must be 'L' or 'U', got '%c'", *uplo_arg);
        return NULL;
    }

    /* LP64 overflow guard */
    if (!g_is_ilp64) {
        int ln_test = (int)n;
        if ((npy_intp)ln_test != n) {
            PyErr_Format(PyExc_OverflowError,
                "Matrix dimension %ld exceeds LP64 LAPACK int32 limit (~46,340). "
                "Install ILP64 numpy for large matrices: pip install numpy "
                "--extra-index-url https://michael-denyer.github.io/numpy-mkl "
                "--force-reinstall",
                (long)n);
            return NULL;
        }
    }

    /* LAPACK uses Fortran conventions (column-major).
     * A C-contiguous (row-major) symmetric matrix passed as-is looks like
     * its own transpose to LAPACK. For a symmetric matrix A = A^T, the
     * upper triangle in C row-major == lower triangle in Fortran column-major.
     * So: C-contiguous input → UPLO='U' to DSYEVR.
     *     F-contiguous input → use the uplo_arg as given (default 'L'). */
    char uplo_char;
    if (is_c) {
        uplo_char = (*uplo_arg == 'L' || *uplo_arg == 'l') ? 'U' : 'L';
    } else {
        uplo_char = (*uplo_arg == 'L' || *uplo_arg == 'l') ? 'L' : 'U';
    }

    double *a_data = (double *)PyArray_DATA(K_arr);

    /* ---- Allocate output arrays ------------------------------------------ */
    npy_intp n_dim = n;
    PyArrayObject *eigenvalues = (PyArrayObject *)PyArray_SimpleNew(1, &n_dim, NPY_FLOAT64);
    if (!eigenvalues) return NULL;

    npy_intp ev_dims[2] = {n, n};
    PyArrayObject *eigenvectors = (PyArrayObject *)PyArray_EMPTY(2, ev_dims, NPY_FLOAT64, 1 /* fortran */);
    if (!eigenvectors) {
        Py_DECREF(eigenvalues);
        return NULL;
    }

    double *w_data = (double *)PyArray_DATA(eigenvalues);
    double *z_data = (double *)PyArray_DATA(eigenvectors);

    /* ---- Dispatch to LP64 or ILP64 -------------------------------------- */
    const char jobz = 'V';
    const char range = 'A';
    int info_result = 0;

    if (g_is_ilp64) {
        /* ILP64 path: long long integers */
        long long ln = (long long)n;
        long long lda = ln;
        long long m = 0;
        long long info = 0;
        const long long il = 0, iu = 0;
        const double vl = 0.0, vu = 0.0, abstol = 0.0;

        long long *isuppz = (long long *)malloc(2 * (size_t)n * sizeof(long long));
        if (!isuppz) { Py_DECREF(eigenvalues); Py_DECREF(eigenvectors); return PyErr_NoMemory(); }

        /* Workspace query */
        double work_query = 0.0;
        long long lwork_q = -1;
        long long iwork_query = 0;
        long long liwork_q = -1;

        Py_BEGIN_ALLOW_THREADS
        g_dsyevr_ilp64(&jobz, &range, &uplo_char,
            &ln, a_data, &lda, &vl, &vu, &il, &iu,
            &abstol, &m, w_data, z_data, &ln, isuppz,
            &work_query, &lwork_q, &iwork_query, &liwork_q, &info);
        Py_END_ALLOW_THREADS

        if (info != 0) {
            free(isuppz); Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
            if (info < 0)
                PyErr_Format(PyExc_ValueError, "DSYEVR workspace query: parameter %lld is invalid", -info);
            else
                PyErr_SetString(PyExc_RuntimeError, "DSYEVR workspace query failed");
            return NULL;
        }

        long long lwork = (long long)ceil(work_query);
        long long liwork = iwork_query;

        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        long long *iwork = (long long *)malloc((size_t)liwork * sizeof(long long));
        if (!work || !iwork) {
            free(work); free(iwork); free(isuppz);
            Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
            return PyErr_NoMemory();
        }

        info = 0;
        Py_BEGIN_ALLOW_THREADS
        g_dsyevr_ilp64(&jobz, &range, &uplo_char,
            &ln, a_data, &lda, &vl, &vu, &il, &iu,
            &abstol, &m, w_data, z_data, &ln, isuppz,
            work, &lwork, iwork, &liwork, &info);
        Py_END_ALLOW_THREADS

        free(work); free(iwork); free(isuppz);
        info_result = (int)info;

        if (info_result == 0 && m != ln) {
            Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
            PyErr_Format(PyExc_RuntimeError,
                "DSYEVR returned %lld eigenvalues, expected %lld", m, ln);
            return NULL;
        }
    } else {
        /* LP64 path: int integers */
        int ln = (int)n;
        int lda = ln;
        int m = 0;
        int info = 0;
        const int il = 0, iu = 0;
        const double vl = 0.0, vu = 0.0, abstol = 0.0;

        int *isuppz = (int *)malloc(2 * (size_t)n * sizeof(int));
        if (!isuppz) { Py_DECREF(eigenvalues); Py_DECREF(eigenvectors); return PyErr_NoMemory(); }

        /* Workspace query */
        double work_query = 0.0;
        int lwork_q = -1;
        int iwork_query = 0;
        int liwork_q = -1;

        Py_BEGIN_ALLOW_THREADS
        g_dsyevr_lp64(&jobz, &range, &uplo_char,
            &ln, a_data, &lda, &vl, &vu, &il, &iu,
            &abstol, &m, w_data, z_data, &ln, isuppz,
            &work_query, &lwork_q, &iwork_query, &liwork_q, &info);
        Py_END_ALLOW_THREADS

        if (info != 0) {
            free(isuppz); Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
            if (info < 0)
                PyErr_Format(PyExc_ValueError, "DSYEVR workspace query: parameter %d is invalid", -info);
            else
                PyErr_SetString(PyExc_RuntimeError, "DSYEVR workspace query failed");
            return NULL;
        }

        int lwork = (int)ceil(work_query);
        int liwork = iwork_query;

        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        int *iwork = (int *)malloc((size_t)liwork * sizeof(int));
        if (!work || !iwork) {
            free(work); free(iwork); free(isuppz);
            Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
            return PyErr_NoMemory();
        }

        info = 0;
        Py_BEGIN_ALLOW_THREADS
        g_dsyevr_lp64(&jobz, &range, &uplo_char,
            &ln, a_data, &lda, &vl, &vu, &il, &iu,
            &abstol, &m, w_data, z_data, &ln, isuppz,
            work, &lwork, iwork, &liwork, &info);
        Py_END_ALLOW_THREADS

        free(work); free(iwork); free(isuppz);
        info_result = info;

        if (info_result == 0 && m != ln) {
            Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
            PyErr_Format(PyExc_RuntimeError,
                "DSYEVR returned %d eigenvalues, expected %d", m, ln);
            return NULL;
        }
    }

    /* ---- Check results ---------------------------------------------------- */
    if (info_result < 0) {
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_Format(PyExc_ValueError,
            "DSYEVR parameter %d is invalid", -info_result);
        return NULL;
    }
    if (info_result > 0) {
        Py_DECREF(eigenvalues);
        Py_DECREF(eigenvectors);
        PyErr_SetString(PyExc_RuntimeError, "DSYEVR failed to converge");
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
        "saving ~250 GB at 125k samples.\n"
        "\n"
        "LAPACK is resolved at runtime from numpy's bundled BLAS/LAPACK via dlopen.\n"
        "No compile-time LAPACK dependency — works with OpenBLAS, MKL, Accelerate.\n"
        "ILP64 (dsyevr_64_) is auto-detected; LP64 (dsyevr_) used as fallback.\n"
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
        "    RuntimeError: DSYEVR convergence failure, unexpected partial spectrum,\n"
        "                  or LAPACK symbol not resolved.\n"
        "    MemoryError:  workspace allocation failure.\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_eigen_accel",
    "C extension: LAPACK DSYEVR eigendecomposition with O(N) workspace.\n"
    "LAPACK resolved at runtime via dlopen — no link-time dependency.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__eigen_accel(void)
{
    import_array();

    /* Discover LAPACK and resolve DSYEVR before creating the module.
     * If discovery fails, we still create the module but eigh_dsyevr()
     * will raise RuntimeError when called. This allows introspection
     * (ABI_VERSION, IS_ILP64) even when LAPACK is unavailable. */
    int lapack_found = discover_lapack();

    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    /* Expose runtime ILP64 detection to Python */
    if (PyModule_AddIntConstant(m, "IS_ILP64", g_is_ilp64) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (!lapack_found) {
        /* Module loads but eigh_dsyevr will raise RuntimeError */
        if (PyErr_WarnEx(PyExc_RuntimeWarning,
                "_eigen_accel: DSYEVR symbol not found in numpy's LAPACK. "
                "eigh_dsyevr() will not be available. "
                "Falling back to numpy.linalg.eigh (DSYEVD).", 1) < 0) {
            Py_DECREF(m);
            return NULL;
        }
    }

    return m;
}
