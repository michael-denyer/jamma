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

/* Debug flag: set EIGEN_ACCEL_DEBUG=1 to enable discovery diagnostics on stderr */
static int debug_enabled(void) {
    const char *val = getenv("EIGEN_ACCEL_DEBUG");
    return val && val[0] == '1';
}

/* Symbol names to try for DSYEVR, in priority order.
 *
 * Naming conventions across LAPACK providers:
 *   MKL ILP64:          dsyevr_64_     (built with -Dblas-symbol-suffix=_64)
 *   scipy-openblas64:   scipy_dsyevr_64_  (PyPI numpy 2.x bundles this)
 *   OpenBLAS ILP64:     dsyevr64_      (SYMBOLSUFFIX=64_)
 *   MKL LP64 / system:  dsyevr_        (standard Fortran name mangling)
 *   Accelerate (macOS): dsyevr_        (standard)
 */
static const char *ilp64_symbol_names[] = {
    "dsyevr_64_",           /* MKL ILP64, numpy-mkl ILP64 */
    "scipy_dsyevr_64_",     /* scipy-openblas64 (PyPI numpy wheels) */
    "dsyevr64_",            /* OpenBLAS SYMBOLSUFFIX=64_ */
    NULL
};

static const char *lp64_symbol_names[] = {
    "dsyevr_",              /* Standard Fortran, MKL LP64, Accelerate */
    NULL
};

/* Try to resolve dsyevr from a dlopen handle (or RTLD_DEFAULT).
 * Returns 1 if found, 0 if not. Sets g_is_ilp64 and the function pointer. */
static int try_resolve_dsyevr(void *handle) {
    int dbg = debug_enabled();

    /* Try ILP64 symbols first */
    for (const char **name = ilp64_symbol_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "_eigen_accel:   resolved %s\n", *name);
            g_dsyevr_ilp64 = (dsyevr_ilp64_fn)sym;
            g_is_ilp64 = 1;
            return 1;
        }
    }

    /* Try LP64 symbols */
    for (const char **name = lp64_symbol_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "_eigen_accel:   resolved %s\n", *name);
            g_dsyevr_lp64 = (dsyevr_lp64_fn)sym;
            g_is_ilp64 = 0;
            return 1;
        }
    }

    return 0;
}

/* Scan a directory for LAPACK-providing shared libraries and try to dlopen them.
 * Returns 1 if DSYEVR was resolved, 0 if not. */
static int scan_dir_for_lapack(const char *dirpath) {
    int dbg = debug_enabled();
    DIR *dir = opendir(dirpath);
    if (!dir) {
        if (dbg) fprintf(stderr, "_eigen_accel:   scan_dir %s — opendir failed\n", dirpath);
        return 0;
    }
    if (dbg) fprintf(stderr, "_eigen_accel:   scan_dir %s — opened\n", dirpath);

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Look for openblas or mkl shared libraries */
        if (strstr(entry->d_name, "openblas") || strstr(entry->d_name, "libmkl")) {
            /* Must be a .so or .dylib */
            if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib"))
                continue;

            char fullpath[4096];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);

            if (dbg) fprintf(stderr, "_eigen_accel:   trying dlopen: %s\n", fullpath);
            void *handle = dlopen(fullpath, RTLD_LAZY | RTLD_GLOBAL);
            if (!handle) {
                if (dbg) fprintf(stderr, "_eigen_accel:   dlopen failed: %s\n", dlerror());
                continue;
            }

            if (try_resolve_dsyevr(handle)) {
                if (dbg) fprintf(stderr, "_eigen_accel:   resolved dsyevr from %s (ilp64=%d)\n", fullpath, g_is_ilp64);
                g_lapack_handle = handle;
                closedir(dir);
                return 1;
            }
            if (dbg) fprintf(stderr, "_eigen_accel:   dsyevr not found in %s\n", entry->d_name);
            dlclose(handle);
        }
    }
    closedir(dir);
    return 0;
}

/* Force numpy to load its BLAS/LAPACK by running a trivial linalg operation.
 * numpy loads BLAS lazily — until a linalg function is called, the BLAS
 * library may not be loaded into the process.
 *
 * Uses C API directly (not PyRun_String) because __builtins__ is unavailable
 * in the globals dict during module init, making import statements fail. */
static void force_numpy_blas_load(void) {
    PyObject *np = PyImport_ImportModule("numpy");
    if (!np) { PyErr_Clear(); return; }

    PyObject *linalg = PyObject_GetAttrString(np, "linalg");
    if (!linalg) { PyErr_Clear(); Py_DECREF(np); return; }

    PyObject *eigh = PyObject_GetAttrString(linalg, "eigh");
    PyObject *eye = PyObject_GetAttrString(np, "eye");
    if (!eigh || !eye) {
        PyErr_Clear();
        Py_XDECREF(eigh); Py_XDECREF(eye);
        Py_DECREF(linalg); Py_DECREF(np);
        return;
    }

    /* numpy.eye(2) */
    PyObject *two = PyLong_FromLong(2);
    PyObject *eye_result = PyObject_CallFunctionObjArgs(eye, two, NULL);
    Py_DECREF(two);

    if (eye_result) {
        /* numpy.linalg.eigh(eye(2)) — forces BLAS load */
        PyObject *eigh_result = PyObject_CallFunctionObjArgs(eigh, eye_result, NULL);
        if (eigh_result) {
            Py_DECREF(eigh_result);
        } else {
            PyErr_Clear();
        }
        Py_DECREF(eye_result);
    } else {
        PyErr_Clear();
    }

    Py_DECREF(eigh); Py_DECREF(eye);
    Py_DECREF(linalg); Py_DECREF(np);
}

/* Scan /proc/self/maps (Linux) for already-loaded BLAS/LAPACK libraries.
 * After numpy loads its BLAS, the library appears in the process map.
 * We can dlopen it by path with RTLD_NOLOAD to get a handle without
 * loading it again, then resolve dsyevr from that handle.
 * Returns 1 if DSYEVR was resolved, 0 if not. */
static int scan_proc_maps_for_lapack(void) {
#ifdef __linux__
    int dbg = debug_enabled();
    FILE *fp = fopen("/proc/self/maps", "r");
    if (!fp) {
        if (dbg) fprintf(stderr, "_eigen_accel:   /proc/self/maps — fopen failed\n");
        return 0;
    }

    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        char *path = strchr(line, '/');
        if (!path) continue;

        char *nl = strchr(path, '\n');
        if (nl) *nl = '\0';

        char *basename = strrchr(path, '/');
        if (!basename) continue;
        basename++;

        if (!strstr(basename, "openblas") && !strstr(basename, "libmkl"))
            continue;
        if (!strstr(basename, ".so"))
            continue;

        if (dbg) fprintf(stderr, "_eigen_accel:   /proc/self/maps candidate: %s\n", path);

        /* Found a candidate — dlopen with RTLD_NOLOAD to get existing handle */
        void *handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) {
            if (dbg) fprintf(stderr, "_eigen_accel:   RTLD_NOLOAD failed, trying full load: %s\n", dlerror());
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        }
        if (!handle) {
            if (dbg) fprintf(stderr, "_eigen_accel:   dlopen failed: %s\n", dlerror());
            continue;
        }

        if (try_resolve_dsyevr(handle)) {
            if (dbg) fprintf(stderr, "_eigen_accel:   resolved dsyevr from /proc/self/maps (ilp64=%d)\n", g_is_ilp64);
            g_lapack_handle = handle;
            fclose(fp);
            return 1;
        }
        if (dbg) fprintf(stderr, "_eigen_accel:   dsyevr not found in %s\n", basename);
        dlclose(handle);
    }
    fclose(fp);
#endif
    return 0;
}

/* Main LAPACK discovery function. Called once at module init.
 * Returns 1 if DSYEVR was resolved, 0 if not. */
static int discover_lapack(void) {
    int dbg = debug_enabled();

    /* 1. Try process-global symbols first (macOS Accelerate, LD_PRELOAD, etc.) */
    if (dbg) fprintf(stderr, "_eigen_accel: step 1 — RTLD_DEFAULT\n");
    if (try_resolve_dsyevr(RTLD_DEFAULT)) {
        if (dbg) fprintf(stderr, "_eigen_accel: found via RTLD_DEFAULT (ilp64=%d)\n", g_is_ilp64);
        return 1;
    }

    /* 2. Force numpy to load its BLAS (lazy load), then check again. */
    if (dbg) fprintf(stderr, "_eigen_accel: step 2 — force numpy BLAS load\n");
    force_numpy_blas_load();
    if (try_resolve_dsyevr(RTLD_DEFAULT)) {
        if (dbg) fprintf(stderr, "_eigen_accel: found via RTLD_DEFAULT after numpy load (ilp64=%d)\n", g_is_ilp64);
        return 1;
    }

    /* 3. Scan /proc/self/maps for the already-loaded BLAS library. */
    if (dbg) fprintf(stderr, "_eigen_accel: step 3 — /proc/self/maps scan\n");
    if (scan_proc_maps_for_lapack()) {
        if (dbg) fprintf(stderr, "_eigen_accel: found via /proc/self/maps (ilp64=%d)\n", g_is_ilp64);
        return 1;
    }

    /* 4. Fallback: scan numpy's lib directories for BLAS/LAPACK shared libs.
     * Uses C API to resolve numpy.__file__ and build candidate paths.
     * Avoids PyRun_String (__builtins__ unavailable during module init). */
    if (dbg) fprintf(stderr, "_eigen_accel: step 4 — numpy dir scan\n");
    PyObject *np2 = PyImport_ImportModule("numpy");
    if (!np2) { PyErr_Clear(); return 0; }

    PyObject *np_file = PyObject_GetAttrString(np2, "__file__");
    if (!np_file) { PyErr_Clear(); Py_DECREF(np2); return 0; }

    /* Get the resolved parent directory of numpy/__init__.py */
    PyObject *pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) { PyErr_Clear(); Py_DECREF(np_file); Py_DECREF(np2); return 0; }

    PyObject *Path = PyObject_GetAttrString(pathlib, "Path");
    if (!Path) { PyErr_Clear(); Py_DECREF(pathlib); Py_DECREF(np_file); Py_DECREF(np2); return 0; }

    PyObject *p = PyObject_CallFunctionObjArgs(Path, np_file, NULL);
    Py_DECREF(np_file);
    if (!p) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return 0; }

    /* resolve().parent → np_dir */
    PyObject *resolved = PyObject_CallMethod(p, "resolve", NULL);
    Py_DECREF(p);
    if (!resolved) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return 0; }

    PyObject *np_dir = PyObject_GetAttrString(resolved, "parent");
    Py_DECREF(resolved);
    if (!np_dir) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return 0; }

    /* Build candidate directories:
     *   np_dir / '.libs'
     *   np_dir / '_core' / '.libs'
     *   np_dir.parent / 'numpy.libs'  */
    const char *subpaths[] = { ".libs", "_core/.libs", NULL };
    for (int si = 0; subpaths[si]; si++) {
        PyObject *candidate = PyObject_CallMethod(np_dir, "__truediv__", "s", subpaths[si]);
        if (!candidate) { PyErr_Clear(); continue; }
        PyObject *cstr = PyObject_Str(candidate);
        Py_DECREF(candidate);
        if (!cstr) { PyErr_Clear(); continue; }
        const char *dirpath = PyUnicode_AsUTF8(cstr);
        if (dirpath && scan_dir_for_lapack(dirpath)) {
            Py_DECREF(cstr); Py_DECREF(np_dir); Py_DECREF(Path);
            Py_DECREF(pathlib); Py_DECREF(np2);
            return 1;
        }
        Py_DECREF(cstr);
    }

    /* np_dir.parent / 'numpy.libs' */
    PyObject *np_parent = PyObject_GetAttrString(np_dir, "parent");
    if (np_parent) {
        PyObject *candidate = PyObject_CallMethod(np_parent, "__truediv__", "s", "numpy.libs");
        if (candidate) {
            PyObject *cstr = PyObject_Str(candidate);
            Py_DECREF(candidate);
            if (cstr) {
                const char *dirpath = PyUnicode_AsUTF8(cstr);
                if (dirpath && scan_dir_for_lapack(dirpath)) {
                    Py_DECREF(cstr); Py_DECREF(np_parent); Py_DECREF(np_dir);
                    Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2);
                    return 1;
                }
                Py_DECREF(cstr);
            }
        } else {
            PyErr_Clear();
        }
        Py_DECREF(np_parent);
    } else {
        PyErr_Clear();
    }

    Py_DECREF(np_dir); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2);
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

        /* EIGEN-06: bounds check on workspace query result */
        if (lwork <= 0 || liwork <= 0) {
            free(isuppz);
            Py_DECREF(eigenvalues);
            Py_DECREF(eigenvectors);
            PyErr_Format(PyExc_RuntimeError,
                "DSYEVR workspace query returned invalid sizes: "
                "lwork=%lld, liwork=%lld. "
                "This indicates a LAPACK internal error.",
                lwork, liwork);
            return NULL;
        }

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

        /* EIGEN-06: bounds check on workspace query result */
        if (lwork <= 0 || liwork <= 0) {
            free(isuppz);
            Py_DECREF(eigenvalues);
            Py_DECREF(eigenvectors);
            PyErr_Format(PyExc_RuntimeError,
                "DSYEVR workspace query returned invalid sizes: "
                "lwork=%d, liwork=%d. "
                "This indicates a LAPACK internal error.",
                lwork, liwork);
            return NULL;
        }

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
