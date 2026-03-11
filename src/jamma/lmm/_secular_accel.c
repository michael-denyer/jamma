/*
 * _secular_accel.c — C extension calling LAPACK DLAED4 for rank-1 secular equation.
 *
 * Exported functions:
 *   rank1_eigenvalue_update(d, rho, z)   -> (eigenvalues, eigenvectors)
 *   rank1_eigenvalues_and_norms(d, rho, z) -> (eigenvalues, norms)
 *
 * Computes all eigenvalues and eigenvectors of D + rho * z * z^T where:
 *   D   = diagonal matrix with ascending entries d[0] < d[1] < ... < d[n-1]
 *   rho = scalar (positive or negative)
 *   z   = rank-1 update vector (will be normalized to unit norm internally)
 *
 * Algorithm: n calls to LAPACK DLAED4, one per eigenvalue. Each DLAED4 call
 * solves the secular equation for the i-th eigenvalue using deflation and the
 * Li algorithm (LAWN 89, Ren-Cang Li 1993). This is the numerically stable
 * reference implementation used inside LAPACK's DSYEVD divide-and-conquer.
 *
 * Negative rho handling:
 *   DLAED4 requires rho > 0. For negative rho, we exploit the identity:
 *     eig(D + rho*z*z^T) = -eig(-D + |rho|*z_rev*z_rev^T), reversed
 *   where -D reversal = -d[::-1] (negate and reverse d) and z_rev = z[::-1].
 *   This transforms the problem to positive rho without loss of accuracy.
 *
 * LAPACK resolution (runtime dlopen):
 *   At module init, discovers numpy's bundled LAPACK library via dlopen and
 *   resolves DLAED4 with dlsym. Tries dlaed4_64_ first (ILP64), then dlaed4_
 *   (LP64). This avoids link-time LAPACK dependencies.
 *
 *   ILP64: n, i, info = long long; rho = double* (unchanged in both variants)
 *   LP64:  n, i, info = int; rho = double* (unchanged in both variants)
 *
 * Key implementation notes:
 *   - z is normalized internally; rho is adjusted by ||z_orig||^2
 *   - delta output from dlaed4 is used for eigenvector formula:
 *     v_k = z / delta_k, then normalize to unit length
 *   - i is 1-indexed (Fortran convention)
 *   - Eigenvectors returned as C-contiguous (n, n) column matrix:
 *     eigenvectors[:, j] is the eigenvector for eigenvalues[j]
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #error "_secular_accel does not support Windows"
#else
  #include <dlfcn.h>
  #include <dirent.h>
#endif

/* ABI version: bump when function signatures or array layout expectations change. */
#define ABI_VERSION 2

/* ---------------------------------------------------------------------------
 * DLAED4 function pointer types.
 *
 * LP64:  n, i, info are int; rho is ALWAYS double*
 * ILP64: n, i, info are long long; rho is ALWAYS double*
 * ------------------------------------------------------------------------- */
typedef void (*dlaed4_lp64_fn)(
    int *n, int *i, double *d, double *z, double *delta,
    double *rho, double *dlam, int *info);

typedef void (*dlaed4_ilp64_fn)(
    long long *n, long long *i, double *d, double *z, double *delta,
    double *rho, double *dlam, long long *info);

/* Global state: resolved at module init */
static int g_is_ilp64 = 0;
static dlaed4_lp64_fn g_dlaed4_lp64 = NULL;
static dlaed4_ilp64_fn g_dlaed4_ilp64 = NULL;
static void *g_lapack_handle = NULL;

/* ---------------------------------------------------------------------------
 * LAPACK library discovery — mirrors _eigen_accel.c exactly.
 * ------------------------------------------------------------------------- */

static int debug_enabled(void) {
    const char *val = getenv("SECULAR_ACCEL_DEBUG");
    return val && val[0] == '1';
}

/* Symbol names to try for DLAED4, in priority order. */
static const char *ilp64_symbol_names[] = {
    "dlaed4_64_",           /* MKL ILP64, numpy-mkl ILP64 */
    "scipy_dlaed4_64_",     /* scipy-openblas64 (PyPI numpy wheels) */
    "dlaed464_",            /* OpenBLAS SYMBOLSUFFIX=64_ */
    NULL
};

static const char *lp64_symbol_names[] = {
    "dlaed4_",              /* Standard Fortran, MKL LP64, Accelerate */
    NULL
};

/* Try to resolve dlaed4 from a dlopen handle (or RTLD_DEFAULT).
 * Returns 1 if found, 0 if not. */
static int try_resolve_dlaed4(void *handle) {
    int dbg = debug_enabled();

    for (const char **name = ilp64_symbol_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "_secular_accel:   resolved %s\n", *name);
            g_dlaed4_ilp64 = (dlaed4_ilp64_fn)sym;
            g_is_ilp64 = 1;
            return 1;
        }
    }

    for (const char **name = lp64_symbol_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "_secular_accel:   resolved %s\n", *name);
            g_dlaed4_lp64 = (dlaed4_lp64_fn)sym;
            g_is_ilp64 = 0;
            return 1;
        }
    }

    return 0;
}

/* Scan a directory for LAPACK-providing shared libraries and try to dlopen them. */
static int scan_dir_for_lapack(const char *dirpath) {
    int dbg = debug_enabled();
    DIR *dir = opendir(dirpath);
    if (!dir) {
        if (dbg) fprintf(stderr, "_secular_accel:   scan_dir %s — opendir failed\n", dirpath);
        return 0;
    }
    if (dbg) fprintf(stderr, "_secular_accel:   scan_dir %s — opened\n", dirpath);

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, "openblas") || strstr(entry->d_name, "libmkl")) {
            if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib"))
                continue;

            char fullpath[4096];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);

            if (dbg) fprintf(stderr, "_secular_accel:   trying dlopen: %s\n", fullpath);
            void *handle = dlopen(fullpath, RTLD_LAZY | RTLD_GLOBAL);
            if (!handle) {
                if (dbg) fprintf(stderr, "_secular_accel:   dlopen failed: %s\n", dlerror());
                continue;
            }

            if (try_resolve_dlaed4(handle)) {
                if (dbg) fprintf(stderr, "_secular_accel:   resolved dlaed4 from %s (ilp64=%d)\n", fullpath, g_is_ilp64);
                g_lapack_handle = handle;
                closedir(dir);
                return 1;
            }
            if (dbg) fprintf(stderr, "_secular_accel:   dlaed4 not found in %s\n", entry->d_name);
            dlclose(handle);
        }
    }
    closedir(dir);
    return 0;
}

/* Force numpy to load its BLAS/LAPACK by running a trivial linalg operation. */
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

    PyObject *two = PyLong_FromLong(2);
    PyObject *eye_result = PyObject_CallFunctionObjArgs(eye, two, NULL);
    Py_DECREF(two);

    if (eye_result) {
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

/* Scan /proc/self/maps (Linux) for already-loaded BLAS/LAPACK libraries. */
static int scan_proc_maps_for_lapack(void) {
#ifdef __linux__
    int dbg = debug_enabled();
    FILE *fp = fopen("/proc/self/maps", "r");
    if (!fp) {
        if (dbg) fprintf(stderr, "_secular_accel:   /proc/self/maps — fopen failed\n");
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

        if (dbg) fprintf(stderr, "_secular_accel:   /proc/self/maps candidate: %s\n", path);

        void *handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) {
            if (dbg) fprintf(stderr, "_secular_accel:   RTLD_NOLOAD failed, trying full load: %s\n", dlerror());
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        }
        if (!handle) {
            if (dbg) fprintf(stderr, "_secular_accel:   dlopen failed: %s\n", dlerror());
            continue;
        }

        if (try_resolve_dlaed4(handle)) {
            if (dbg) fprintf(stderr, "_secular_accel:   resolved dlaed4 from /proc/self/maps (ilp64=%d)\n", g_is_ilp64);
            g_lapack_handle = handle;
            fclose(fp);
            return 1;
        }
        if (dbg) fprintf(stderr, "_secular_accel:   dlaed4 not found in %s\n", basename);
        dlclose(handle);
    }
    fclose(fp);
#endif
    return 0;
}

/* Main LAPACK discovery function. Called once at module init. */
static int discover_lapack(void) {
    int dbg = debug_enabled();

    /* 1. Try process-global symbols first (macOS Accelerate, LD_PRELOAD, etc.) */
    if (dbg) fprintf(stderr, "_secular_accel: step 1 — RTLD_DEFAULT\n");
    if (try_resolve_dlaed4(RTLD_DEFAULT)) {
        if (dbg) fprintf(stderr, "_secular_accel: found via RTLD_DEFAULT (ilp64=%d)\n", g_is_ilp64);
        return 1;
    }

    /* 2. Force numpy to load its BLAS (lazy load), then check again. */
    if (dbg) fprintf(stderr, "_secular_accel: step 2 — force numpy BLAS load\n");
    force_numpy_blas_load();
    if (try_resolve_dlaed4(RTLD_DEFAULT)) {
        if (dbg) fprintf(stderr, "_secular_accel: found via RTLD_DEFAULT after numpy load (ilp64=%d)\n", g_is_ilp64);
        return 1;
    }

    /* 3. Scan /proc/self/maps for the already-loaded BLAS library. */
    if (dbg) fprintf(stderr, "_secular_accel: step 3 — /proc/self/maps scan\n");
    if (scan_proc_maps_for_lapack()) {
        if (dbg) fprintf(stderr, "_secular_accel: found via /proc/self/maps (ilp64=%d)\n", g_is_ilp64);
        return 1;
    }

    /* 4. Fallback: scan numpy's lib directories for BLAS/LAPACK shared libs. */
    if (dbg) fprintf(stderr, "_secular_accel: step 4 — numpy dir scan\n");
    PyObject *np2 = PyImport_ImportModule("numpy");
    if (!np2) { PyErr_Clear(); return 0; }

    PyObject *np_file = PyObject_GetAttrString(np2, "__file__");
    if (!np_file) { PyErr_Clear(); Py_DECREF(np2); return 0; }

    PyObject *pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) { PyErr_Clear(); Py_DECREF(np_file); Py_DECREF(np2); return 0; }

    PyObject *Path = PyObject_GetAttrString(pathlib, "Path");
    if (!Path) { PyErr_Clear(); Py_DECREF(pathlib); Py_DECREF(np_file); Py_DECREF(np2); return 0; }

    PyObject *p = PyObject_CallFunctionObjArgs(Path, np_file, NULL);
    Py_DECREF(np_file);
    if (!p) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return 0; }

    PyObject *resolved = PyObject_CallMethod(p, "resolve", NULL);
    Py_DECREF(p);
    if (!resolved) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return 0; }

    PyObject *np_dir = PyObject_GetAttrString(resolved, "parent");
    Py_DECREF(resolved);
    if (!np_dir) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return 0; }

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
 * call_dlaed4 — wrapper that dispatches to LP64 or ILP64 dlaed4.
 *
 * Parameters:
 *   n:     dimension of the problem
 *   i:     0-indexed eigenvalue index (converted to 1-indexed internally)
 *   d:     working copy of diagonal (length n), sorted ascending, POSITIVE rho only
 *   z:     working copy of update vector (length n), unit norm
 *   delta: output buffer (length n), d[j] - dlam for eigenvector formula
 *   rho:   rank-1 weight — MUST be positive (caller handles negative rho transformation)
 *   dlam:  output: the i-th eigenvalue
 *
 * Returns info from dlaed4 (0 = success).
 * ------------------------------------------------------------------------- */
static int call_dlaed4(int n, int i_zero,
                        double *d, double *z, double *delta,
                        double rho, double *dlam)
{
    if (g_is_ilp64) {
        long long ln = (long long)n;
        long long li = (long long)(i_zero + 1);  /* 1-indexed */
        long long info = 0;
        g_dlaed4_ilp64(&ln, &li, d, z, delta, &rho, dlam, &info);
        return (int)info;
    } else {
        int ln = n;
        int li = i_zero + 1;  /* 1-indexed */
        int info = 0;
        g_dlaed4_lp64(&ln, &li, d, z, delta, &rho, dlam, &info);
        return info;
    }
}


/* ---------------------------------------------------------------------------
 * py_rank1_eigenvalue_update — Python-callable rank-1 update.
 *
 * Args:
 *   d:   float64 ndarray, shape (n,), ascending diagonal of D
 *   rho: float scalar, rank-1 weight (positive or negative)
 *   z:   float64 ndarray, shape (n,), rank-1 update vector (normalized internally)
 *
 * Returns:
 *   (eigenvalues, eigenvectors) tuple
 *   eigenvalues:  float64 ndarray, shape (n,), ascending order
 *   eigenvectors: float64 ndarray, shape (n, n), C-contiguous
 *                 eigenvectors[:, j] is the eigenvector for eigenvalues[j]
 *
 * Implementation:
 *   1. Normalize z_unit = z / ||z||; rho_eff = rho * ||z||^2
 *   2. For positive rho: call dlaed4 n times, eigenvectors from delta
 *   3. For negative rho: transform via negation/reversal, call dlaed4 n times
 *      with positive rho, then invert the transformation on results
 *   4. Transpose in-place to get column-major eigenvector storage
 * ------------------------------------------------------------------------- */
static PyObject *py_rank1_eigenvalue_update(PyObject *self, PyObject *args)
{
    PyArrayObject *d_arr = NULL;
    double rho;
    PyArrayObject *z_arr = NULL;

    if (!PyArg_ParseTuple(args, "O!dO!", &PyArray_Type, &d_arr, &rho, &PyArray_Type, &z_arr)) {
        return NULL;
    }

    /* Check that DLAED4 was resolved at init time */
    if (!g_dlaed4_lp64 && !g_dlaed4_ilp64) {
        PyErr_SetString(PyExc_RuntimeError,
            "DLAED4 symbol not resolved. numpy's LAPACK library could not be found. "
            "Ensure numpy is installed and its bundled BLAS/LAPACK is accessible.");
        return NULL;
    }

    /* ---- Validate inputs -------------------------------------------------- */
    if (PyArray_NDIM(d_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "d must be a 1D array");
        return NULL;
    }
    if (PyArray_TYPE(d_arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "d must be float64");
        return NULL;
    }

    npy_intp n = PyArray_DIM(d_arr, 0);

    if (PyArray_NDIM(z_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "z must be a 1D array");
        return NULL;
    }
    if (PyArray_TYPE(z_arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "z must be float64");
        return NULL;
    }
    if (PyArray_DIM(z_arr, 0) != n) {
        PyErr_Format(PyExc_ValueError,
            "z length %ld must match d length %ld",
            (long)PyArray_DIM(z_arr, 0), (long)n);
        return NULL;
    }

    if (n == 0) {
        npy_intp zero = 0;
        PyObject *w_empty = PyArray_SimpleNew(1, &zero, NPY_FLOAT64);
        npy_intp dims2[2] = {0, 0};
        PyObject *v_empty = PyArray_SimpleNew(2, dims2, NPY_FLOAT64);
        if (!w_empty || !v_empty) {
            Py_XDECREF(w_empty); Py_XDECREF(v_empty);
            return NULL;
        }
        PyObject *result = Py_BuildValue("(OO)", w_empty, v_empty);
        Py_DECREF(w_empty); Py_DECREF(v_empty);
        return result;
    }

    /* LP64 overflow guard */
    if (!g_is_ilp64) {
        int ln_test = (int)n;
        if ((npy_intp)ln_test != n) {
            PyErr_Format(PyExc_OverflowError,
                "Dimension %ld exceeds LP64 LAPACK int32 limit. "
                "Install ILP64 numpy for large matrices.",
                (long)n);
            return NULL;
        }
    }

    /* ---- Get contiguous double pointers ---------------------------------- */
    PyArrayObject *d_c = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)d_arr, NPY_FLOAT64, 1, 1);
    if (!d_c) return NULL;

    PyArrayObject *z_c = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)z_arr, NPY_FLOAT64, 1, 1);
    if (!z_c) { Py_DECREF(d_c); return NULL; }

    double *d_ptr = (double *)PyArray_DATA(d_c);
    double *z_ptr = (double *)PyArray_DATA(z_c);

    /* ---- Normalize z and adjust rho -------------------------------------- */
    double z_norm_sq = 0.0;
    for (npy_intp k = 0; k < n; k++) {
        z_norm_sq += z_ptr[k] * z_ptr[k];
    }
    double z_norm = sqrt(z_norm_sq);

    /* rho_eff = rho * ||z_orig||^2 so that:
     *   rho_eff * z_unit @ z_unit^T = rho * z_orig @ z_orig^T */
    double rho_eff = rho * z_norm_sq;

    /* Normalize z in-place */
    if (z_norm > 0.0) {
        for (npy_intp k = 0; k < n; k++) {
            z_ptr[k] /= z_norm;
        }
    }

    /* Determine if we need the negative rho transformation.
     * DLAED4 requires rho > 0. For rho_eff < 0, we transform the problem:
     *   eig(D + rho*z*z^T) = -eig(-D_rev + |rho|*z_rev*z_rev^T), reversed
     * where D_rev = -d[::-1] and z_rev = z[::-1]. */
    int negative_rho = (rho_eff < 0.0);
    double rho_pos = negative_rho ? -rho_eff : rho_eff;

    /* ---- Allocate output arrays ------------------------------------------ */
    npy_intp n_dim = n;
    PyArrayObject *eigenvalues = (PyArrayObject *)PyArray_SimpleNew(1, &n_dim, NPY_FLOAT64);
    if (!eigenvalues) {
        Py_DECREF(d_c); Py_DECREF(z_c);
        return NULL;
    }

    npy_intp ev_dims[2] = {n, n};
    /* Allocate eigenvectors row-major; we transpose at the end.
     * During computation, row i = eigenvector i (temporary storage). */
    PyArrayObject *eigenvectors = (PyArrayObject *)PyArray_SimpleNew(2, ev_dims, NPY_FLOAT64);
    if (!eigenvectors) {
        Py_DECREF(d_c); Py_DECREF(z_c); Py_DECREF(eigenvalues);
        return NULL;
    }

    double *w_data = (double *)PyArray_DATA(eigenvalues);
    double *v_data = (double *)PyArray_DATA(eigenvectors);

    /* ---- Allocate workspace ---------------------------------------------- */
    /* d_work: working copy of d for each dlaed4 call (dlaed4 may overwrite) */
    double *d_work = (double *)malloc((size_t)n * sizeof(double));
    /* z_work: working copy of z for each dlaed4 call */
    double *z_work = (double *)malloc((size_t)n * sizeof(double));
    /* delta: output from dlaed4, reused each call */
    double *delta = (double *)malloc((size_t)n * sizeof(double));
    /* d_base and z_base: base arrays to copy from each iteration */
    double *d_base = (double *)malloc((size_t)n * sizeof(double));
    double *z_base = (double *)malloc((size_t)n * sizeof(double));

    if (!d_work || !z_work || !delta || !d_base || !z_base) {
        free(d_work); free(z_work); free(delta); free(d_base); free(z_base);
        Py_DECREF(d_c); Py_DECREF(z_c); Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
        return PyErr_NoMemory();
    }

    /* Prepare d_base and z_base depending on rho sign */
    if (negative_rho) {
        /* For negative rho: use -d[::-1] and z[::-1] with rho_pos = |rho_eff|
         * Identity: eig(D + rho*z*z^T) = -eig(-D_rev + |rho|*z_rev*z_rev^T), reversed */
        for (npy_intp k = 0; k < n; k++) {
            d_base[k] = -d_ptr[n - 1 - k];
            z_base[k] = z_ptr[n - 1 - k];
        }
    } else {
        memcpy(d_base, d_ptr, (size_t)n * sizeof(double));
        memcpy(z_base, z_ptr, (size_t)n * sizeof(double));
    }

    /* ---- Compute eigenvalues and eigenvectors via n dlaed4 calls --------- */
    int error_i = -1;
    int error_info = 0;

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp i = 0; i < n; i++) {
        /* Restore working copies for each call (dlaed4 may modify them) */
        memcpy(d_work, d_base, (size_t)n * sizeof(double));
        memcpy(z_work, z_base, (size_t)n * sizeof(double));

        double dlam = 0.0;
        int info = call_dlaed4((int)n, (int)i, d_work, z_work, delta, rho_pos, &dlam);

        if (info != 0) {
            error_i = (int)i;
            error_info = info;
            break;
        }

        if (negative_rho) {
            /* For the negated problem, i-th dlaed4 call gives eigenvalue of -A
             * at position i (ascending), corresponding to original eigenvalue
             * at position n-1-i (descending in original, ascending when negated).
             *
             * Original eigenvalue[n-1-i] = -dlam_neg[i]
             * Original eigenvector[n-1-i]: reverse the components of the
             *   negated eigenvector (z_base/delta_neg → original z/delta mapping) */
            npy_intp dest_i = n - 1 - i;
            w_data[dest_i] = -dlam;

            /* Eigenvector: v_k = z_base[k] / delta[k], normalized
             * Then reverse: base index k maps to original index n-1-k.
             * Use d_work as scratch (already modified by dlaed4, no longer needed). */
            double *row = v_data + dest_i * n;  /* row dest_i = eigenvec dest_i */
            double norm_sq = 0.0;
            for (npy_intp k = 0; k < n; k++) {
                double val = z_base[k] / delta[k];
                d_work[k] = val;  /* scratch: store unnormalized component */
                norm_sq += val * val;
            }
            double inv_norm = (norm_sq > 0.0) ? (1.0 / sqrt(norm_sq)) : 0.0;
            /* Reverse: base index k maps to original index n-1-k */
            for (npy_intp k = 0; k < n; k++) {
                row[n - 1 - k] = d_work[k] * inv_norm;
            }
        } else {
            w_data[i] = dlam;

            /* Eigenvector i: v_i = z_unit / delta, then normalize.
             * delta[k] = d[k] - dlam for the secular equation (LAWN 89). */
            double *row = v_data + i * n;  /* row i = eigenvec i (temporary row-major) */
            double norm_sq = 0.0;
            for (npy_intp k = 0; k < n; k++) {
                double val = z_ptr[k] / delta[k];
                row[k] = val;
                norm_sq += val * val;
            }
            if (norm_sq > 0.0) {
                double inv_norm = 1.0 / sqrt(norm_sq);
                for (npy_intp k = 0; k < n; k++) {
                    row[k] *= inv_norm;
                }
            }
        }
    }

    Py_END_ALLOW_THREADS

    free(d_work); free(z_work); free(delta); free(d_base); free(z_base);
    Py_DECREF(d_c); Py_DECREF(z_c);

    if (error_i >= 0) {
        Py_DECREF(eigenvalues); Py_DECREF(eigenvectors);
        if (error_info < 0) {
            PyErr_Format(PyExc_ValueError,
                "DLAED4(i=%d): parameter %d is invalid", error_i + 1, -error_info);
        } else {
            PyErr_Format(PyExc_RuntimeError,
                "DLAED4(i=%d) failed to converge (info=%d)", error_i + 1, error_info);
        }
        return NULL;
    }

    /* ---- Transpose in-place: rows → columns ------------------------------ */
    /* v_data currently stores: row j = eigenvec j (temporary row-major layout)
     * We need: column j = eigenvec j (C-contiguous column-major semantics)
     * i.e., v_data[k*n + j] = component k of eigenvec j
     * Transpose makes v_data[j*n + k] ↔ v_data[k*n + j]. */
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = i + 1; j < n; j++) {
            double tmp = v_data[i * n + j];
            v_data[i * n + j] = v_data[j * n + i];
            v_data[j * n + i] = tmp;
        }
    }

    /* ---- Return (eigenvalues, eigenvectors) ------------------------------ */
    PyObject *result = Py_BuildValue("(OO)",
        (PyObject *)eigenvalues, (PyObject *)eigenvectors);
    Py_DECREF(eigenvalues);
    Py_DECREF(eigenvectors);
    return result;
}


/* ---------------------------------------------------------------------------
 * py_rank1_eigenvalues_and_norms — like py_rank1_eigenvalue_update but returns
 * only (eigenvalues, norms) instead of (eigenvalues, eigenvectors).
 *
 * Memory: O(n) output vs O(n^2) for full eigenvector matrix. At n=83k, this
 * avoids allocating a 55 GB eigenvector matrix.
 *
 * Args:
 *   d:   float64 ndarray, shape (n,), ascending diagonal of D
 *   rho: float scalar, rank-1 weight (positive or negative)
 *   z:   float64 ndarray, shape (n,), rank-1 update vector (normalized internally)
 *
 * Returns:
 *   (eigenvalues, norms) tuple
 *   eigenvalues: float64 ndarray, shape (n,), ascending order
 *   norms:       float64 ndarray, shape (n,), where norms[k] = ||z_unit / delta_k||_2
 *                for eigenvalue k. This is the normalization factor needed by the
 *                backward pass Cauchy multiply without materialising eigenvectors.
 *
 * The norm formula: after each dlaed4 call, delta[k] = d[k] - eigenvalue[i].
 *   norm_i = sqrt(sum_k( (z_unit[k] / delta[k])^2 ))
 * This is exactly 1/||v_i||_unnorm where v_i = z_unit / delta is the unnormalized
 * eigenvector. The unit eigenvector would be v_i * (1/norm_i).
 *
 * Deflation guard: if |delta[k]| < 1e-300, skip that term in norm_sq to prevent
 * inf/NaN from machine-epsilon poles (z component is effectively deflated by dlaed4).
 * ------------------------------------------------------------------------- */
static PyObject *py_rank1_eigenvalues_and_norms(PyObject *self, PyObject *args)
{
    PyArrayObject *d_arr = NULL;
    double rho;
    PyArrayObject *z_arr = NULL;

    if (!PyArg_ParseTuple(args, "O!dO!", &PyArray_Type, &d_arr, &rho, &PyArray_Type, &z_arr)) {
        return NULL;
    }

    /* Check that DLAED4 was resolved at init time */
    if (!g_dlaed4_lp64 && !g_dlaed4_ilp64) {
        PyErr_SetString(PyExc_RuntimeError,
            "DLAED4 symbol not resolved. numpy's LAPACK library could not be found. "
            "Ensure numpy is installed and its bundled BLAS/LAPACK is accessible.");
        return NULL;
    }

    /* ---- Validate inputs -------------------------------------------------- */
    if (PyArray_NDIM(d_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "d must be a 1D array");
        return NULL;
    }
    if (PyArray_TYPE(d_arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "d must be float64");
        return NULL;
    }

    npy_intp n = PyArray_DIM(d_arr, 0);

    if (PyArray_NDIM(z_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "z must be a 1D array");
        return NULL;
    }
    if (PyArray_TYPE(z_arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "z must be float64");
        return NULL;
    }
    if (PyArray_DIM(z_arr, 0) != n) {
        PyErr_Format(PyExc_ValueError,
            "z length %ld must match d length %ld",
            (long)PyArray_DIM(z_arr, 0), (long)n);
        return NULL;
    }

    if (n == 0) {
        npy_intp zero = 0;
        PyObject *w_empty = PyArray_SimpleNew(1, &zero, NPY_FLOAT64);
        PyObject *norms_empty = PyArray_SimpleNew(1, &zero, NPY_FLOAT64);
        if (!w_empty || !norms_empty) {
            Py_XDECREF(w_empty); Py_XDECREF(norms_empty);
            return NULL;
        }
        PyObject *result = Py_BuildValue("(OO)", w_empty, norms_empty);
        Py_DECREF(w_empty); Py_DECREF(norms_empty);
        return result;
    }

    /* LP64 overflow guard */
    if (!g_is_ilp64) {
        int ln_test = (int)n;
        if ((npy_intp)ln_test != n) {
            PyErr_Format(PyExc_OverflowError,
                "Dimension %ld exceeds LP64 LAPACK int32 limit. "
                "Install ILP64 numpy for large matrices.",
                (long)n);
            return NULL;
        }
    }

    /* ---- Get contiguous double pointers ---------------------------------- */
    PyArrayObject *d_c = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)d_arr, NPY_FLOAT64, 1, 1);
    if (!d_c) return NULL;

    PyArrayObject *z_c = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)z_arr, NPY_FLOAT64, 1, 1);
    if (!z_c) { Py_DECREF(d_c); return NULL; }

    double *d_ptr = (double *)PyArray_DATA(d_c);
    double *z_ptr = (double *)PyArray_DATA(z_c);

    /* ---- Normalize z and adjust rho -------------------------------------- */
    double z_norm_sq = 0.0;
    for (npy_intp k = 0; k < n; k++) {
        z_norm_sq += z_ptr[k] * z_ptr[k];
    }
    double z_norm = sqrt(z_norm_sq);

    double rho_eff = rho * z_norm_sq;

    /* Normalize z in-place */
    if (z_norm > 0.0) {
        for (npy_intp k = 0; k < n; k++) {
            z_ptr[k] /= z_norm;
        }
    }

    int negative_rho = (rho_eff < 0.0);
    double rho_pos = negative_rho ? -rho_eff : rho_eff;

    /* ---- Allocate output arrays ------------------------------------------ */
    npy_intp n_dim = n;
    PyArrayObject *eigenvalues = (PyArrayObject *)PyArray_SimpleNew(1, &n_dim, NPY_FLOAT64);
    if (!eigenvalues) {
        Py_DECREF(d_c); Py_DECREF(z_c);
        return NULL;
    }

    PyArrayObject *norms_arr = (PyArrayObject *)PyArray_SimpleNew(1, &n_dim, NPY_FLOAT64);
    if (!norms_arr) {
        Py_DECREF(d_c); Py_DECREF(z_c); Py_DECREF(eigenvalues);
        return NULL;
    }

    double *w_data = (double *)PyArray_DATA(eigenvalues);
    double *norms_data = (double *)PyArray_DATA(norms_arr);

    /* ---- Allocate workspace ---------------------------------------------- */
    double *d_work = (double *)malloc((size_t)n * sizeof(double));
    double *z_work = (double *)malloc((size_t)n * sizeof(double));
    double *delta  = (double *)malloc((size_t)n * sizeof(double));
    double *d_base = (double *)malloc((size_t)n * sizeof(double));
    double *z_base = (double *)malloc((size_t)n * sizeof(double));

    if (!d_work || !z_work || !delta || !d_base || !z_base) {
        free(d_work); free(z_work); free(delta); free(d_base); free(z_base);
        Py_DECREF(d_c); Py_DECREF(z_c); Py_DECREF(eigenvalues); Py_DECREF(norms_arr);
        return PyErr_NoMemory();
    }

    /* Prepare d_base and z_base depending on rho sign */
    if (negative_rho) {
        for (npy_intp k = 0; k < n; k++) {
            d_base[k] = -d_ptr[n - 1 - k];
            z_base[k] = z_ptr[n - 1 - k];
        }
    } else {
        memcpy(d_base, d_ptr, (size_t)n * sizeof(double));
        memcpy(z_base, z_ptr, (size_t)n * sizeof(double));
    }

    /* ---- Compute eigenvalues and norms via n dlaed4 calls ---------------- */
    int error_i = -1;
    int error_info = 0;

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp i = 0; i < n; i++) {
        memcpy(d_work, d_base, (size_t)n * sizeof(double));
        memcpy(z_work, z_base, (size_t)n * sizeof(double));

        double dlam = 0.0;
        int info = call_dlaed4((int)n, (int)i, d_work, z_work, delta, rho_pos, &dlam);

        if (info != 0) {
            error_i = (int)i;
            error_info = info;
            break;
        }

        /* Compute norm_sq = sum_k( (z_base[k] / delta[k])^2 )
         * Deflation guard: skip terms where |delta[k]| < 1e-300 to avoid
         * inf/NaN from machine-epsilon poles in the denominator. */
        double norm_sq = 0.0;
        for (npy_intp k = 0; k < n; k++) {
            double dk = delta[k];
            if (dk < -1e-300 || dk > 1e-300) {
                double val = z_base[k] / dk;
                norm_sq += val * val;
            }
        }
        double norm_val = (norm_sq > 0.0) ? sqrt(norm_sq) : 0.0;

        if (negative_rho) {
            /* Negation/reversal identity: eigenvalue dest_i = -dlam (reversed) */
            npy_intp dest_i = n - 1 - i;
            w_data[dest_i] = -dlam;
            norms_data[dest_i] = norm_val;
        } else {
            w_data[i] = dlam;
            norms_data[i] = norm_val;
        }
    }

    Py_END_ALLOW_THREADS

    free(d_work); free(z_work); free(delta); free(d_base); free(z_base);
    Py_DECREF(d_c); Py_DECREF(z_c);

    if (error_i >= 0) {
        Py_DECREF(eigenvalues); Py_DECREF(norms_arr);
        if (error_info < 0) {
            PyErr_Format(PyExc_ValueError,
                "DLAED4(i=%d): parameter %d is invalid", error_i + 1, -error_info);
        } else {
            PyErr_Format(PyExc_RuntimeError,
                "DLAED4(i=%d) failed to converge (info=%d)", error_i + 1, error_info);
        }
        return NULL;
    }

    /* ---- Return (eigenvalues, norms) ------------------------------------- */
    PyObject *result = Py_BuildValue("(OO)",
        (PyObject *)eigenvalues, (PyObject *)norms_arr);
    Py_DECREF(eigenvalues);
    Py_DECREF(norms_arr);
    return result;
}


/* ---------------------------------------------------------------------------
 * Module definition
 * ------------------------------------------------------------------------- */
static PyMethodDef methods[] = {
    {
        "rank1_eigenvalue_update",
        py_rank1_eigenvalue_update,
        METH_VARARGS,
        "rank1_eigenvalue_update(d, rho, z) -> (eigenvalues, eigenvectors)\n"
        "\n"
        "Compute eigenvalues and eigenvectors of D + rho * z * z^T via LAPACK DLAED4.\n"
        "\n"
        "Args:\n"
        "    d:   float64 ndarray, shape (n,), ascending diagonal entries of D.\n"
        "    rho: float, rank-1 weight (positive or negative).\n"
        "    z:   float64 ndarray, shape (n,), rank-1 update vector.\n"
        "         Normalized to unit norm internally; rho is adjusted accordingly.\n"
        "\n"
        "Returns:\n"
        "    (eigenvalues, eigenvectors) tuple:\n"
        "    eigenvalues:  float64 ndarray, shape (n,), ascending order.\n"
        "    eigenvectors: float64 ndarray, shape (n, n), C-contiguous.\n"
        "                  eigenvectors[:, j] is the eigenvector for eigenvalues[j].\n"
        "\n"
        "Raises:\n"
        "    ValueError:   d or z are not 1D float64, sizes mismatch, or DLAED4\n"
        "                  detects invalid parameters.\n"
        "    RuntimeError: DLAED4 convergence failure or LAPACK symbol not resolved.\n"
        "    MemoryError:  workspace allocation failure.\n"
    },
    {
        "rank1_eigenvalues_and_norms",
        py_rank1_eigenvalues_and_norms,
        METH_VARARGS,
        "rank1_eigenvalues_and_norms(d, rho, z) -> (eigenvalues, norms)\n"
        "\n"
        "Compute eigenvalues and per-eigenvalue norms of D + rho * z * z^T via LAPACK DLAED4.\n"
        "\n"
        "Like rank1_eigenvalue_update but returns norms instead of the full eigenvector\n"
        "matrix. Memory: O(n) output vs O(n^2). At n=83k, avoids a 55 GB allocation.\n"
        "\n"
        "Args:\n"
        "    d:   float64 ndarray, shape (n,), ascending diagonal entries of D.\n"
        "    rho: float, rank-1 weight (positive or negative).\n"
        "    z:   float64 ndarray, shape (n,), rank-1 update vector.\n"
        "         Normalized to unit norm internally; rho is adjusted accordingly.\n"
        "\n"
        "Returns:\n"
        "    (eigenvalues, norms) tuple:\n"
        "    eigenvalues: float64 ndarray, shape (n,), ascending order.\n"
        "    norms:       float64 ndarray, shape (n,).\n"
        "                 norms[k] = ||z_unit / delta_k||_2, where delta_k is the\n"
        "                 DLAED4 delta output for eigenvalue k. This is the\n"
        "                 normalization factor for the unnormalized eigenvector\n"
        "                 z_unit / delta_k, needed by the backward pass Cauchy multiply.\n"
        "\n"
        "Raises:\n"
        "    ValueError:   d or z are not 1D float64, sizes mismatch, or DLAED4\n"
        "                  detects invalid parameters.\n"
        "    RuntimeError: DLAED4 convergence failure or LAPACK symbol not resolved.\n"
        "    MemoryError:  workspace allocation failure.\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_secular_accel",
    "C extension: rank-1 secular equation eigenvalue update via LAPACK DLAED4.\n"
    "LAPACK resolved at runtime via dlopen — no link-time dependency.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__secular_accel(void)
{
    import_array();

    int lapack_found = discover_lapack();

    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddIntConstant(m, "IS_ILP64", g_is_ilp64) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (!lapack_found) {
        if (PyErr_WarnEx(PyExc_RuntimeWarning,
                "_secular_accel: DLAED4 symbol not found in numpy's LAPACK. "
                "rank1_eigenvalue_update() will not be available. "
                "Falling back to pure-Python implementation.", 1) < 0) {
            Py_DECREF(m);
            return NULL;
        }
    }

    return m;
}
