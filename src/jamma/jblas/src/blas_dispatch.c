/**
 * blas_dispatch.c -- Three-tier dgemm discovery and dispatch wrapper.
 *
 * At jblas init time, discovers external dgemm implementations in this order:
 *   1. System BLAS (MKL, OpenBLAS, Accelerate -- via numpy's bundled libs)
 *   2. Bundled BLIS (dlopen'd from a known path relative to the extension .so)
 *   3. jblas own blocking dgemm (the default, already set by platform.c)
 *
 * When an external dgemm is found, replaces jblas_dispatch.dgemm with a
 * wrapper that converts row-major C = A*B into the Fortran column-major
 * convention expected by the external library.
 *
 * ILP64 / LP64 detection uses the same dlopen+dlsym pattern proven in
 * _eigen_accel.c: try ILP64-suffixed symbols first, fall back to LP64.
 * LP64 dimensions are guarded against int32 overflow at N > 46340.
 *
 * The dlopen machinery is Unix-only (#if !defined(_WIN32)); on Windows
 * blas_dispatch_init() returns 0 immediately (no external dispatch).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "jblas.h"

#if !defined(_WIN32)

#include <dlfcn.h>
#include <dirent.h>

/* ---------------------------------------------------------------------------
 * Module-level state
 * ---------------------------------------------------------------------------
 */
static int g_is_ilp64 = 0;
static jblas_dgemm_lp64_fn  g_dgemm_lp64  = NULL;
static jblas_dgemm_ilp64_fn g_dgemm_ilp64 = NULL;
static jblas_cblas_dgemm_fn g_cblas_dgemm  = NULL;  /* preferred over Fortran */
static const char *g_backend_name = "jblas-own";
static void *g_blas_handle = NULL;

/* LP64 overflow guard: floor(sqrt(2^31 - 1)) */
#define LP64_DIM_MAX 46340

/* LP64 overflow counter: incremented when dimensions exceed LP64_DIM_MAX
 * and the fallback to jblas-own dgemm is used.  Resettable by py_eigh. */
static int g_lp64_overflow_count = 0;

int blas_dispatch_lp64_overflow_count(void) {
    return __atomic_load_n(&g_lp64_overflow_count, __ATOMIC_RELAXED);
}

void blas_dispatch_reset_lp64_overflow(void) {
    __atomic_store_n(&g_lp64_overflow_count, 0, __ATOMIC_RELAXED);
}

/* ---------------------------------------------------------------------------
 * Debug flag
 * ---------------------------------------------------------------------------
 */
static int _debug_enabled(void) {
    const char *val = getenv("JBLAS_DISPATCH_DEBUG");
    return val && val[0] == '1';
}

/* ---------------------------------------------------------------------------
 * Backend name detection from library path
 * ---------------------------------------------------------------------------
 */
static const char *_detect_backend_name(const char *lib_path, int is_ilp64) {
    if (lib_path) {
        if (strstr(lib_path, "mkl"))
            return is_ilp64 ? "MKL-ILP64" : "MKL-LP64";
        if (strstr(lib_path, "openblas"))
            return is_ilp64 ? "OpenBLAS-ILP64" : "OpenBLAS-LP64";
    }
#ifdef __APPLE__
    return "Accelerate";
#else
    return is_ilp64 ? "system-BLAS-ILP64" : "system-BLAS-LP64";
#endif
}

/* ---------------------------------------------------------------------------
 * Symbol resolution
 * ---------------------------------------------------------------------------
 */
static const char *ilp64_dgemm_names[] = {
    "dgemm_64_",              /* MKL ILP64 */
    "scipy_dgemm_64_",        /* scipy-openblas64 */
    "dgemm64_",               /* OpenBLAS INTERFACE64=1 */
    NULL
};
static const char *lp64_dgemm_names[] = {
    "dgemm_",                 /* Standard Fortran / Accelerate */
    NULL
};

/**
 * try_resolve_dgemm -- Try to resolve a dgemm symbol from a dlopen handle.
 * Returns 1 if found, 0 if not. Sets g_is_ilp64 and the function pointer.
 *
 * lib_path: hint for backend name detection (may be NULL for RTLD_DEFAULT).
 */
static int try_resolve_dgemm(void *handle, const char *lib_path) {
    int dbg = _debug_enabled();

    /* Try ILP64 symbols first */
    for (const char **name = ilp64_dgemm_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved %s\n", *name);
            g_dgemm_ilp64 = (jblas_dgemm_ilp64_fn)sym;
            g_is_ilp64 = 1;
            g_backend_name = _detect_backend_name(lib_path, 1);
            return 1;
        }
    }

    /* Try LP64 symbols */
    for (const char **name = lp64_dgemm_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved %s\n", *name);
            g_dgemm_lp64 = (jblas_dgemm_lp64_fn)sym;
            g_is_ilp64 = 0;
            g_backend_name = _detect_backend_name(lib_path, 0);

            /* Also try cblas_dgemm — row-major native, no A/B swap needed.
             * Accelerate/MKL can choose optimal algorithm for the layout. */
            void *cblas_sym = dlsym(handle, "cblas_dgemm");
            if (cblas_sym) {
                g_cblas_dgemm = (jblas_cblas_dgemm_fn)cblas_sym;
                if (dbg) fprintf(stderr, "jblas_dispatch:   also resolved cblas_dgemm\n");
            }
            return 1;
        }
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 * Directory scanning
 * ---------------------------------------------------------------------------
 */

/**
 * scan_dir_for_blas -- Scan a directory for BLAS-providing shared libraries.
 * Returns 1 if dgemm was resolved, 0 if not.
 */
static int scan_dir_for_blas(const char *dirpath) {
    int dbg = _debug_enabled();
    DIR *dir = opendir(dirpath);
    if (!dir) {
        if (dbg) fprintf(stderr, "jblas_dispatch:   scan_dir %s -- opendir failed\n", dirpath);
        return 0;
    }
    if (dbg) fprintf(stderr, "jblas_dispatch:   scan_dir %s -- opened\n", dirpath);

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Look for openblas, mkl, or blis shared libraries */
        if (!strstr(entry->d_name, "openblas") &&
            !strstr(entry->d_name, "libmkl") &&
            !strstr(entry->d_name, "libblis"))
            continue;
        /* Must be a .so or .dylib */
        if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib"))
            continue;

        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);

        if (dbg) fprintf(stderr, "jblas_dispatch:   trying dlopen: %s\n", fullpath);
        void *handle = dlopen(fullpath, RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   dlopen failed: %s\n", dlerror());
            continue;
        }

        if (try_resolve_dgemm(handle, fullpath)) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dgemm from %s (ilp64=%d)\n",
                             fullpath, g_is_ilp64);
            g_blas_handle = handle;
            closedir(dir);
            return 1;
        }
        if (dbg) fprintf(stderr, "jblas_dispatch:   dgemm not found in %s\n", entry->d_name);
        dlclose(handle);
    }
    closedir(dir);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Force numpy BLAS load (identical pattern to _eigen_accel.c)
 * ---------------------------------------------------------------------------
 */
static void force_numpy_blas_load(void) {
    int dbg = _debug_enabled();
    PyObject *np = PyImport_ImportModule("numpy");
    if (!np) {
        if (dbg) fprintf(stderr, "jblas_dispatch: force_numpy_blas_load: numpy import failed\n");
        PyErr_Clear(); return;
    }

    PyObject *linalg = PyObject_GetAttrString(np, "linalg");
    if (!linalg) {
        if (dbg) fprintf(stderr, "jblas_dispatch: force_numpy_blas_load: numpy.linalg not found\n");
        PyErr_Clear(); Py_DECREF(np); return;
    }

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
            if (dbg) fprintf(stderr, "jblas_dispatch: force_numpy_blas_load: eigh(eye(2)) failed\n");
            PyErr_Clear();
        }
        Py_DECREF(eye_result);
    } else {
        if (dbg) fprintf(stderr, "jblas_dispatch: force_numpy_blas_load: eye(2) failed\n");
        PyErr_Clear();
    }

    Py_DECREF(eigh); Py_DECREF(eye);
    Py_DECREF(linalg); Py_DECREF(np);
}

/* ---------------------------------------------------------------------------
 * Scan /proc/self/maps for already-loaded BLAS libraries (Linux only)
 * ---------------------------------------------------------------------------
 */
static int scan_proc_maps_for_blas(void) {
#ifdef __linux__
    int dbg = _debug_enabled();
    FILE *fp = fopen("/proc/self/maps", "r");
    if (!fp) {
        if (dbg) fprintf(stderr, "jblas_dispatch:   /proc/self/maps -- fopen failed\n");
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

        if (dbg) fprintf(stderr, "jblas_dispatch:   /proc/self/maps candidate: %s\n", path);

        void *handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   RTLD_NOLOAD failed, trying full load: %s\n", dlerror());
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        }
        if (!handle) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   dlopen failed: %s\n", dlerror());
            continue;
        }

        if (try_resolve_dgemm(handle, path)) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dgemm from /proc/self/maps (ilp64=%d)\n", g_is_ilp64);
            g_blas_handle = handle;
            fclose(fp);
            return 1;
        }
        if (dbg) fprintf(stderr, "jblas_dispatch:   dgemm not found in %s\n", basename);
        dlclose(handle);
    }
    fclose(fp);
#endif
    return 0;
}

/* ---------------------------------------------------------------------------
 * discover_system_blas -- Full system BLAS discovery (4-step pattern)
 * ---------------------------------------------------------------------------
 */
static int discover_system_blas(void) {
    int dbg = _debug_enabled();

    /* Step 1: RTLD_DEFAULT (catches macOS Accelerate, LD_PRELOAD) */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 1 -- RTLD_DEFAULT\n");
    if (try_resolve_dgemm(RTLD_DEFAULT, NULL)) {
        if (dbg) fprintf(stderr, "jblas_dispatch: found via RTLD_DEFAULT (ilp64=%d, backend=%s)\n",
                         g_is_ilp64, g_backend_name);
        return 1;
    }

    /* Step 2: Force numpy to load its BLAS, then retry RTLD_DEFAULT */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 2 -- force numpy BLAS load\n");
    force_numpy_blas_load();
    if (try_resolve_dgemm(RTLD_DEFAULT, NULL)) {
        if (dbg) fprintf(stderr, "jblas_dispatch: found via RTLD_DEFAULT after numpy load (ilp64=%d, backend=%s)\n",
                         g_is_ilp64, g_backend_name);
        return 1;
    }

    /* Step 3: /proc/self/maps scan (Linux only) */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 3 -- /proc/self/maps scan\n");
    if (scan_proc_maps_for_blas()) {
        if (dbg) fprintf(stderr, "jblas_dispatch: found via /proc/self/maps (ilp64=%d, backend=%s)\n",
                         g_is_ilp64, g_backend_name);
        return 1;
    }

    /* Step 4: Scan numpy's lib directories */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 4 -- numpy dir scan\n");
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
        if (dirpath && scan_dir_for_blas(dirpath)) {
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
                if (dirpath && scan_dir_for_blas(dirpath)) {
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
 * discover_bundled_blis -- Look for libblis.{so,dylib} relative to extension
 * ---------------------------------------------------------------------------
 */
static int discover_bundled_blis(void) {
    int dbg = _debug_enabled();

    /* Use dladdr on blas_dispatch_init to find our own .so path */
    Dl_info info;
    if (!dladdr((void *)blas_dispatch_init, &info) || !info.dli_fname) {
        if (dbg) fprintf(stderr, "jblas_dispatch: dladdr failed for blas_dispatch_init\n");
        return 0;
    }

    /* Build path: dirname(extension.so)/libs/libblis.{so,dylib} */
    char ext_dir[4096];
    strncpy(ext_dir, info.dli_fname, sizeof(ext_dir) - 1);
    ext_dir[sizeof(ext_dir) - 1] = '\0';
    char *last_slash = strrchr(ext_dir, '/');
    if (!last_slash) return 0;
    *last_slash = '\0';

#ifdef __APPLE__
    const char *blis_name = "libblis.dylib";
#else
    const char *blis_name = "libblis.so";
#endif

    char blis_path[4096];
    snprintf(blis_path, sizeof(blis_path), "%s/libs/%s", ext_dir, blis_name);

    if (dbg) fprintf(stderr, "jblas_dispatch: trying bundled BLIS: %s\n", blis_path);

    /* RTLD_LOCAL: don't pollute global symbol namespace (Pitfall 2 from RESEARCH) */
    void *handle = dlopen(blis_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        if (dbg) fprintf(stderr, "jblas_dispatch: bundled BLIS not found: %s\n", dlerror());
        return 0;
    }

    if (try_resolve_dgemm(handle, blis_path)) {
        g_backend_name = "BLIS";
        g_blas_handle = handle;
        if (dbg) fprintf(stderr, "jblas_dispatch: resolved dgemm from bundled BLIS\n");
        return 1;
    }

    if (dbg) fprintf(stderr, "jblas_dispatch: dgemm not found in bundled BLIS\n");
    dlclose(handle);
    return 0;
}

/* ---------------------------------------------------------------------------
 * LP64 overflow guard — shared by both the simplified and full-signature
 * dispatch wrappers.  Returns 1 if overflow detected (caller must fall back
 * to jblas own dgemm), 0 if dimensions fit in int32.
 * ---------------------------------------------------------------------------
 */
static int _lp64_overflow_guard(npy_intp M, npy_intp N, npy_intp K,
                                npy_intp lda, npy_intp ldb, npy_intp ldc)
{
    if (g_is_ilp64)
        return 0;
    if (M <= LP64_DIM_MAX && N <= LP64_DIM_MAX && K <= LP64_DIM_MAX &&
        lda <= LP64_DIM_MAX && ldb <= LP64_DIM_MAX && ldc <= LP64_DIM_MAX)
        return 0;

    __atomic_add_fetch(&g_lp64_overflow_count, 1, __ATOMIC_RELAXED);
    static int warned = 0;
    if (!warned) {
        warned = 1;
        fprintf(stderr,
            "jblas_dispatch: WARNING: LP64 overflow guard triggered "
            "(M=%ld N=%ld K=%ld > %d). Falling back to jblas own dgemm "
            "which is much slower. Install ILP64 numpy for large matrices.\n",
            (long)M, (long)N, (long)K, LP64_DIM_MAX);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * Row-major wrapper: converts C = A * B to Fortran dgemm convention
 * ---------------------------------------------------------------------------
 * For row-major C = A*B, call Fortran dgemm as:
 *   dgemm('N','N', n, m, k, 1.0, B, n, A, k, 0.0, C, n)
 * This transposes the problem: C^T = B^T * A^T in column-major.
 * ---------------------------------------------------------------------------
 */
static void _dgemm_external_wrapper(
    npy_intp m, npy_intp n, npy_intp k,
    const double *A,
    const double *B,
    double       *C)
{
    if (_lp64_overflow_guard(m, n, k, k, n, n)) {
        jblas_dgemm_dispatch_fn(m, n, k, A, B, C);
        return;
    }

    const double alpha = 1.0;
    const double beta  = 0.0;

    /* Prefer CBLAS: row-major native, no swap needed.
     * CBLAS requires ld >= max(dim, 1) even for zero-size matrices. */
    if (g_cblas_dgemm) {
        int ik = k > 0 ? (int)k : 1;
        int in_ = n > 0 ? (int)n : 1;
        g_cblas_dgemm(JBLAS_CblasRowMajor,
                      JBLAS_CblasNoTrans, JBLAS_CblasNoTrans,
                      (int)m, (int)n, (int)k,
                      alpha, A, ik, B, in_,
                      beta,  C, in_);
        return;
    }

    if (g_is_ilp64) {
        const long long lm = (long long)m;
        const long long ln = (long long)n;
        const long long lk = (long long)k;
        /* Swap A/B and m/n for row-major -> column-major conversion */
        g_dgemm_ilp64("N", "N", &ln, &lm, &lk,
                       &alpha, B, &ln, A, &lk,
                       &beta,  C, &ln);
    } else {
        const int im = (int)m;
        const int in_ = (int)n;
        const int ik = (int)k;
        g_dgemm_lp64("N", "N", &in_, &im, &ik,
                      &alpha, B, &in_, A, &ik,
                      &beta,  C, &in_);
    }
}

/* ---------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------------
 */

int blas_dispatch_init(void) {
    int dbg = _debug_enabled();

    /* Try system BLAS first */
    if (discover_system_blas()) {
        if (dbg) fprintf(stderr, "jblas_dispatch: using %s for dgemm\n", g_backend_name);
        jblas_dispatch.dgemm = _dgemm_external_wrapper;
        return 0;
    }

    /* Try bundled BLIS */
    if (discover_bundled_blis()) {
        if (dbg) fprintf(stderr, "jblas_dispatch: using BLIS for dgemm\n");
        jblas_dispatch.dgemm = _dgemm_external_wrapper;
        return 0;
    }

    /* No external dgemm found -- jblas own dgemm stays in dispatch table */
    if (dbg) fprintf(stderr, "jblas_dispatch: no external dgemm found, using jblas-own\n");
    return 0;
}

const char *blas_backend_name(void) {
    return g_backend_name;
}

int blas_is_ilp64(void) {
    return g_is_ilp64;
}

int blas_has_external(void) {
    return g_dgemm_lp64 != NULL || g_dgemm_ilp64 != NULL;
}

/* ---------------------------------------------------------------------------
 * Full-signature external dgemm wrapper
 * ---------------------------------------------------------------------------
 * Row-major: C(M×N) = alpha * op_r(A)(M×K) * op_r(B)(K×N) + beta * C
 *
 * Converts to Fortran column-major by exploiting the identity:
 *   C^T = alpha * op(B^T) * op(A^T) + beta * C^T
 * So: dgemm(transb_row, transa_row, N, M, K, alpha,
 *           B_ptr, ldb_row, A_ptr, lda_row, beta, C_ptr, ldc_row)
 */
/* Returns 1 on success, 0 if LP64 overflow guard triggered (caller must
 * fall back to jblas own dgemm). */
static int _dgemm_external_full(
    npy_intp M, npy_intp N, npy_intp K,
    const double *A, npy_intp lda,
    const double *B, npy_intp ldb,
    double       *C, npy_intp ldc,
    int transa, int transb,
    double alpha, double beta)
{
    if (_lp64_overflow_guard(M, N, K, lda, ldb, ldc))
        return 0;

    /* Prefer CBLAS C interface: handles row-major natively, no A/B swap.
     * Accelerate/MKL can choose optimal algorithm for the access pattern.
     * CBLAS leading dimension rules (row-major):
     *   lda >= max(cols_of_A, 1): NoTrans → K, Trans → M.
     *   ldb >= max(cols_of_B, 1): NoTrans → N, Trans → K.
     *   ldc >= max(N, 1) always. */
    if (g_cblas_dgemm) {
        int ta = transa ? JBLAS_CblasTrans : JBLAS_CblasNoTrans;
        int tb = transb ? JBLAS_CblasTrans : JBLAS_CblasNoTrans;
        int ilda = (int)(lda > 0 ? lda : 1);
        int ildb = (int)(ldb > 0 ? ldb : 1);
        int ildc = (int)(ldc > 0 ? ldc : 1);
        g_cblas_dgemm(JBLAS_CblasRowMajor, ta, tb,
                      (int)M, (int)N, (int)K,
                      alpha, A, ilda, B, ildb,
                      beta,  C, ildc);
        return 1;
    }

    /* Fortran interface fallback: row-major → column-major swap */
    const char *transa_f = transb ? "T" : "N";
    const char *transb_f = transa ? "T" : "N";

    if (g_is_ilp64) {
        const long long lM = (long long)M, lN = (long long)N, lK = (long long)K;
        const long long llda = (long long)lda, lldb = (long long)ldb;
        const long long lldc = (long long)ldc;
        g_dgemm_ilp64(transa_f, transb_f, &lN, &lM, &lK,
                       &alpha, B, &lldb, A, &llda,
                       &beta,  C, &lldc);
    } else {
        const int iM = (int)M, iN = (int)N, iK = (int)K;
        const int ilda = (int)lda, ildb = (int)ldb, ildc = (int)ldc;
        g_dgemm_lp64(transa_f, transb_f, &iN, &iM, &iK,
                      &alpha, B, &ildb, A, &ilda,
                      &beta,  C, &ildc);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * Public full-signature dispatch API
 * ---------------------------------------------------------------------------
 */

void jblas_dgemm_ext(npy_intp M, npy_intp N, npy_intp K,
                     const double *A, npy_intp lda,
                     const double *B, npy_intp ldb,
                     double *C, npy_intp ldc,
                     int transa, int transb)
{
    if ((g_dgemm_lp64 || g_dgemm_ilp64) &&
        _dgemm_external_full(M, N, K, A, lda, B, ldb, C, ldc,
                             transa, transb, 1.0, 0.0)) {
        return;
    }
    /* No external BLAS, or LP64 overflow guard triggered */
    jblas_dgemm_c(M, N, K, A, lda, B, ldb, C, ldc, transa, transb);
}

void jblas_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K,
                        const double *A, npy_intp lda,
                        const double *B, npy_intp ldb,
                        double *C, npy_intp ldc,
                        int transa, int transb,
                        double alpha, double beta,
                        jblas_workspace_t *ws)
{
    if ((g_dgemm_lp64 || g_dgemm_ilp64) &&
        _dgemm_external_full(M, N, K, A, lda, B, ldb, C, ldc,
                             transa, transb, alpha, beta)) {
        return;
    }
    jblas_dgemm_ws(M, N, K, A, lda, B, ldb, C, ldc,
                   transa, transb, alpha, beta, ws);
}

#else /* _WIN32 */

/* Windows: no external dispatch -- always use jblas own dgemm */
int blas_dispatch_init(void) {
    return 0;
}

const char *blas_backend_name(void) {
    return "jblas-own";
}

int blas_is_ilp64(void) {
    return 0;
}

int blas_has_external(void) {
    return 0;
}

void jblas_dgemm_ext(npy_intp M, npy_intp N, npy_intp K,
                     const double *A, npy_intp lda,
                     const double *B, npy_intp ldb,
                     double *C, npy_intp ldc,
                     int transa, int transb)
{
    jblas_dgemm_c(M, N, K, A, lda, B, ldb, C, ldc, transa, transb);
}

void jblas_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K,
                        const double *A, npy_intp lda,
                        const double *B, npy_intp ldb,
                        double *C, npy_intp ldc,
                        int transa, int transb,
                        double alpha, double beta,
                        jblas_workspace_t *ws)
{
    jblas_dgemm_ws(M, N, K, A, lda, B, ldb, C, ldc,
                   transa, transb, alpha, beta, ws);
}

int blas_dispatch_lp64_overflow_count(void) {
    return 0;
}

void blas_dispatch_reset_lp64_overflow(void) {
}

#endif /* !_WIN32 */
