/**
 * blas_dispatch.c -- BLAS/LAPACK discovery and dispatch wrapper.
 *
 * Dispatch priority (consistency with GEMMA over raw speed):
 *   1. ILP64 with LAPACK (dsyevd): MKL-ILP64, Accelerate-ILP64
 *   2. ILP64 BLAS-only: BLIS-ILP64 (no LAPACK)
 *   3. jblas own blocking dgemm — consistent, no integer overflow concerns
 *   4. LP64 (detected but not wired for dgemm — different FP accumulation)
 *
 * Discovery model: discover-all-then-select-best.  All three discovery paths
 * (system BLAS, pip-installed MKL, bundled BLIS) run unconditionally.  The
 * best candidate is selected based on capabilities (ILP64 + LAPACK > ILP64
 * BLAS-only > jblas-own > LP64).
 *
 * When an external dgemm is found, replaces jblas_dispatch.dgemm with a
 * wrapper.  CBLAS backends handle row-major natively; Fortran backends
 * use the A/B swap trick for column-major conversion.
 *
 * The dlopen machinery is Unix-only (#if !defined(_WIN32)); on Windows
 * blas_dispatch_init() returns 0 immediately (no external dispatch).
 */

/* _GNU_SOURCE required on Linux for Dl_info and dladdr (used in
 * discover_bundled_blis).  Must be defined before any includes. */
#define _GNU_SOURCE

#include <limits.h>
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
static jblas_dgemm_lp64_fn       g_dgemm_lp64       = NULL;
static jblas_dgemm_ilp64_fn      g_dgemm_ilp64      = NULL;
static jblas_cblas_dgemm_fn      g_cblas_dgemm       = NULL;  /* LP64 CBLAS */
static jblas_cblas_dgemm_ilp64_fn g_cblas_dgemm_ilp64 = NULL;  /* ILP64 CBLAS (Accelerate) */
static const char *g_backend_name = "jblas-own";
static void *g_blas_handle = NULL;

/* dsyrk dispatch pointers — ILP64 only (LP64 not wired, same policy as dgemm) */
static jblas_cblas_dsyrk_ilp64_fn g_cblas_dsyrk_ilp64 = NULL;
static jblas_dsyrk_ilp64_fn       g_dsyrk_ilp64       = NULL;

/* dsyevd dispatch pointers (Fortran) */
static jblas_dsyevd_lp64_fn       g_dsyevd_lp64       = NULL;
static jblas_dsyevd_ilp64_fn      g_dsyevd_ilp64      = NULL;

/* LAPACKE dsyevd dispatch pointers (C interface, row-major) */
static jblas_lapacke_dsyevd_lp64_fn   g_lapacke_dsyevd_lp64   = NULL;
static jblas_lapacke_dsyevd_ilp64_fn  g_lapacke_dsyevd_ilp64  = NULL;

/* Capability flags */
static int g_has_dsyrk  = 0;
static int g_has_dsyevd = 0;
static int g_has_lapacke_dsyevd = 0;

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
    return is_ilp64 ? "Accelerate-ILP64" : "Accelerate";
#else
    return is_ilp64 ? "system-BLAS-ILP64" : "system-BLAS-LP64";
#endif
}

/* ---------------------------------------------------------------------------
 * Candidate struct for discover-all-then-select-best pattern
 * ---------------------------------------------------------------------------
 */
typedef struct {
    int     found;
    int     is_ilp64;
    int     has_lapack;     /* has LAPACK dsyevd (only routine currently resolved) */
    int     has_dsyrk;
    const char *name;
    void   *handle;
    /* dgemm */
    jblas_dgemm_lp64_fn        dgemm_lp64;
    jblas_dgemm_ilp64_fn       dgemm_ilp64;
    jblas_cblas_dgemm_fn       cblas_dgemm;
    jblas_cblas_dgemm_ilp64_fn cblas_dgemm_ilp64;
    /* dsyrk */
    jblas_cblas_dsyrk_fn       cblas_dsyrk;
    jblas_cblas_dsyrk_ilp64_fn cblas_dsyrk_ilp64;
    jblas_dsyrk_lp64_fn        dsyrk_lp64;
    jblas_dsyrk_ilp64_fn       dsyrk_ilp64;
    /* dsyevd (Fortran) */
    jblas_dsyevd_lp64_fn       dsyevd_lp64;
    jblas_dsyevd_ilp64_fn      dsyevd_ilp64;
    /* LAPACKE dsyevd (C interface, row-major — no transpose needed) */
    jblas_lapacke_dsyevd_lp64_fn   lapacke_dsyevd_lp64;
    jblas_lapacke_dsyevd_ilp64_fn  lapacke_dsyevd_ilp64;
    int     has_lapacke_dsyevd;
} blas_candidate_t;

/* ---------------------------------------------------------------------------
 * Symbol resolution — dgemm
 * ---------------------------------------------------------------------------
 */
static const char *ilp64_dgemm_names[] = {
    "dgemm_64_",              /* MKL ILP64 */
    "scipy_dgemm_64_",        /* scipy-openblas64 */
    "dgemm64_",               /* OpenBLAS INTERFACE64=1, BLIS -b 64 */
    NULL
};
/* Apple Accelerate ILP64 (macOS 13.3+): uses $NEWLAPACK$ILP64 suffix.
 * Fortran interface has no trailing underscore. */
static const char *accel_ilp64_dgemm_names[] = {
    "dgemm$NEWLAPACK$ILP64",
    NULL
};
static const char *accel_ilp64_cblas_names[] = {
    "cblas_dgemm$NEWLAPACK$ILP64",
    NULL
};
static const char *lp64_dgemm_names[] = {
    "dgemm_",                 /* Standard Fortran / Accelerate */
    NULL
};

/**
 * try_resolve_dgemm_candidate -- Try to resolve dgemm from a dlopen handle.
 * Populates the candidate struct instead of globals.
 * Returns 1 if found, 0 if not.
 *
 * lib_path: hint for backend name detection (may be NULL for RTLD_DEFAULT).
 */
static int try_resolve_dgemm_candidate(void *handle, const char *lib_path,
                                        blas_candidate_t *c) {
    int dbg = _debug_enabled();

    /* Try ILP64 symbols first (MKL, OpenBLAS) */
    for (const char **name = ilp64_dgemm_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved %s\n", *name);
            c->dgemm_ilp64 = (jblas_dgemm_ilp64_fn)sym;
            c->is_ilp64 = 1;
            c->name = _detect_backend_name(lib_path, 1);
            c->found = 1;
            c->handle = handle;
            return 1;
        }
    }

    /* Try Apple Accelerate ILP64 (macOS 13.3+) — prefer CBLAS for row-major */
    for (const char **name = accel_ilp64_cblas_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved %s (Accelerate ILP64 CBLAS)\n", *name);
            c->cblas_dgemm_ilp64 = (jblas_cblas_dgemm_ilp64_fn)sym;
            c->is_ilp64 = 1;
            c->name = "Accelerate-ILP64";
            c->found = 1;
            c->handle = handle;
            /* Also try Fortran interface as fallback */
            for (const char **fn = accel_ilp64_dgemm_names; *fn; fn++) {
                void *fsym = dlsym(handle, *fn);
                if (fsym) {
                    c->dgemm_ilp64 = (jblas_dgemm_ilp64_fn)fsym;
                    if (dbg) fprintf(stderr, "jblas_dispatch:   also resolved %s\n", *fn);
                }
            }
            return 1;
        }
    }

    /* Try LP64 symbols */
    for (const char **name = lp64_dgemm_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved %s\n", *name);
            c->dgemm_lp64 = (jblas_dgemm_lp64_fn)sym;
            c->is_ilp64 = 0;
            c->name = _detect_backend_name(lib_path, 0);
            c->found = 1;
            c->handle = handle;

            /* Also try cblas_dgemm — row-major native, no A/B swap needed. */
            void *cblas_sym = dlsym(handle, "cblas_dgemm");
            if (cblas_sym) {
                c->cblas_dgemm = (jblas_cblas_dgemm_fn)cblas_sym;
                if (dbg) fprintf(stderr, "jblas_dispatch:   also resolved cblas_dgemm\n");
            }
            return 1;
        }
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dsyrk
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dsyrk(void *handle, blas_candidate_t *c, int is_blis) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
        /* ILP64 dsyrk symbols */
#ifdef __APPLE__
        /* Accelerate ILP64 */
        void *sym = dlsym(handle, "cblas_dsyrk$NEWLAPACK$ILP64");
        if (sym) {
            c->cblas_dsyrk_ilp64 = (jblas_cblas_dsyrk_ilp64_fn)sym;
            c->has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved cblas_dsyrk$NEWLAPACK$ILP64\n");
            /* Also try Fortran */
            void *fsym = dlsym(handle, "dsyrk$NEWLAPACK$ILP64");
            if (fsym) {
                c->dsyrk_ilp64 = (jblas_dsyrk_ilp64_fn)fsym;
                if (dbg) fprintf(stderr, "jblas_dispatch:   also resolved dsyrk$NEWLAPACK$ILP64\n");
            }
            return;
        }
#endif
        /* MKL ILP64 */
        void *sym64 = dlsym(handle, "dsyrk_64_");
        if (sym64) {
            c->dsyrk_ilp64 = (jblas_dsyrk_ilp64_fn)sym64;
            c->has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyrk_64_\n");
            return;
        }
        /* OpenBLAS ILP64 */
        void *sym64b = dlsym(handle, "dsyrk64_");
        if (sym64b) {
            c->dsyrk_ilp64 = (jblas_dsyrk_ilp64_fn)sym64b;
            c->has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyrk64_\n");
            return;
        }
        /* BLIS with ILP64: cblas_dsyrk takes 64-bit int args */
        if (is_blis) {
            void *bsym = dlsym(handle, "cblas_dsyrk");
            if (bsym) {
                c->cblas_dsyrk_ilp64 = (jblas_cblas_dsyrk_ilp64_fn)bsym;
                c->has_dsyrk = 1;
                if (dbg) fprintf(stderr, "jblas_dispatch:   resolved cblas_dsyrk (BLIS ILP64)\n");
                return;
            }
        }
    }

    /* LP64 dsyrk symbols */
    void *csym = dlsym(handle, "cblas_dsyrk");
    if (csym && !c->is_ilp64) {
        c->cblas_dsyrk = (jblas_cblas_dsyrk_fn)csym;
        c->has_dsyrk = 1;
        if (dbg) fprintf(stderr, "jblas_dispatch:   resolved cblas_dsyrk (LP64)\n");
        return;
    }
    void *fsym = dlsym(handle, "dsyrk_");
    if (fsym && !c->is_ilp64) {
        c->dsyrk_lp64 = (jblas_dsyrk_lp64_fn)fsym;
        c->has_dsyrk = 1;
        if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyrk_ (LP64)\n");
        return;
    }
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dsyevd
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dsyevd(void *handle, blas_candidate_t *c, int is_blis) {
    int dbg = _debug_enabled();

    /* BLIS has no LAPACK — skip */
    if (is_blis)
        return;

    if (c->is_ilp64) {
#ifdef __APPLE__
        /* Accelerate ILP64: Fortran only (no LAPACKE in Accelerate) */
        void *sym = dlsym(handle, "dsyevd$NEWLAPACK$ILP64");
        if (sym) {
            c->dsyevd_ilp64 = (jblas_dsyevd_ilp64_fn)sym;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyevd$NEWLAPACK$ILP64\n");
        }
        /* No LAPACKE on Accelerate — skip LAPACKE resolution */
        return;
#endif
        /* MKL/OpenBLAS ILP64: try LAPACKE first (C interface, row-major) */
        void *le64 = dlsym(handle, "LAPACKE_dsyevd");
        if (le64) {
            /* When loaded from an ILP64 library, LAPACKE_dsyevd uses
             * lapack_int = long long.  Cast to our ILP64 typedef. */
            c->lapacke_dsyevd_ilp64 = (jblas_lapacke_dsyevd_ilp64_fn)le64;
            c->has_lapacke_dsyevd = 1;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved LAPACKE_dsyevd (ILP64)\n");
        }

        /* Also resolve Fortran dsyevd as fallback */
        void *sym64 = dlsym(handle, "dsyevd_64_");
        if (sym64) {
            c->dsyevd_ilp64 = (jblas_dsyevd_ilp64_fn)sym64;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyevd_64_\n");
            return;
        }
        /* OpenBLAS ILP64 */
        void *sym64b = dlsym(handle, "dsyevd64_");
        if (sym64b) {
            c->dsyevd_ilp64 = (jblas_dsyevd_ilp64_fn)sym64b;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyevd64_\n");
            return;
        }
        return;
    }

    /* LP64: try LAPACKE first */
    void *le = dlsym(handle, "LAPACKE_dsyevd");
    if (le) {
        c->lapacke_dsyevd_lp64 = (jblas_lapacke_dsyevd_lp64_fn)le;
        c->has_lapacke_dsyevd = 1;
        c->has_lapack = 1;
        if (dbg) fprintf(stderr, "jblas_dispatch:   resolved LAPACKE_dsyevd (LP64)\n");
    }

    /* LP64 Fortran dsyevd */
    void *fsym = dlsym(handle, "dsyevd_");
    if (fsym) {
        c->dsyevd_lp64 = (jblas_dsyevd_lp64_fn)fsym;
        c->has_lapack = 1;
        if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dsyevd_ (LP64)\n");
    }
}

/* ---------------------------------------------------------------------------
 * Directory scanning (populates candidate)
 * ---------------------------------------------------------------------------
 */

/**
 * scan_dir_for_blas_candidate -- Scan a directory for BLAS-providing shared libraries.
 * Returns 1 if dgemm was resolved, 0 if not.
 */
static int scan_dir_for_blas_candidate(const char *dirpath, blas_candidate_t *c) {
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

        if (try_resolve_dgemm_candidate(handle, fullpath, c)) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dgemm from %s (ilp64=%d)\n",
                             fullpath, c->is_ilp64);
            try_resolve_dsyrk(handle, c, 0);
            try_resolve_dsyevd(handle, c, 0);
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
static int scan_proc_maps_for_blas_candidate(blas_candidate_t *c) {
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

        if (try_resolve_dgemm_candidate(handle, path, c)) {
            if (dbg) fprintf(stderr, "jblas_dispatch:   resolved dgemm from /proc/self/maps (ilp64=%d)\n", c->is_ilp64);
            try_resolve_dsyrk(handle, c, 0);
            try_resolve_dsyevd(handle, c, 0);
            fclose(fp);
            return 1;
        }
        if (dbg) fprintf(stderr, "jblas_dispatch:   dgemm not found in %s\n", basename);
        dlclose(handle);
    }
    fclose(fp);
#else
    (void)c;
#endif
    return 0;
}

/* ---------------------------------------------------------------------------
 * discover_system_blas -- Full system BLAS discovery (4-step pattern)
 * Populates a blas_candidate_t instead of setting globals.
 * ---------------------------------------------------------------------------
 */
static void discover_system_blas(blas_candidate_t *c) {
    int dbg = _debug_enabled();

    /* Step 1: RTLD_DEFAULT (catches macOS Accelerate, LD_PRELOAD) */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 1 -- RTLD_DEFAULT\n");
    if (try_resolve_dgemm_candidate(RTLD_DEFAULT, NULL, c)) {
        if (dbg) fprintf(stderr, "jblas_dispatch: found via RTLD_DEFAULT (ilp64=%d, backend=%s)\n",
                         c->is_ilp64, c->name);
        try_resolve_dsyrk(RTLD_DEFAULT, c, 0);
        try_resolve_dsyevd(RTLD_DEFAULT, c, 0);
        return;
    }

    /* Step 2: Force numpy to load its BLAS, then retry RTLD_DEFAULT */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 2 -- force numpy BLAS load\n");
    force_numpy_blas_load();
    if (try_resolve_dgemm_candidate(RTLD_DEFAULT, NULL, c)) {
        if (dbg) fprintf(stderr, "jblas_dispatch: found via RTLD_DEFAULT after numpy load (ilp64=%d, backend=%s)\n",
                         c->is_ilp64, c->name);
        try_resolve_dsyrk(RTLD_DEFAULT, c, 0);
        try_resolve_dsyevd(RTLD_DEFAULT, c, 0);
        return;
    }

    /* Step 3: /proc/self/maps scan (Linux only) */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 3 -- /proc/self/maps scan\n");
    if (scan_proc_maps_for_blas_candidate(c)) {
        if (dbg) fprintf(stderr, "jblas_dispatch: found via /proc/self/maps (ilp64=%d, backend=%s)\n",
                         c->is_ilp64, c->name);
        return;
    }

    /* Step 4: Scan numpy's lib directories */
    if (dbg) fprintf(stderr, "jblas_dispatch: step 4 -- numpy dir scan\n");
    PyObject *np2 = PyImport_ImportModule("numpy");
    if (!np2) { PyErr_Clear(); return; }

    PyObject *np_file = PyObject_GetAttrString(np2, "__file__");
    if (!np_file) { PyErr_Clear(); Py_DECREF(np2); return; }

    PyObject *pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) { PyErr_Clear(); Py_DECREF(np_file); Py_DECREF(np2); return; }

    PyObject *Path = PyObject_GetAttrString(pathlib, "Path");
    if (!Path) { PyErr_Clear(); Py_DECREF(pathlib); Py_DECREF(np_file); Py_DECREF(np2); return; }

    PyObject *p = PyObject_CallFunctionObjArgs(Path, np_file, NULL);
    Py_DECREF(np_file);
    if (!p) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return; }

    PyObject *resolved = PyObject_CallMethod(p, "resolve", NULL);
    Py_DECREF(p);
    if (!resolved) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return; }

    PyObject *np_dir = PyObject_GetAttrString(resolved, "parent");
    Py_DECREF(resolved);
    if (!np_dir) { PyErr_Clear(); Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2); return; }

    const char *subpaths[] = { ".libs", "_core/.libs", NULL };
    for (int si = 0; subpaths[si]; si++) {
        PyObject *candidate = PyObject_CallMethod(np_dir, "__truediv__", "s", subpaths[si]);
        if (!candidate) { PyErr_Clear(); continue; }
        PyObject *cstr = PyObject_Str(candidate);
        Py_DECREF(candidate);
        if (!cstr) { PyErr_Clear(); continue; }
        const char *dirpath = PyUnicode_AsUTF8(cstr);
        if (dirpath && scan_dir_for_blas_candidate(dirpath, c)) {
            Py_DECREF(cstr); Py_DECREF(np_dir); Py_DECREF(Path);
            Py_DECREF(pathlib); Py_DECREF(np2);
            return;
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
                if (dirpath && scan_dir_for_blas_candidate(dirpath, c)) {
                    Py_DECREF(cstr); Py_DECREF(np_parent); Py_DECREF(np_dir);
                    Py_DECREF(Path); Py_DECREF(pathlib); Py_DECREF(np2);
                    return;
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
}

/* ---------------------------------------------------------------------------
 * discover_pip_mkl -- Look for pip-installed MKL (site-packages/mkl)
 * ---------------------------------------------------------------------------
 */
static void discover_pip_mkl(blas_candidate_t *c) {
    int dbg = _debug_enabled();
    if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- trying import mkl\n");

    PyObject *mkl = PyImport_ImportModule("mkl");
    if (!mkl) {
        PyErr_Clear();
        if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- mkl module not found\n");
        return;
    }

    PyObject *mkl_file = PyObject_GetAttrString(mkl, "__file__");
    Py_DECREF(mkl);
    if (!mkl_file) {
        PyErr_Clear();
        if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- mkl.__file__ not found\n");
        return;
    }

    PyObject *pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) { PyErr_Clear(); Py_DECREF(mkl_file); return; }

    PyObject *Path = PyObject_GetAttrString(pathlib, "Path");
    Py_DECREF(pathlib);
    if (!Path) { PyErr_Clear(); Py_DECREF(mkl_file); return; }

    PyObject *mkl_path = PyObject_CallFunctionObjArgs(Path, mkl_file, NULL);
    Py_DECREF(mkl_file);
    if (!mkl_path) { PyErr_Clear(); Py_DECREF(Path); return; }

    PyObject *mkl_dir = PyObject_GetAttrString(mkl_path, "parent");
    Py_DECREF(mkl_path);
    if (!mkl_dir) { PyErr_Clear(); Py_DECREF(Path); return; }

    /* Try mkl_dir / 'mkl.libs' and mkl_dir.parent / 'mkl.libs' */
    const char *mkl_lib_paths[] = { "mkl.libs", NULL };

    for (int attempt = 0; attempt < 2; attempt++) {
        PyObject *base = attempt == 0 ? mkl_dir : PyObject_GetAttrString(mkl_dir, "parent");
        if (!base) { PyErr_Clear(); continue; }

        for (int pi = 0; mkl_lib_paths[pi]; pi++) {
            PyObject *libs_dir = PyObject_CallMethod(base, "__truediv__", "s", mkl_lib_paths[pi]);
            if (!libs_dir) { PyErr_Clear(); continue; }
            PyObject *libs_str = PyObject_Str(libs_dir);
            Py_DECREF(libs_dir);
            if (!libs_str) { PyErr_Clear(); continue; }
            const char *dirpath = PyUnicode_AsUTF8(libs_str);
            if (!dirpath) { Py_DECREF(libs_str); continue; }

            if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- trying dir: %s\n", dirpath);

            /* MKL libraries must be loaded in dependency order:
             * core first, then sequential, then ilp64 */
            const char *mkl_libs[] = {
                "libmkl_core",
                "libmkl_sequential",
                "libmkl_intel_ilp64",
                NULL
            };
            void *last_handle = NULL;

            for (int li = 0; mkl_libs[li]; li++) {
                /* Scan directory for matching .so/.dylib */
                DIR *dir = opendir(dirpath);
                if (!dir) break;

                struct dirent *entry;
                while ((entry = readdir(dir)) != NULL) {
                    if (!strstr(entry->d_name, mkl_libs[li]))
                        continue;
                    if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib"))
                        continue;

                    char fullpath[4096];
                    snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
                    if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- dlopen %s\n", fullpath);

                    void *h = dlopen(fullpath, RTLD_LAZY | RTLD_GLOBAL);
                    if (h) {
                        last_handle = h;
                        if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- loaded %s\n", entry->d_name);
                    } else {
                        if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- dlopen failed: %s\n", dlerror());
                    }
                    break;
                }
                closedir(dir);
            }

            if (last_handle) {
                /* Try to resolve symbols from RTLD_DEFAULT (all loaded globally) */
                if (try_resolve_dgemm_candidate(RTLD_DEFAULT, dirpath, c)) {
                    if (!c->is_ilp64) {
                        /* Loaded ILP64 MKL libs but only resolved LP64 symbols.
                         * Don't label as ILP64 — would cause ABI mismatch. */
                        if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- "
                                         "WARNING: resolved LP64 dgemm from ILP64 MKL path, skipping\n");
                        c->found = 0;
                        Py_DECREF(libs_str);
                        if (attempt == 1) Py_DECREF(base);
                        Py_DECREF(mkl_dir); Py_DECREF(Path);
                        return;
                    }
                    c->name = "MKL-ILP64";
                    try_resolve_dsyrk(RTLD_DEFAULT, c, 0);
                    try_resolve_dsyevd(RTLD_DEFAULT, c, 0);
                    if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- resolved (ilp64=%d, lapack=%d)\n",
                                     c->is_ilp64, c->has_lapack);
                    Py_DECREF(libs_str);
                    if (attempt == 1) Py_DECREF(base);
                    Py_DECREF(mkl_dir); Py_DECREF(Path);
                    return;
                }
            }
            Py_DECREF(libs_str);
        }
        if (attempt == 1) Py_DECREF(base);
    }

    Py_DECREF(mkl_dir); Py_DECREF(Path);
    if (dbg) fprintf(stderr, "jblas_dispatch: pip-mkl -- not found\n");
}

/* ---------------------------------------------------------------------------
 * discover_bundled_blis -- Look for libblis.{so,dylib} relative to extension
 * Populates a blas_candidate_t instead of setting globals.
 * ---------------------------------------------------------------------------
 */
static void discover_bundled_blis(blas_candidate_t *c) {
    int dbg = _debug_enabled();

    /* Use dladdr on blas_dispatch_init to find our own .so path */
    Dl_info info;
    if (!dladdr((void *)blas_dispatch_init, &info) || !info.dli_fname) {
        if (dbg) fprintf(stderr, "jblas_dispatch: dladdr failed for blas_dispatch_init\n");
        return;
    }

    /* Build path: dirname(extension.so)/libs/libblis.{so,dylib} */
    char ext_dir[4096];
    strncpy(ext_dir, info.dli_fname, sizeof(ext_dir) - 1);
    ext_dir[sizeof(ext_dir) - 1] = '\0';
    char *last_slash = strrchr(ext_dir, '/');
    if (!last_slash) return;
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
        return;
    }

    /* BLIS with -b 64 exports dgemm_ with 64-bit int args (same symbol name,
     * different ABI).  Check bli_info_get_int_type_size() to detect. */
    typedef long (*bli_info_int_size_fn)(void);
    bli_info_int_size_fn get_int_size =
        (bli_info_int_size_fn)dlsym(handle, "bli_info_get_int_type_size");
    int blis_is_ilp64 = (get_int_size && get_int_size() == 64);

    if (dbg) fprintf(stderr, "jblas_dispatch: BLIS bli_info_get_int_type_size=%s%s\n",
                     get_int_size ? "" : "(not found)",
                     blis_is_ilp64 ? "64" : get_int_size ? "32" : "");

    if (blis_is_ilp64) {
        /* BLIS ILP64: dgemm_ takes 64-bit int pointers — resolve as ILP64 */
        void *sym = dlsym(handle, "dgemm_");
        if (sym) {
            c->dgemm_ilp64 = (jblas_dgemm_ilp64_fn)sym;
            c->dgemm_lp64 = NULL;
            c->is_ilp64 = 1;
            c->name = "BLIS-ILP64";
            c->handle = handle;
            c->found = 1;

            /* Also resolve cblas_dgemm for row-major native dispatch. */
            void *cblas_sym = dlsym(handle, "cblas_dgemm");
            if (cblas_sym) {
                c->cblas_dgemm_ilp64 = (jblas_cblas_dgemm_ilp64_fn)cblas_sym;
                if (dbg) fprintf(stderr, "jblas_dispatch:   also resolved cblas_dgemm (ILP64)\n");
            }

            /* BLIS has dsyrk but no LAPACK */
            try_resolve_dsyrk(handle, c, 1);
            /* BLIS has no LAPACK — skip dsyevd */

            if (dbg) fprintf(stderr, "jblas_dispatch: resolved dgemm from bundled BLIS-ILP64\n");
            return;
        }
    }

    /* LP64 BLIS or ILP64 without dgemm_ — try normal resolution */
    if (try_resolve_dgemm_candidate(handle, blis_path, c)) {
        c->name = "BLIS";
        try_resolve_dsyrk(handle, c, 1);
        /* No dsyevd for BLIS */
        if (dbg) fprintf(stderr, "jblas_dispatch: resolved dgemm from bundled BLIS (LP64)\n");
        return;
    }

    if (dbg) fprintf(stderr, "jblas_dispatch: dgemm not found in bundled BLIS\n");
    dlclose(handle);
}

/* ---------------------------------------------------------------------------
 * select_best_backend -- Choose the highest-capability candidate.
 *
 * Scoring: ILP64 + LAPACK = 4, ILP64 BLAS-only = 3, LP64 = 1, not found = 0
 * Returns pointer to highest-scoring candidate, or NULL if no candidates found.
 * LP64 candidates (score=1) are returned for logging but not wired for dgemm
 * by the caller.
 * ---------------------------------------------------------------------------
 */
/* Validate candidate invariants.  Returns 1 if valid, 0 if inconsistent.
 * When invalid, zeros out the candidate (found=0) so it cannot be selected
 * — prevents NULL function pointer dereferences from broken discovery. */
static int _validate_candidate(blas_candidate_t *c, const char *label) {
    if (!c->found) return 1;  /* not-found is always valid */
    int valid = 1;
    /* found=1 requires at least one dgemm pointer */
    if (!c->dgemm_lp64 && !c->dgemm_ilp64 &&
        !c->cblas_dgemm && !c->cblas_dgemm_ilp64) {
        fprintf(stderr, "jblas_dispatch: WARN: %s found=1 but no dgemm pointers — disabling\n", label);
        valid = 0;
    }
    /* has_lapack requires at least one dsyevd pointer */
    if (c->has_lapack && !c->dsyevd_lp64 && !c->dsyevd_ilp64 &&
        !c->lapacke_dsyevd_lp64 && !c->lapacke_dsyevd_ilp64) {
        fprintf(stderr, "jblas_dispatch: WARN: %s has_lapack=1 but no dsyevd pointers — disabling\n", label);
        valid = 0;
    }
    /* has_lapacke_dsyevd requires has_lapack */
    if (c->has_lapacke_dsyevd && !c->has_lapack) {
        fprintf(stderr, "jblas_dispatch: WARN: %s has_lapacke_dsyevd=1 but has_lapack=0 — disabling\n", label);
        valid = 0;
    }
    /* has_dsyrk requires at least one dsyrk pointer */
    if (c->has_dsyrk && !c->cblas_dsyrk && !c->cblas_dsyrk_ilp64 &&
        !c->dsyrk_lp64 && !c->dsyrk_ilp64) {
        fprintf(stderr, "jblas_dispatch: WARN: %s has_dsyrk=1 but no dsyrk pointers — disabling\n", label);
        valid = 0;
    }
    if (!valid) {
        /* Zero out the candidate so it cannot be selected */
        const char *saved_name = c->name;
        memset(c, 0, sizeof(*c));
        c->name = saved_name;  /* preserve for diagnostic logging */
    }
    return valid;
}

static int _score_candidate(const blas_candidate_t *c) {
    if (!c->found) return 0;
    if (c->is_ilp64 && c->has_lapack) return 4;
    if (c->is_ilp64) return 3;
    return 1;  /* LP64 */
}

static blas_candidate_t *select_best_backend(blas_candidate_t *system,
                                              blas_candidate_t *pip_mkl,
                                              blas_candidate_t *blis) {
    int s_sys  = _score_candidate(system);
    int s_pip  = _score_candidate(pip_mkl);
    int s_blis = _score_candidate(blis);
    int dbg = _debug_enabled();

    if (dbg) fprintf(stderr, "jblas_dispatch: scores: system=%d pip_mkl=%d blis=%d\n",
                     s_sys, s_pip, s_blis);

    blas_candidate_t *best = NULL;
    int best_score = 0;

    if (s_sys > best_score)  { best = system;  best_score = s_sys; }
    if (s_pip > best_score)  { best = pip_mkl; best_score = s_pip; }
    if (s_blis > best_score) { best = blis;    best_score = s_blis; }

    /* LP64-only candidates (score=1) are detected but not wired for dgemm */
    if (best_score <= 1 && best && !best->is_ilp64)
        return best;  /* Return it so blas_dispatch_init can log the LP64 info */

    return best;
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

    /* Prefer CBLAS: row-major native, no swap needed. */
    if (g_cblas_dgemm_ilp64) {
        long lk = k > 0 ? (long)k : 1;
        long ln = n > 0 ? (long)n : 1;
        g_cblas_dgemm_ilp64(JBLAS_CblasRowMajor,
                            JBLAS_CblasNoTrans, JBLAS_CblasNoTrans,
                            (long)m, (long)n, (long)k,
                            alpha, A, lk, B, ln,
                            beta,  C, ln);
        return;
    }
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
 * Public API — dispatch init (discover-all-then-select-best)
 * ---------------------------------------------------------------------------
 */

int blas_dispatch_init(void) {
    int dbg = _debug_enabled();

    blas_candidate_t system  = {0};
    blas_candidate_t pip_mkl = {0};
    blas_candidate_t blis    = {0};

    /* All three discovery paths run unconditionally */
    discover_system_blas(&system);
    discover_pip_mkl(&pip_mkl);
    discover_bundled_blis(&blis);

    /* Validate invariants — invalid candidates are zeroed out (found=0)
     * so they cannot be selected, preventing NULL dereferences. */
    _validate_candidate(&system,  "system");
    _validate_candidate(&pip_mkl, "pip_mkl");
    _validate_candidate(&blis,    "blis");

    blas_candidate_t *best = select_best_backend(&system, &pip_mkl, &blis);

    if (best && best->is_ilp64) {
        /* ILP64 backend — wire dgemm */
        if (dbg) fprintf(stderr, "jblas_dispatch: using %s (ILP64) for dgemm\n", best->name);
        g_dgemm_ilp64 = best->dgemm_ilp64;
        g_dgemm_lp64 = best->dgemm_lp64;
        g_cblas_dgemm = best->cblas_dgemm;
        g_cblas_dgemm_ilp64 = best->cblas_dgemm_ilp64;
        g_is_ilp64 = 1;
        g_backend_name = best->name;
        g_blas_handle = best->handle;
        jblas_dispatch.dgemm = _dgemm_external_wrapper;

        /* Wire dsyrk — only ILP64 pointers (LP64 dsyrk is not dispatched,
         * same policy as dgemm: jblas-own preferred for LP64 consistency) */
        if (best->has_dsyrk) {
            g_cblas_dsyrk_ilp64 = best->cblas_dsyrk_ilp64;
            g_dsyrk_ilp64 = best->dsyrk_ilp64;
            g_has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jblas_dispatch: vendor dsyrk wired\n");
        }

        /* Wire dsyevd — prefer LAPACKE (C, row-major) over Fortran */
        if (best->has_lapack) {
            g_dsyevd_lp64 = best->dsyevd_lp64;
            g_dsyevd_ilp64 = best->dsyevd_ilp64;
            g_has_dsyevd = 1;
            if (best->has_lapacke_dsyevd) {
                g_lapacke_dsyevd_lp64 = best->lapacke_dsyevd_lp64;
                g_lapacke_dsyevd_ilp64 = best->lapacke_dsyevd_ilp64;
                g_has_lapacke_dsyevd = 1;
                if (dbg) fprintf(stderr, "jblas_dispatch: vendor LAPACKE dsyevd wired (row-major)\n");
            } else {
                if (dbg) fprintf(stderr, "jblas_dispatch: vendor dsyevd wired (Fortran, transpose required)\n");
            }
        }

        return 0;
    }

    if (best && best->found && !best->is_ilp64) {
        /* LP64 found but not ILP64 — prefer jblas-own for consistency */
        if (dbg) fprintf(stderr, "jblas_dispatch: LP64 %s available but preferring jblas-own for consistency\n",
                         best->name);
        fprintf(stderr,
            "jblas_dispatch: INFO: LP64 BLAS (%s) detected but not used — "
            "jblas own dgemm preferred for numerical consistency with GEMMA. "
            "Install ILP64 numpy for faster external BLAS dispatch.\n",
            best->name);
        /* Reset backend name — LP64 is available but not active */
        g_backend_name = "jblas-own";
        return 0;
    }

    /* No external dgemm found — jblas own stays */
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
    /* Only true when external BLAS is actually wired into the dispatch table
     * (i.e., ILP64 found).  LP64-only discovery does not wire dispatch. */
    return jblas_dispatch.dgemm != jblas_dgemm_dispatch_fn;
}

int blas_has_dsyrk(void)  { return g_has_dsyrk; }
int blas_has_dsyevd(void) { return g_has_dsyevd; }
int blas_has_lapacke_dsyevd(void) { return g_has_lapacke_dsyevd; }

/* ---------------------------------------------------------------------------
 * jblas_dsyrk_ext — Vendor-dispatch dsyrk: C = X @ X.T
 * ---------------------------------------------------------------------------
 */
void jblas_dsyrk_ext(npy_intp N, npy_intp K,
                     const double *X, npy_intp ldx,
                     double *C, npy_intp ldc)
{
    if (N <= 0 || K <= 0) {
        /* Zero C for N>0, K==0 case (consistent with jblas_dsyrk_c) */
        for (npy_intp i = 0; i < N; i++)
            memset(C + i * ldc, 0, (size_t)N * sizeof(double));
        return;
    }
    if (g_has_dsyrk && g_is_ilp64) {
        if (g_cblas_dsyrk_ilp64) {
            /* Row-major, lower, no-trans: C = 1.0 * X @ X.T + 0.0 * C */
            g_cblas_dsyrk_ilp64(JBLAS_CblasRowMajor, JBLAS_CblasLower,
                                JBLAS_CblasNoTrans,
                                (long)N, (long)K,
                                1.0, X, (long)ldx,
                                0.0, C, (long)ldc);
            /* Mirror lower to upper (vendor only fills lower) */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++)
                    C[i * ldc + j] = C[j * ldc + i];
            return;
        }
        /* Fortran ILP64 fallback: row-major lower = col-major upper */
        if (g_dsyrk_ilp64) {
            const long long n = (long long)N, k = (long long)K;
            const long long lda = (long long)ldx, ldc_f = (long long)ldc;
            const double alpha = 1.0, beta = 0.0;
            g_dsyrk_ilp64("U", "T", &n, &k, &alpha, X, &lda, &beta, C, &ldc_f);
            /* Fortran col-major upper = row-major lower; mirror lower to upper */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++)
                    C[i * ldc + j] = C[j * ldc + i];
            return;
        }
    }
    /* Fallback to jblas own */
    jblas_dsyrk_c(N, K, X, ldx, C, ldc);
}

/* ---------------------------------------------------------------------------
 * jblas_dsyevd_ext — Vendor-dispatch dsyevd for eigh
 *
 * Prefers LAPACKE C interface (row-major, no transpose) when available (MKL).
 * Falls back to Fortran dsyevd + eigenvector transpose (Accelerate, OpenBLAS).
 *
 * Input: K is row-major symmetric, lower triangle populated.
 * Output: K overwritten with eigenvectors stored columnwise in row-major
 *         (K[i*ldk+j] = component i of eigenvector j).
 *         eigenvalues[k] = k-th eigenvalue, ascending.
 *
 * Returns: JBLAS_EXT_SUCCESS, JBLAS_EXT_UNAVAILABLE, JBLAS_EXT_ALLOC_FAIL,
 *          or positive int for LAPACK error (info capped to INT_MAX for ILP64).
 * ---------------------------------------------------------------------------
 */

/* Safely narrow LAPACK info (long long) to int return.  Logs the full value
 * when truncation would occur (ILP64 eigenvalue index > INT_MAX). */
static int _info_to_int(long long info, npy_intp N) {
    if (info > INT_MAX || info < INT_MIN) {
        fprintf(stderr,
            "jblas_dsyevd_ext: LAPACK info=%lld exceeds int range (N=%ld) "
            "— returning capped value\n", info, (long)N);
        return info > 0 ? INT_MAX : INT_MIN;
    }
    return (int)info;
}

int jblas_dsyevd_ext(npy_intp N, double *K, npy_intp ldk,
                     double *eigenvalues)
{
    if (!g_has_dsyevd || !g_is_ilp64)
        return JBLAS_EXT_UNAVAILABLE;

    /* --- LAPACKE path (MKL): row-major natively, no transpose needed.
     * Only used when Fortran ILP64 dsyevd is NOT available.  When both
     * exist, we prefer Fortran because dsyevd_64_ is an unambiguous ILP64
     * symbol, whereas LAPACKE_dsyevd is unsuffixed and could resolve to
     * the LP64 variant on systems with mixed LP64/ILP64 MKL. --- */
    if (g_has_lapacke_dsyevd && g_lapacke_dsyevd_ilp64 && !g_dsyevd_ilp64) {
        long long info = g_lapacke_dsyevd_ilp64(
            JBLAS_LAPACK_ROW_MAJOR, 'V', 'L',
            (long long)N, K, (long long)ldk, eigenvalues);
        if (info != 0) return _info_to_int(info, N);
        return JBLAS_EXT_SUCCESS;
    }

    /* --- Fortran path (Accelerate, MKL, OpenBLAS): col-major + transpose.
     * Preferred when available because ILP64 symbol names (dsyevd_64_,
     * dsyevd$NEWLAPACK$ILP64) are unambiguous — no LP64/ILP64 confusion. --- */
    if (g_dsyevd_ilp64) {
        long long n = (long long)N;
        long long lda = (long long)ldk;
        long long info = 0;

        /* Workspace query */
        long long lwork = -1, liwork = -1;
        double work_query;
        long long iwork_query;
        g_dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues,
                        &work_query, &lwork, &iwork_query, &liwork, &info);
        if (info != 0) {
            fprintf(stderr,
                "jblas_dsyevd_ext: Fortran dsyevd workspace query failed "
                "(info=%lld, N=%lld) — likely ABI mismatch or corrupt LAPACK\n",
                info, n);
            return (int)info;
        }

        lwork = (long long)work_query + 1;  /* +1 for double→integer rounding */
        liwork = iwork_query;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        long long *iwork = (long long *)malloc((size_t)liwork * sizeof(long long));
        if (!work || !iwork) { free(work); free(iwork); return JBLAS_EXT_ALLOC_FAIL; }

        /* Compute: UPLO='U' because row-major lower = col-major upper.
         * The matrix is symmetric so A = A^T — no input transpose needed,
         * just the UPLO swap. */
        g_dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues,
                        work, &lwork, iwork, &liwork, &info);
        free(work);
        free(iwork);
        if (info != 0) return _info_to_int(info, N);

        /* Transpose eigenvectors: Fortran dsyevd writes eigenvectors as
         * columns in column-major layout.  In our row-major memory, those
         * columns appear as rows.  Transpose to get the standard row-major
         * columnwise convention (K[i*ldk+j] = component i of eigvec j). */
        for (npy_intp i = 0; i < N; i++)
            for (npy_intp j = i + 1; j < N; j++) {
                double tmp = K[i * ldk + j];
                K[i * ldk + j] = K[j * ldk + i];
                K[j * ldk + i] = tmp;
            }
        return JBLAS_EXT_SUCCESS;
    }
    return JBLAS_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * Full-signature external dgemm wrapper
 * ---------------------------------------------------------------------------
 */
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

    if (g_cblas_dgemm_ilp64) {
        int ta = transa ? JBLAS_CblasTrans : JBLAS_CblasNoTrans;
        int tb = transb ? JBLAS_CblasTrans : JBLAS_CblasNoTrans;
        long llda = (long)(lda > 0 ? lda : 1);
        long lldb = (long)(ldb > 0 ? ldb : 1);
        long lldc = (long)(ldc > 0 ? ldc : 1);
        g_cblas_dgemm_ilp64(JBLAS_CblasRowMajor, ta, tb,
                            (long)M, (long)N, (long)K,
                            alpha, A, llda, B, lldb,
                            beta,  C, lldc);
        return 1;
    }
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

    /* Fortran interface fallback: row-major -> column-major swap */
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

/* Windows: no external dispatch -- always use jblas own */
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

int blas_has_dsyrk(void) {
    return 0;
}

int blas_has_dsyevd(void) {
    return 0;
}

int blas_has_lapacke_dsyevd(void) {
    return 0;
}

void jblas_dsyrk_ext(npy_intp N, npy_intp K,
                     const double *X, npy_intp ldx,
                     double *C, npy_intp ldc)
{
    jblas_dsyrk_c(N, K, X, ldx, C, ldc);
}

int jblas_dsyevd_ext(npy_intp N, double *K, npy_intp ldk,
                     double *eigenvalues)
{
    (void)N; (void)K; (void)ldk; (void)eigenvalues;
    return JBLAS_EXT_UNAVAILABLE;
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
