/**
 * blas_dispatch.c -- BLAS/LAPACK discovery and dispatch wrapper.
 *
 * Dispatch priority (consistency with GEMMA over raw speed):
 *   1. ILP64 with LAPACK (dsyevd): MKL-ILP64, Accelerate-ILP64
 *   2. numpy fallback (no vendor BLAS found)
 *
 * LP64 BLAS is detected but never wired -- its different FP accumulation
 * order would diverge from GEMMA -- so an LP64-only host uses the numpy
 * fallback.
 *
 * Discovery model: discover-all-then-select-best.  Both discovery paths
 * (system BLAS, pip-installed MKL) run unconditionally.  The best candidate
 * is selected based on capabilities (ILP64 + LAPACK > numpy-fallback).
 *
 * When an external dgemm is found, the vendor function pointers are wired.
 * CBLAS backends handle row-major natively; Fortran backends use the A/B
 * swap trick for column-major conversion.
 *
 * The dlopen machinery is POSIX-only. `run_build()` refuses to compile the
 * C extensions on Windows at all (see `compile_and_link.py`), so this file
 * carries no Windows stub path.
 */

/* _GNU_SOURCE required on glibc for RTLD_DEFAULT in <dlfcn.h>. Must be
 * defined before any system headers so feature-test macro selection is
 * consistent across the translation unit. macOS's <dlfcn.h> exposes
 * RTLD_DEFAULT unconditionally; the standard manylinux baseline image
 * happens to enable it via its default CFLAGS, but the AVX2 manylinux
 * image (gcc-toolset-14) does not — the define here makes the build
 * portable regardless of base image. (The BLIS strip removed this
 * define along with the dladdr usage that originally motivated it; the
 * RTLD_DEFAULT usage remained and silently relied on base-image
 * defaults.)
 */
#define _GNU_SOURCE

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "jlinalg.h"

#include <dlfcn.h>
#include <dirent.h>

/* ---------------------------------------------------------------------------
 * Debug flag
 * ---------------------------------------------------------------------------
 */
static int _debug_enabled(void) {
    const char *val = getenv("JLINALG_DISPATCH_DEBUG");
    return val && val[0] == '1';
}

/* JLINALG_NO_VENDOR_DGEMM — leave vendor dgemm unwired even when an ILP64
 * backend resolves, so blas_has_external() reports 0 with the extension
 * loaded and the rest of dispatch intact.  That is the state an LP64-only
 * host is permanently in (distro or conda numpy), and CI never reaches it
 * because PyPI numpy ships ILP64 scipy_openblas64.  Truthy values follow
 * jamma.core.constants.env_flag: anything except unset, "" and "0". */
static int _no_vendor_dgemm(void) {
    const char *val = getenv("JLINALG_NO_VENDOR_DGEMM");
    return val && val[0] != '\0' && !(val[0] == '0' && val[1] == '\0');
}

/* ---------------------------------------------------------------------------
 * Backend name detection from library path
 * ---------------------------------------------------------------------------
 */
static const char *_detect_backend_name(const char *lib_path, int is_ilp64) {
    if (lib_path) {
        if (strstr(lib_path, "mkl")) return is_ilp64 ? "MKL-ILP64" : "MKL-LP64";
        if (strstr(lib_path, "openblas")) return is_ilp64 ? "OpenBLAS-ILP64" : "OpenBLAS-LP64";
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
    int found;
    int is_ilp64;
    int has_lapack; /* has LAPACK dsyevd (only routine currently resolved) */
    int has_dsyrk;
    const char *name;
    void *handle;
    /* dgemm */
    jlinalg_dgemm_ilp64_fn dgemm_ilp64;
    jlinalg_cblas_dgemm_ilp64_fn cblas_dgemm_ilp64;
    /* dsyrk */
    jlinalg_cblas_dsyrk_ilp64_fn cblas_dsyrk_ilp64;
    jlinalg_dsyrk_ilp64_fn dsyrk_ilp64;
    /* dsyevd (Fortran) */
    jlinalg_dsyevd_ilp64_fn dsyevd_ilp64;
    /* LAPACKE dsyevd (C interface, row-major — no transpose needed) */
    jlinalg_lapacke_dsyevd_ilp64_fn lapacke_dsyevd_ilp64;
    int has_lapacke_dsyevd;
    /* dsyevr (Fortran) — memory-pressure fallback for dsyevd */
    jlinalg_dsyevr_ilp64_fn dsyevr_ilp64;
    int has_dsyevr;
} blas_candidate_t;

/* ---------------------------------------------------------------------------
 * Module-level state
 *
 * g_active is the winning candidate, copied in whole by blas_dispatch_init().
 * Every has_* accessor below derives its answer from whichever pointer field
 * that candidate carries, instead of a second bank of hand-set booleans that
 * could drift from the pointers they describe.
 * ---------------------------------------------------------------------------
 */
static blas_candidate_t g_active = {0};
static const char *g_backend_name = "numpy-fallback";
static int g_has_vendor_dgemm =
    0; /* dgemm actually wired; JLINALG_NO_VENDOR_DGEMM can suppress it */

/* ---------------------------------------------------------------------------
 * Symbol resolution -- one resolver, driven by a name table
 *
 * Every routine below looks up a short ordered list of candidate names in
 * one function-pointer field of blas_candidate_t. resolve_first_symbol() is
 * that lookup, done once, and the only place in this file that calls the
 * POSIX symbol-lookup primitive. SYMS[] describes, per routine, which field
 * the resolved pointer lands in (by offset), so one loop drives all of them.
 * ---------------------------------------------------------------------------
 */

/* Try each name in order; return the first resolved symbol, or NULL. */
static void *resolve_first_symbol(void *handle, const char *const *names, const char **found_name) {
    for (const char *const *name = names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (found_name) *found_name = *name;
            return sym;
        }
    }
    return NULL;
}

typedef struct {
    const char *label;        /* for debug logging */
    const char *const *names; /* candidate symbol names, in try order, NULL-terminated */
    size_t field_offset;      /* offsetof(blas_candidate_t, <pointer field>) */
    size_t flag_offset;       /* offsetof(blas_candidate_t, <has_* flag>), or (size_t)-1 for none */
} blas_sym_entry_t;

/* dgemm: MKL/OpenBLAS ILP64 Fortran names, tried on every platform. Apple's
 * CBLAS + Fortran-fallback pair is resolved separately below, since it also
 * sets is_ilp64/name/found/handle rather than only a pointer field. */
static const char *const ilp64_dgemm_names[] = {"dgemm_64_",       /* MKL ILP64 */
                                                "scipy_dgemm_64_", /* scipy-openblas64 */
                                                "dgemm64_",        /* OpenBLAS INTERFACE64=1 */
                                                NULL};
/* Apple Accelerate ILP64 (macOS 13.3+): uses $NEWLAPACK$ILP64 suffix.
 * Fortran interface has no trailing underscore. */
static const char *const accel_ilp64_dgemm_names[] = {"dgemm$NEWLAPACK$ILP64", NULL};
static const char *const accel_ilp64_cblas_names[] = {"cblas_dgemm$NEWLAPACK$ILP64", NULL};

/**
 * try_resolve_dgemm_candidate -- Try to resolve dgemm from a dlopen handle.
 * Populates the candidate struct instead of globals.
 * Returns 1 if found, 0 if not.
 *
 * lib_path: hint for backend name detection (may be NULL for RTLD_DEFAULT).
 */
static int try_resolve_dgemm_candidate(void *handle, const char *lib_path, blas_candidate_t *c) {
    int dbg = _debug_enabled();
    const char *matched = NULL;

    /* Try ILP64 symbols first (MKL, OpenBLAS) */
    void *sym = resolve_first_symbol(handle, ilp64_dgemm_names, &matched);
    if (sym) {
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved %s\n", matched);
        c->dgemm_ilp64 = (jlinalg_dgemm_ilp64_fn)sym;
        c->is_ilp64 = 1;
        c->name = _detect_backend_name(lib_path, 1);
        c->found = 1;
        c->handle = handle;
        return 1;
    }

    /* Try Apple Accelerate ILP64 (macOS 13.3+) — prefer CBLAS for row-major */
    void *cblas_sym = resolve_first_symbol(handle, accel_ilp64_cblas_names, &matched);
    if (cblas_sym) {
        if (dbg)
            fprintf(stderr, "jlinalg_dispatch:   resolved %s (Accelerate ILP64 CBLAS)\n", matched);
        c->cblas_dgemm_ilp64 = (jlinalg_cblas_dgemm_ilp64_fn)cblas_sym;
        c->is_ilp64 = 1;
        c->name = "Accelerate-ILP64";
        c->found = 1;
        c->handle = handle;
        /* Also try Fortran interface as fallback */
        const char *fmatched = NULL;
        void *fsym = resolve_first_symbol(handle, accel_ilp64_dgemm_names, &fmatched);
        if (fsym) {
            c->dgemm_ilp64 = (jlinalg_dgemm_ilp64_fn)fsym;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   also resolved %s\n", fmatched);
        }
        return 1;
    }

    /* Detect an LP64-only backend so callers can log it, but do not wire it:
     * its FP accumulation order diverges from GEMMA. */
    static const char *const lp64_dgemm_names[] = {"dgemm_", NULL};
    if (resolve_first_symbol(handle, lp64_dgemm_names, NULL)) {
        c->is_ilp64 = 0;
        c->name = _detect_backend_name(lib_path, 0);
        c->found = 1;
        c->handle = handle;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgemm_ (LP64, not wired)\n");
        return 1;
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dsyrk, dsyevd, dsyevr (ILP64-only, one table, one loop)
 *
 * Each of these three requires c->is_ilp64 already set by dgemm resolution.
 * Every entry's candidate-name list carries the Apple $NEWLAPACK$ILP64 name
 * first, then the MKL/OpenBLAS names; a lookup against a non-Accelerate
 * handle simply never matches the Apple name, so one list works on every
 * platform and no #ifdef __APPLE__ survives in these three routines. Each
 * entry names the primary pointer field to set (by offset) and the has_*
 * flag to set alongside it (by offset); resolve_syms_table() is the one loop
 * that walks SYMS[] and does both assignments through a byte pointer.
 *
 * A resolved symbol always fills a `void *`-sized function-pointer slot
 * regardless of the pointer typedef in blas_candidate_t, so writing through
 * `void **` at the recorded offset is exactly what each routine's own
 * `c->field = (typedef)sym;` used to do.
 * ---------------------------------------------------------------------------
 */
static const char *const dsyrk_names[] = {"cblas_dsyrk$NEWLAPACK$ILP64", /* Accelerate ILP64 */
                                          "dsyrk_64_",                   /* MKL ILP64 */
                                          "dsyrk64_",                    /* OpenBLAS ILP64 */
                                          NULL};
static const char *const dsyevd_names[] = {"dsyevd$NEWLAPACK$ILP64", /* Accelerate ILP64 */
                                           "dsyevd_64_",             /* MKL ILP64 */
                                           "dsyevd64_",              /* OpenBLAS ILP64 */
                                           NULL};
static const char *const dsyevr_names[] = {"dsyevr$NEWLAPACK$ILP64", /* Accelerate ILP64 */
                                           "dsyevr_64_",             /* MKL ILP64 */
                                           "dsyevr64_",              /* OpenBLAS ILP64 */
                                           NULL};
/* Apple also exposes a Fortran dsyrk alongside the CBLAS one; resolved as a
 * secondary pointer on the same candidate when the primary (CBLAS) name hits. */
static const char *const dsyrk_fortran_fallback_names[] = {"dsyrk$NEWLAPACK$ILP64", NULL};
static const char *const lapacke_dsyevd_names[] = {"LAPACKE_dsyevd", NULL};

static const blas_sym_entry_t SYMS[] = {
    {"dsyrk", dsyrk_names, offsetof(blas_candidate_t, cblas_dsyrk_ilp64),
     offsetof(blas_candidate_t, has_dsyrk)},
    {"dsyevd", dsyevd_names, offsetof(blas_candidate_t, dsyevd_ilp64),
     offsetof(blas_candidate_t, has_lapack)},
    {"dsyevr", dsyevr_names, offsetof(blas_candidate_t, dsyevr_ilp64),
     offsetof(blas_candidate_t, has_dsyevr)},
};
#define N_SYMS (sizeof(SYMS) / sizeof(SYMS[0]))

/* Resolve every table entry against one handle, writing the primary pointer
 * and has_* flag fields in blas_candidate_t at their recorded offsets. */
static void resolve_syms_table(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();
    for (size_t i = 0; i < N_SYMS; i++) {
        const blas_sym_entry_t *entry = &SYMS[i];
        const char *matched = NULL;
        void *sym = resolve_first_symbol(handle, entry->names, &matched);
        if (!sym) continue;
        *(void **)((char *)c + entry->field_offset) = sym;
        *(int *)((char *)c + entry->flag_offset) = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved %s (%s)\n", matched, entry->label);
    }

    /* Two secondary pointers hang off a table hit but land in a field
     * distinct from the routine's own has_* flag, so they stay outside the
     * generic loop: Accelerate's Fortran dsyrk (alongside its CBLAS entry),
     * and LAPACKE_dsyevd (a fallback C interface with its own has_* flag,
     * MKL/OpenBLAS only -- Accelerate carries no LAPACKE). */
    if (c->has_dsyrk && c->cblas_dsyrk_ilp64 && !c->dsyrk_ilp64) {
        const char *fmatched = NULL;
        void *fsym = resolve_first_symbol(handle, dsyrk_fortran_fallback_names, &fmatched);
        if (fsym) {
            c->dsyrk_ilp64 = (jlinalg_dsyrk_ilp64_fn)fsym;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   also resolved %s\n", fmatched);
        }
    }
    if (!c->has_lapacke_dsyevd) {
        const char *matched = NULL;
        void *le64 = resolve_first_symbol(handle, lapacke_dsyevd_names, &matched);
        if (le64) {
            /* When loaded from an ILP64 library, LAPACKE_dsyevd uses
             * lapack_int = long long.  Cast to our ILP64 typedef. */
            c->lapacke_dsyevd_ilp64 = (jlinalg_lapacke_dsyevd_ilp64_fn)le64;
            c->has_lapacke_dsyevd = 1;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved %s (ILP64)\n", matched);
        }
    }
}

/* Resolve the Level-3/LAPACK ops that hang off a dgemm candidate: dsyrk,
 * dsyevd, and its memory-pressure fallback dsyevr.  Every dgemm-resolution
 * site runs the same table against the same handle, so they share one entry
 * point.  A no-op on an LP64 candidate, which never wires these ops. */
static void resolve_secondary_ops(void *handle, blas_candidate_t *c) {
    if (!c->is_ilp64) return;
    resolve_syms_table(handle, c);
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
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   scan_dir %s -- opendir failed\n", dirpath);
        return 0;
    }
    if (dbg) fprintf(stderr, "jlinalg_dispatch:   scan_dir %s -- opened\n", dirpath);

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Look for openblas or mkl shared libraries */
        if (!strstr(entry->d_name, "openblas") && !strstr(entry->d_name, "libmkl")) continue;
        /* Must be a .so or .dylib */
        if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib")) continue;

        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);

        if (dbg) fprintf(stderr, "jlinalg_dispatch:   trying dlopen: %s\n", fullpath);
        void *handle = dlopen(fullpath, RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   dlopen failed: %s\n", dlerror());
            continue;
        }

        if (try_resolve_dgemm_candidate(handle, fullpath, c)) {
            if (dbg)
                fprintf(stderr, "jlinalg_dispatch:   resolved dgemm from %s (ilp64=%d)\n", fullpath,
                        c->is_ilp64);
            resolve_secondary_ops(handle, c);
            closedir(dir);
            return 1;
        }
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   dgemm not found in %s\n", entry->d_name);
        dlclose(handle);
    }
    closedir(dir);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Force numpy BLAS load
 * ---------------------------------------------------------------------------
 */
static void force_numpy_blas_load(void) {
    int dbg = _debug_enabled();
    PyObject *np = PyImport_ImportModule("numpy");
    if (!np) {
        if (dbg) fprintf(stderr, "jlinalg_dispatch: force_numpy_blas_load: numpy import failed\n");
        PyErr_Clear();
        return;
    }

    PyObject *linalg = PyObject_GetAttrString(np, "linalg");
    if (!linalg) {
        if (dbg)
            fprintf(stderr, "jlinalg_dispatch: force_numpy_blas_load: numpy.linalg not found\n");
        PyErr_Clear();
        Py_DECREF(np);
        return;
    }

    PyObject *eigh = PyObject_GetAttrString(linalg, "eigh");
    PyObject *eye = PyObject_GetAttrString(np, "eye");
    if (!eigh || !eye) {
        PyErr_Clear();
        Py_XDECREF(eigh);
        Py_XDECREF(eye);
        Py_DECREF(linalg);
        Py_DECREF(np);
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
            if (dbg)
                fprintf(stderr, "jlinalg_dispatch: force_numpy_blas_load: eigh(eye(2)) failed\n");
            PyErr_Clear();
        }
        Py_DECREF(eye_result);
    } else {
        if (dbg) fprintf(stderr, "jlinalg_dispatch: force_numpy_blas_load: eye(2) failed\n");
        PyErr_Clear();
    }

    Py_DECREF(eigh);
    Py_DECREF(eye);
    Py_DECREF(linalg);
    Py_DECREF(np);
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
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   /proc/self/maps -- fopen failed\n");
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

        if (!strstr(basename, "openblas") && !strstr(basename, "libmkl")) continue;
        if (!strstr(basename, ".so")) continue;

        if (dbg) fprintf(stderr, "jlinalg_dispatch:   /proc/self/maps candidate: %s\n", path);

        void *handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) {
            if (dbg)
                fprintf(stderr, "jlinalg_dispatch:   RTLD_NOLOAD failed, trying full load: %s\n",
                        dlerror());
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        }
        if (!handle) {
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   dlopen failed: %s\n", dlerror());
            continue;
        }

        if (try_resolve_dgemm_candidate(handle, path, c)) {
            if (dbg)
                fprintf(stderr,
                        "jlinalg_dispatch:   resolved dgemm from /proc/self/maps (ilp64=%d)\n",
                        c->is_ilp64);
            resolve_secondary_ops(handle, c);
            fclose(fp);
            return 1;
        }
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   dgemm not found in %s\n", basename);
        dlclose(handle);
    }
    fclose(fp);
#else
    (void)c;
#endif
    return 0;
}

/* ---------------------------------------------------------------------------
 * Directory probing -- delegates to jamma.jlinalg._blas_dirs.probe_plan()
 *
 * Finding candidate directories is pathlib/importlib work with no need for
 * dlopen, so it lives in Python. This C side keeps every dlopen call and
 * symbol lookup; it only asks Python where to look. `_run_probe_plan` calls
 * the plan once and hands each `(kind, path)` pair to `visit` in order,
 * stopping early when `visit` resolves dgemm (mirrors the early-return shape
 * the callers already had).
 * ---------------------------------------------------------------------------
 */
typedef int (*blas_dir_visitor_fn)(const char *kind, const char *dirpath, blas_candidate_t *c);

static int _run_probe_plan(blas_dir_visitor_fn visit, blas_candidate_t *c) {
    int dbg = _debug_enabled();
    int found = 0;

    PyObject *dirs_mod = PyImport_ImportModule("jamma.jlinalg._blas_dirs");
    if (!dirs_mod) {
        if (dbg) fprintf(stderr, "jlinalg_dispatch: _blas_dirs import failed\n");
        PyErr_Clear();
        return 0;
    }

    PyObject *plan = PyObject_CallMethod(dirs_mod, "probe_plan", NULL);
    Py_DECREF(dirs_mod);
    if (!plan) {
        if (dbg) fprintf(stderr, "jlinalg_dispatch: probe_plan() failed\n");
        PyErr_Clear();
        return 0;
    }

    Py_ssize_t n = PySequence_Length(plan);
    for (Py_ssize_t i = 0; i < n && !found; i++) {
        PyObject *entry = PySequence_GetItem(plan, i);
        if (!entry) {
            PyErr_Clear();
            continue;
        }
        PyObject *kind_obj = PySequence_GetItem(entry, 0);
        PyObject *path_obj = PySequence_GetItem(entry, 1);
        if (kind_obj && path_obj) {
            const char *kind = PyUnicode_AsUTF8(kind_obj);
            const char *dirpath = PyUnicode_AsUTF8(path_obj);
            if (kind && dirpath) {
                if (dbg)
                    fprintf(stderr, "jlinalg_dispatch: probe_plan entry kind=%s path=%s\n", kind,
                            dirpath);
                found = visit(kind, dirpath, c);
            } else {
                PyErr_Clear();
            }
        } else {
            PyErr_Clear();
        }
        Py_XDECREF(kind_obj);
        Py_XDECREF(path_obj);
        Py_DECREF(entry);
    }

    Py_DECREF(plan);
    return found;
}

/* ---------------------------------------------------------------------------
 * discover_system_blas -- Full system BLAS discovery (4-step pattern)
 * Populates a blas_candidate_t instead of setting globals.
 * ---------------------------------------------------------------------------
 */
static int _visit_system_blas_dir(const char *kind, const char *dirpath, blas_candidate_t *c) {
    if (strcmp(kind, "openblas_or_mkl") != 0) return 0;
    return scan_dir_for_blas_candidate(dirpath, c);
}

static void discover_system_blas(blas_candidate_t *c) {
    int dbg = _debug_enabled();

    /* Step 1: RTLD_DEFAULT (catches macOS Accelerate, LD_PRELOAD) */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: step 1 -- RTLD_DEFAULT\n");
    if (try_resolve_dgemm_candidate(RTLD_DEFAULT, NULL, c)) {
        if (dbg)
            fprintf(stderr, "jlinalg_dispatch: found via RTLD_DEFAULT (ilp64=%d, backend=%s)\n",
                    c->is_ilp64, c->name);
        resolve_secondary_ops(RTLD_DEFAULT, c);
        return;
    }

    /* Step 2: Force numpy to load its BLAS, then retry RTLD_DEFAULT */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: step 2 -- force numpy BLAS load\n");
    force_numpy_blas_load();
    if (try_resolve_dgemm_candidate(RTLD_DEFAULT, NULL, c)) {
        if (dbg)
            fprintf(stderr,
                    "jlinalg_dispatch: found via RTLD_DEFAULT after numpy load (ilp64=%d, "
                    "backend=%s)\n",
                    c->is_ilp64, c->name);
        resolve_secondary_ops(RTLD_DEFAULT, c);
        return;
    }

    /* Step 3: /proc/self/maps scan (Linux only) */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: step 3 -- /proc/self/maps scan\n");
    if (scan_proc_maps_for_blas_candidate(c)) {
        if (dbg)
            fprintf(stderr, "jlinalg_dispatch: found via /proc/self/maps (ilp64=%d, backend=%s)\n",
                    c->is_ilp64, c->name);
        return;
    }

    /* Step 4: Scan numpy's lib directories (candidate dirs come from Python) */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: step 4 -- numpy dir scan\n");
    _run_probe_plan(_visit_system_blas_dir, c);
}

/* ---------------------------------------------------------------------------
 * discover_pip_mkl -- Look for pip-installed MKL (site-packages/mkl)
 * ---------------------------------------------------------------------------
 */
static int _visit_pip_mkl_dir(const char *kind, const char *dirpath, blas_candidate_t *c) {
    int dbg = _debug_enabled();
    if (strcmp(kind, "mkl") != 0) return 0;

    if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- trying dir: %s\n", dirpath);

    /* MKL libraries must be loaded in dependency order:
     * core first, then sequential, then ilp64 */
    const char *mkl_libs[] = {"libmkl_core", "libmkl_sequential", "libmkl_intel_ilp64", NULL};
    void *last_handle = NULL;

    for (int li = 0; mkl_libs[li]; li++) {
        /* Scan directory for matching .so/.dylib */
        DIR *dir = opendir(dirpath);
        if (!dir) break;

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (!strstr(entry->d_name, mkl_libs[li])) continue;
            if (!strstr(entry->d_name, ".so") && !strstr(entry->d_name, ".dylib")) continue;

            char fullpath[4096];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
            if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- dlopen %s\n", fullpath);

            void *h = dlopen(fullpath, RTLD_LAZY | RTLD_GLOBAL);
            if (h) {
                last_handle = h;
                if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- loaded %s\n", entry->d_name);
            } else {
                if (dbg)
                    fprintf(stderr, "jlinalg_dispatch: pip-mkl -- dlopen failed: %s\n", dlerror());
            }
            break;
        }
        closedir(dir);
    }

    if (!last_handle) return 0;

    /* Try to resolve symbols from RTLD_DEFAULT (all loaded globally) */
    if (!try_resolve_dgemm_candidate(RTLD_DEFAULT, dirpath, c)) return 0;

    if (!c->is_ilp64) {
        /* Loaded ILP64 MKL libs but only resolved LP64 symbols.
         * Don't label as ILP64 — would cause ABI mismatch. */
        if (dbg)
            fprintf(stderr, "jlinalg_dispatch: pip-mkl -- "
                            "WARNING: resolved LP64 dgemm from ILP64 MKL path, skipping\n");
        c->found = 0;
        return 0;
    }

    c->name = "MKL-ILP64";
    resolve_secondary_ops(RTLD_DEFAULT, c);
    if (dbg)
        fprintf(stderr, "jlinalg_dispatch: pip-mkl -- resolved (ilp64=%d, lapack=%d)\n",
                c->is_ilp64, c->has_lapack);
    return 1;
}

static void discover_pip_mkl(blas_candidate_t *c) {
    int dbg = _debug_enabled();
    if (!_run_probe_plan(_visit_pip_mkl_dir, c)) {
        if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- not found\n");
    }
}

/* ---------------------------------------------------------------------------
 * Candidate scoring and selection
 *
 * A has_* flag and its pointer are always set together by the same table
 * entry in resolve_syms_table() (or, for dgemm, by the same branch in
 * try_resolve_dgemm_candidate()), so a flag can no longer be true with its
 * pointer NULL -- that invariant used to need a separate validation pass
 * that zeroed out an inconsistent candidate; construction now makes the
 * inconsistency it guarded against unrepresentable.
 *
 * _score_candidate:    ILP64 + LAPACK = 4, ILP64 BLAS-only = 3, LP64 = 1.
 * select_best_backend: returns highest-scoring candidate (NULL if none).
 *   LP64 candidates are returned for logging but not wired for dgemm.
 * ---------------------------------------------------------------------------
 */

static int _score_candidate(const blas_candidate_t *c) {
    if (!c->found) return 0;
    if (c->is_ilp64 && c->has_lapack) return 4;
    if (c->is_ilp64) return 3;
    return 1; /* LP64 */
}

static blas_candidate_t *select_best_backend(blas_candidate_t *system, blas_candidate_t *pip_mkl) {
    int s_sys = _score_candidate(system);
    int s_pip = _score_candidate(pip_mkl);
    int dbg = _debug_enabled();

    if (dbg) fprintf(stderr, "jlinalg_dispatch: scores: system=%d pip_mkl=%d\n", s_sys, s_pip);

    blas_candidate_t *best = NULL;
    int best_score = 0;

    if (s_sys > best_score) {
        best = system;
        best_score = s_sys;
    }
    if (s_pip > best_score) {
        best = pip_mkl;
        best_score = s_pip;
    }

    return best;
}

/* ---------------------------------------------------------------------------
 * Public API — dispatch init (discover-all-then-select-best)
 * ---------------------------------------------------------------------------
 */

int blas_dispatch_init(void) {
    int dbg = _debug_enabled();

    blas_candidate_t system = {0};
    blas_candidate_t pip_mkl = {0};

    /* Both discovery paths run unconditionally */
    discover_system_blas(&system);
    discover_pip_mkl(&pip_mkl);

    blas_candidate_t *best = select_best_backend(&system, &pip_mkl);

    if (best && best->is_ilp64) {
        g_active = *best;
        g_backend_name = g_active.name;

        /* dgemm is wired unless JLINALG_NO_VENDOR_DGEMM asks to leave it
         * unwired for testing the numpy-fallback path on an ILP64 host. */
        if (_no_vendor_dgemm()) {
            fprintf(stderr, "jlinalg_dispatch: INFO: JLINALG_NO_VENDOR_DGEMM set -- "
                            "vendor dgemm left unwired, numpy fallback in use.\n");
            g_active.dgemm_ilp64 = NULL;
            g_active.cblas_dgemm_ilp64 = NULL;
        } else {
            if (dbg)
                fprintf(stderr, "jlinalg_dispatch: using %s (ILP64) for dgemm\n", g_active.name);
            g_has_vendor_dgemm = 1;
        }

        if (dbg) {
            if (g_active.has_dsyrk) fprintf(stderr, "jlinalg_dispatch: vendor dsyrk wired\n");
            if (g_active.has_lapack) {
                fprintf(stderr, "jlinalg_dispatch: vendor dsyevd wired (%s)\n",
                        g_active.has_lapacke_dsyevd ? "LAPACKE, row-major"
                                                    : "Fortran, transpose required");
            }
            if (g_active.has_dsyevr)
                fprintf(stderr,
                        "jlinalg_dispatch: vendor dsyevr wired (memory-pressure fallback)\n");
        }
        return 0;
    }

    if (best && best->found && !best->is_ilp64) {
        /* LP64 found but not ILP64 -- prefer numpy fallback for consistency */
        if (dbg)
            fprintf(stderr,
                    "jlinalg_dispatch: LP64 %s available but preferring numpy fallback for "
                    "consistency\n",
                    best->name);
        fprintf(stderr,
                "jlinalg_dispatch: INFO: LP64 BLAS (%s) detected but not used -- "
                "numpy fallback preferred for numerical consistency with GEMMA. "
                "Install ILP64 numpy for faster external BLAS dispatch.\n",
                best->name);
        return 0;
    }

    /* No external dgemm found -- numpy fallback */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: no external dgemm found, using numpy-fallback\n");
    return 0;
}

const char *blas_backend_name(void) {
    return g_backend_name;
}

int blas_is_ilp64(void) {
    return g_active.is_ilp64;
}

int blas_has_external(void) {
    /* Only true when external BLAS is actually wired (i.e., ILP64 found and
     * JLINALG_NO_VENDOR_DGEMM did not veto it). LP64-only discovery never
     * wires dispatch. */
    return g_has_vendor_dgemm;
}

int blas_has_dsyrk(void) {
    return g_active.cblas_dsyrk_ilp64 != NULL || g_active.dsyrk_ilp64 != NULL;
}
int blas_has_dsyevd(void) {
    return g_active.dsyevd_ilp64 != NULL || g_active.lapacke_dsyevd_ilp64 != NULL;
}
int blas_has_lapacke_dsyevd(void) {
    return g_active.lapacke_dsyevd_ilp64 != NULL;
}
int blas_has_dsyevr(void) {
    return g_active.dsyevr_ilp64 != NULL && g_active.is_ilp64;
}

/* ---------------------------------------------------------------------------
 * jlinalg_dsyrk_ext — Vendor-dispatch dsyrk: C = X @ X.T + beta*C
 * ---------------------------------------------------------------------------
 */
void jlinalg_dsyrk_ext(npy_intp N, npy_intp K, const double *X, npy_intp ldx, double *C,
                       npy_intp ldc, double beta) {
    if (N <= 0) return;
    if (K <= 0) {
        for (npy_intp i = 0; i < N; i++) {
            if (beta == 0.0)
                memset(C + i * ldc, 0, (size_t)N * sizeof(*C));
            else
                for (npy_intp j = 0; j <= i; j++)
                    C[i * ldc + j] *= beta;
        }
        for (npy_intp i = 0; i < N; i++)
            for (npy_intp j = i + 1; j < N; j++)
                C[i * ldc + j] = C[j * ldc + i];
        return;
    }
    if (g_active.is_ilp64) {
        if (g_active.cblas_dsyrk_ilp64) {
            /* Row-major, lower, no-trans: C = X @ X.T + beta * C */
            g_active.cblas_dsyrk_ilp64(JLINALG_CblasRowMajor, JLINALG_CblasLower,
                                       JLINALG_CblasNoTrans, (long)N, (long)K, 1.0, X, (long)ldx,
                                       beta, C, (long)ldc);
            /* Mirror lower to upper (vendor only fills lower) */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++)
                    C[i * ldc + j] = C[j * ldc + i];
            return;
        }
        /* Fortran ILP64 fallback: row-major lower = col-major upper */
        if (g_active.dsyrk_ilp64) {
            const long long n = (long long)N, k = (long long)K;
            const long long lda = (long long)ldx, ldc_f = (long long)ldc;
            const double alpha = 1.0;
            g_active.dsyrk_ilp64("U", "T", &n, &k, &alpha, X, &lda, &beta, C, &ldc_f);
            /* Fortran col-major upper = row-major lower; mirror lower to upper */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++)
                    C[i * ldc + j] = C[j * ldc + i];
            return;
        }
    }
    /* No vendor dsyrk available -- caller should use numpy fallback. */
    fprintf(stderr, "FATAL: jlinalg_dsyrk_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

/* ---------------------------------------------------------------------------
 * jlinalg_dsyevd_ext — Vendor-dispatch dsyevd for eigh
 *
 * Prefers LAPACKE C interface (row-major, no transpose) when available (MKL).
 * Falls back to Fortran dsyevd + eigenvector transpose (Accelerate, OpenBLAS).
 *
 * Input: K is row-major symmetric, lower triangle populated.
 * Output: K overwritten with eigenvectors stored columnwise in row-major
 *         (K[i*ldk+j] = component i of eigenvector j).
 *         eigenvalues[k] = k-th eigenvalue, ascending.
 *
 * Returns: JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL,
 *          or positive int for LAPACK error (info capped to INT_MAX for ILP64).
 * ---------------------------------------------------------------------------
 */

/* Safely narrow LAPACK info (long long) to int return.  Logs the full value
 * when truncation would occur (ILP64 eigenvalue index > INT_MAX). */
static int _info_to_int(long long info, npy_intp N) {
    if (info > INT_MAX || info < INT_MIN) {
        fprintf(stderr,
                "jlinalg: LAPACK info=%lld exceeds int range (N=%ld) "
                "— returning capped value\n",
                info, (long)N);
        return info > 0 ? INT_MAX : INT_MIN;
    }
    return (int)info;
}

int jlinalg_dsyevd_ext(npy_intp N, double *K, npy_intp ldk, double *eigenvalues) {
    if (!blas_has_dsyevd() || !g_active.is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    /* --- LAPACKE path (MKL): row-major natively, no transpose needed.
     * Only used when Fortran ILP64 dsyevd is NOT available.  When both
     * exist, we prefer Fortran because dsyevd_64_ is an unambiguous ILP64
     * symbol, whereas LAPACKE_dsyevd is unsuffixed and could resolve to
     * the LP64 variant on systems with mixed LP64/ILP64 MKL. --- */
    if (g_active.lapacke_dsyevd_ilp64 && !g_active.dsyevd_ilp64) {
        long long info = g_active.lapacke_dsyevd_ilp64(
            JLINALG_LAPACK_ROW_MAJOR, 'V', 'L', (long long)N, K, (long long)ldk, eigenvalues);
        if (info != 0) return _info_to_int(info, N);
        return JLINALG_EXT_SUCCESS;
    }

    /* --- Fortran path (Accelerate, MKL, OpenBLAS): col-major + transpose.
     * Preferred when available because ILP64 symbol names (dsyevd_64_,
     * dsyevd$NEWLAPACK$ILP64) are unambiguous — no LP64/ILP64 confusion. --- */
    if (g_active.dsyevd_ilp64) {
        long long n = (long long)N;
        long long lda = (long long)ldk;
        long long info = 0;

        /* Workspace query */
        long long lwork = -1, liwork = -1;
        double work_query;
        long long iwork_query;
        g_active.dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues, &work_query, &lwork, &iwork_query,
                              &liwork, &info);
        if (info != 0) {
            fprintf(stderr,
                    "jlinalg_dsyevd_ext: Fortran dsyevd workspace query failed "
                    "(info=%lld, N=%lld) — likely ABI mismatch or corrupt LAPACK\n",
                    info, n);
            return (int)info;
        }

        lwork = (long long)work_query + 1; /* +1 for double→integer rounding */
        liwork = iwork_query;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        long long *iwork = (long long *)malloc((size_t)liwork * sizeof(long long));
        if (!work || !iwork) {
            /* CRITICAL: ALLOC_FAIL must be returned BEFORE K is modified.
             * eigh.c relies on K being unmodified when K == eigenvectors
             * so it can fall through to DSYEVR with the original data. */
            free(work);
            free(iwork);
            return JLINALG_EXT_ALLOC_FAIL;
        }

        /* Compute: UPLO='U' because row-major lower = col-major upper.
         * The matrix is symmetric so A = A^T — no input transpose needed,
         * just the UPLO swap. */
        g_active.dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues, work, &lwork, iwork, &liwork,
                              &info);
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
        return JLINALG_EXT_SUCCESS;
    }
    return JLINALG_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * jlinalg_dsyevr_ext — Vendor-dispatch dsyevr for eigh (memory-pressure fallback)
 *
 * DSYEVR uses O(N) workspace vs O(N^2) for DSYEVD.  Eigenvectors are written
 * into a separate Z output buffer (does not require an N x N copy of K).
 *
 * Input: K is row-major symmetric, lower triangle populated (overwritten).
 * Output: eigenvectors in row-major columnwise (Z[i*ldz+j] = component i of eigvec j).
 *         eigenvalues[k] = k-th eigenvalue, ascending.
 *
 * Returns: JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL,
 *          or positive int for LAPACK error (info capped to INT_MAX for ILP64).
 * ---------------------------------------------------------------------------
 */
int jlinalg_dsyevr_ext(npy_intp N, double *K, npy_intp ldk, double *eigenvalues,
                       double *eigenvectors, npy_intp ldz) {
    if (!blas_has_dsyevr()) return JLINALG_EXT_UNAVAILABLE;

    if (g_active.dsyevr_ilp64) {
        long long n = (long long)N;
        long long lda = (long long)ldk;
        long long ldz_f = (long long)N; /* tightly packed Z for Fortran */
        long long info = 0;
        long long m_out = 0;       /* number of eigenvalues found */
        double abstol = 0.0;       /* use default (DLAMCH) */
        long long il = 1, iu = n;  /* all eigenvalues (range='A' ignores these) */
        double vl = 0.0, vu = 0.0; /* unused for range='A' */

        /* Workspace query */
        long long lwork = -1, liwork = -1;
        double work_query;
        long long iwork_query;
        long long isuppz_dummy[2];
        g_active.dsyevr_ilp64("V", "A", "U", &n, K, &lda, &vl, &vu, &il, &iu, &abstol, &m_out,
                              eigenvalues, eigenvectors, &ldz_f, isuppz_dummy, &work_query, &lwork,
                              &iwork_query, &liwork, &info);
        if (info != 0) {
            fprintf(stderr, "jlinalg_dsyevr_ext: workspace query failed (info=%lld, N=%lld)\n",
                    info, n);
            return _info_to_int(info, N);
        }

        lwork = (long long)work_query + 1;
        liwork = iwork_query;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        long long *iwork = (long long *)malloc((size_t)liwork * sizeof(long long));
        long long *isuppz = (long long *)malloc((size_t)(2 * N) * sizeof(long long));
        /* Reuse the caller's tightly packed output buffer when possible. */
        int use_output_as_z = (ldz == N && eigenvectors != K);
        double *Z_col = use_output_as_z ? eigenvectors
                                        : (double *)malloc((size_t)N * (size_t)N * sizeof(double));
        if (!work || !iwork || !isuppz || !Z_col) {
            free(work);
            free(iwork);
            free(isuppz);
            if (!use_output_as_z) free(Z_col);
            return JLINALG_EXT_ALLOC_FAIL;
        }

        /* Compute: UPLO='U' because row-major lower = col-major upper. */
        g_active.dsyevr_ilp64("V", "A", "U", &n, K, &lda, &vl, &vu, &il, &iu, &abstol, &m_out,
                              eigenvalues, Z_col, &ldz_f, isuppz, work, &lwork, iwork, &liwork,
                              &info);
        free(work);
        free(iwork);
        free(isuppz);
        if (info != 0) {
            if (!use_output_as_z) free(Z_col);
            return _info_to_int(info, N);
        }

        /* Verify all eigenvalues were found (range='A' should always give m_out == N) */
        if (m_out != n) {
            fprintf(stderr,
                    "jlinalg_dsyevr_ext: expected %lld eigenvalues but DSYEVR found %lld "
                    "(range='A', N=%lld) — vendor LAPACK ABI mismatch or bug\n",
                    n, m_out, n);
            if (!use_output_as_z) free(Z_col);
            return JLINALG_EXT_COUNT_MISMATCH;
        }

        if (use_output_as_z) {
            /* DSYEVR wrote col-major data into a tight contiguous buffer.
             * Interpreted as row-major, that is the transpose of what Python
             * expects. Transpose in-place to restore row-major columnwise form. */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++) {
                    double tmp = eigenvectors[i * ldz + j];
                    eigenvectors[i * ldz + j] = eigenvectors[j * ldz + i];
                    eigenvectors[j * ldz + i] = tmp;
                }
        } else {
            /* Transpose col-major Z to row-major eigenvectors.
             * Z_col is col-major: Z_col[i + j*N] = component i of eigvec j.
             * eigenvectors is row-major: eigenvectors[i*ldz + j] = component i of eigvec j. */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = 0; j < N; j++)
                    eigenvectors[i * ldz + j] = Z_col[i + j * N];
            free(Z_col);
        }
        return JLINALG_EXT_SUCCESS;
    }

    return JLINALG_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * Full-signature external dgemm wrapper
 * ---------------------------------------------------------------------------
 */
static int _dgemm_external_full(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                                const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                                int transb, double alpha, double beta) {
    if (g_active.cblas_dgemm_ilp64) {
        int ta = transa ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        int tb = transb ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        long llda = (long)(lda > 0 ? lda : 1);
        long lldb = (long)(ldb > 0 ? ldb : 1);
        long lldc = (long)(ldc > 0 ? ldc : 1);
        g_active.cblas_dgemm_ilp64(JLINALG_CblasRowMajor, ta, tb, (long)M, (long)N, (long)K, alpha,
                                   A, llda, B, lldb, beta, C, lldc);
        return 1;
    }

    /* Fortran ILP64 interface fallback: row-major -> column-major swap.
     * LP64 dgemm is never wired, so the ILP64 pointer is the only path here. */
    const char *transa_f = transb ? "T" : "N";
    const char *transb_f = transa ? "T" : "N";

    const long long lM = (long long)M, lN = (long long)N, lK = (long long)K;
    const long long llda = (long long)lda, lldb = (long long)ldb;
    const long long lldc = (long long)ldc;
    g_active.dgemm_ilp64(transa_f, transb_f, &lN, &lM, &lK, &alpha, B, &lldb, A, &llda, &beta, C,
                         &lldc);
    return 1;
}

/* ---------------------------------------------------------------------------
 * Public full-signature dispatch API
 * ---------------------------------------------------------------------------
 */

void jlinalg_dgemm_ext(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                       const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                       int transb) {
    if (g_has_vendor_dgemm &&
        _dgemm_external_full(M, N, K, A, lda, B, ldb, C, ldc, transa, transb, 1.0, 0.0)) {
        return;
    }
    /* No external BLAS wired.
     * Caller should check blas_has_external() and use numpy fallback. */
    fprintf(stderr, "FATAL: jlinalg_dgemm_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}
