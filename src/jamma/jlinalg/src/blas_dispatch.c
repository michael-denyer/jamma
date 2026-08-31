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
#include "blas_dispatch_internal.h"

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

const blas_candidate_t *blas_dispatch_active(void) {
    return &g_active;
}

int blas_dispatch_has_vendor_dgemm(void) {
    return g_has_vendor_dgemm;
}
