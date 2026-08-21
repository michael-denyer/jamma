/**
 * blas_dispatch.c -- BLAS/LAPACK discovery and dispatch wrapper.
 *
 * Dispatch priority (consistency with GEMMA over raw speed):
 *   1. ILP64 with LAPACK (dsyevd): MKL-ILP64, Accelerate-ILP64
 *   2. numpy fallback (no vendor BLAS found)
 *   3. LP64 (detected but not wired for dgemm -- different FP accumulation)
 *
 * Discovery model: discover-all-then-select-best.  Both discovery paths
 * (system BLAS, pip-installed MKL) run unconditionally.  The best candidate
 * is selected based on capabilities (ILP64 + LAPACK > numpy-fallback > LP64).
 *
 * When an external dgemm is found, the vendor function pointers are wired.
 * CBLAS backends handle row-major natively; Fortran backends use the A/B
 * swap trick for column-major conversion.
 *
 * The dlopen machinery is Unix-only (#if !defined(_WIN32)); on Windows
 * blas_dispatch_init() returns 0 immediately (no external dispatch).
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

#if !defined(_WIN32)

#include <dlfcn.h>
#include <dirent.h>

/* ---------------------------------------------------------------------------
 * Module-level state
 * ---------------------------------------------------------------------------
 */
static int g_is_ilp64 = 0;
static jlinalg_dgemm_lp64_fn g_dgemm_lp64 = NULL;
static jlinalg_dgemm_ilp64_fn g_dgemm_ilp64 = NULL;
static jlinalg_cblas_dgemm_fn g_cblas_dgemm = NULL;             /* LP64 CBLAS */
static jlinalg_cblas_dgemm_ilp64_fn g_cblas_dgemm_ilp64 = NULL; /* ILP64 CBLAS (Accelerate) */
static const char *g_backend_name = "numpy-fallback";
static void *g_blas_handle = NULL;

/* dsyrk dispatch pointers — ILP64 only (LP64 not wired, same policy as dgemm) */
static jlinalg_cblas_dsyrk_ilp64_fn g_cblas_dsyrk_ilp64 = NULL;
static jlinalg_dsyrk_ilp64_fn g_dsyrk_ilp64 = NULL;

/* dsyevd dispatch pointers (Fortran) */
static jlinalg_dsyevd_lp64_fn g_dsyevd_lp64 = NULL;
static jlinalg_dsyevd_ilp64_fn g_dsyevd_ilp64 = NULL;

/* LAPACKE dsyevd dispatch pointers (C interface, row-major) */
static jlinalg_lapacke_dsyevd_lp64_fn g_lapacke_dsyevd_lp64 = NULL;
static jlinalg_lapacke_dsyevd_ilp64_fn g_lapacke_dsyevd_ilp64 = NULL;

/* dsyevr dispatch pointers (Fortran) — memory-pressure fallback for dsyevd */
static jlinalg_dsyevr_lp64_fn g_dsyevr_lp64 = NULL;
static jlinalg_dsyevr_ilp64_fn g_dsyevr_ilp64 = NULL;

/* dgeqrf dispatch pointers (Fortran) */
static jlinalg_dgeqrf_lp64_fn g_dgeqrf_lp64 = NULL;
static jlinalg_dgeqrf_ilp64_fn g_dgeqrf_ilp64 = NULL;

/* dorgqr dispatch pointers (Fortran) */
static jlinalg_dorgqr_lp64_fn g_dorgqr_lp64 = NULL;
static jlinalg_dorgqr_ilp64_fn g_dorgqr_ilp64 = NULL;

/* dgesvd dispatch pointers (Fortran) */
static jlinalg_dgesvd_lp64_fn g_dgesvd_lp64 = NULL;
static jlinalg_dgesvd_ilp64_fn g_dgesvd_ilp64 = NULL;

/* Capability flags */
static int g_has_dsyrk = 0;
static int g_has_dsyevd = 0;
static int g_has_lapacke_dsyevd = 0;
static int g_has_dsyevr = 0;
static int g_has_dgeqrf = 0;
static int g_has_dgesvd = 0;

/* LP64 overflow guard: floor(sqrt(2^31 - 1)) */
#define LP64_DIM_MAX 46340

/* LP64 overflow counter: incremented when dimensions exceed LP64_DIM_MAX.
 * Resettable by py_eigh. */
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
    jlinalg_dgemm_lp64_fn dgemm_lp64;
    jlinalg_dgemm_ilp64_fn dgemm_ilp64;
    jlinalg_cblas_dgemm_fn cblas_dgemm;
    jlinalg_cblas_dgemm_ilp64_fn cblas_dgemm_ilp64;
    /* dsyrk */
    jlinalg_cblas_dsyrk_fn cblas_dsyrk;
    jlinalg_cblas_dsyrk_ilp64_fn cblas_dsyrk_ilp64;
    jlinalg_dsyrk_lp64_fn dsyrk_lp64;
    jlinalg_dsyrk_ilp64_fn dsyrk_ilp64;
    /* dsyevd (Fortran) */
    jlinalg_dsyevd_lp64_fn dsyevd_lp64;
    jlinalg_dsyevd_ilp64_fn dsyevd_ilp64;
    /* LAPACKE dsyevd (C interface, row-major — no transpose needed) */
    jlinalg_lapacke_dsyevd_lp64_fn lapacke_dsyevd_lp64;
    jlinalg_lapacke_dsyevd_ilp64_fn lapacke_dsyevd_ilp64;
    int has_lapacke_dsyevd;
    /* dsyevr (Fortran) — memory-pressure fallback for dsyevd */
    jlinalg_dsyevr_lp64_fn dsyevr_lp64;
    jlinalg_dsyevr_ilp64_fn dsyevr_ilp64;
    int has_dsyevr;
    /* dgeqrf (Fortran) */
    jlinalg_dgeqrf_lp64_fn dgeqrf_lp64;
    jlinalg_dgeqrf_ilp64_fn dgeqrf_ilp64;
    int has_dgeqrf;
    /* dorgqr (Fortran) */
    jlinalg_dorgqr_lp64_fn dorgqr_lp64;
    jlinalg_dorgqr_ilp64_fn dorgqr_ilp64;
    /* dgesvd (Fortran) */
    jlinalg_dgesvd_lp64_fn dgesvd_lp64;
    jlinalg_dgesvd_ilp64_fn dgesvd_ilp64;
    int has_dgesvd;
} blas_candidate_t;

/* ---------------------------------------------------------------------------
 * Symbol resolution — dgemm
 * ---------------------------------------------------------------------------
 */
static const char *ilp64_dgemm_names[] = {"dgemm_64_",       /* MKL ILP64 */
                                          "scipy_dgemm_64_", /* scipy-openblas64 */
                                          "dgemm64_",        /* OpenBLAS INTERFACE64=1 */
                                          NULL};
/* Apple Accelerate ILP64 (macOS 13.3+): uses $NEWLAPACK$ILP64 suffix.
 * Fortran interface has no trailing underscore. */
static const char *accel_ilp64_dgemm_names[] = {"dgemm$NEWLAPACK$ILP64", NULL};
static const char *accel_ilp64_cblas_names[] = {"cblas_dgemm$NEWLAPACK$ILP64", NULL};
static const char *lp64_dgemm_names[] = {"dgemm_", /* Standard Fortran / Accelerate */
                                         NULL};

/**
 * try_resolve_dgemm_candidate -- Try to resolve dgemm from a dlopen handle.
 * Populates the candidate struct instead of globals.
 * Returns 1 if found, 0 if not.
 *
 * lib_path: hint for backend name detection (may be NULL for RTLD_DEFAULT).
 */
static int try_resolve_dgemm_candidate(void *handle, const char *lib_path, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    /* Try ILP64 symbols first (MKL, OpenBLAS) */
    for (const char **name = ilp64_dgemm_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved %s\n", *name);
            c->dgemm_ilp64 = (jlinalg_dgemm_ilp64_fn)sym;
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
            if (dbg)
                fprintf(stderr, "jlinalg_dispatch:   resolved %s (Accelerate ILP64 CBLAS)\n",
                        *name);
            c->cblas_dgemm_ilp64 = (jlinalg_cblas_dgemm_ilp64_fn)sym;
            c->is_ilp64 = 1;
            c->name = "Accelerate-ILP64";
            c->found = 1;
            c->handle = handle;
            /* Also try Fortran interface as fallback */
            for (const char **fn = accel_ilp64_dgemm_names; *fn; fn++) {
                void *fsym = dlsym(handle, *fn);
                if (fsym) {
                    c->dgemm_ilp64 = (jlinalg_dgemm_ilp64_fn)fsym;
                    if (dbg) fprintf(stderr, "jlinalg_dispatch:   also resolved %s\n", *fn);
                }
            }
            return 1;
        }
    }

    /* Try LP64 symbols */
    for (const char **name = lp64_dgemm_names; *name; name++) {
        void *sym = dlsym(handle, *name);
        if (sym) {
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved %s\n", *name);
            c->dgemm_lp64 = (jlinalg_dgemm_lp64_fn)sym;
            c->is_ilp64 = 0;
            c->name = _detect_backend_name(lib_path, 0);
            c->found = 1;
            c->handle = handle;

            /* Also try cblas_dgemm — row-major native, no A/B swap needed. */
            void *cblas_sym = dlsym(handle, "cblas_dgemm");
            if (cblas_sym) {
                c->cblas_dgemm = (jlinalg_cblas_dgemm_fn)cblas_sym;
                if (dbg) fprintf(stderr, "jlinalg_dispatch:   also resolved cblas_dgemm\n");
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
static void try_resolve_dsyrk(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
        /* ILP64 dsyrk symbols */
#ifdef __APPLE__
        /* Accelerate ILP64 */
        void *sym = dlsym(handle, "cblas_dsyrk$NEWLAPACK$ILP64");
        if (sym) {
            c->cblas_dsyrk_ilp64 = (jlinalg_cblas_dsyrk_ilp64_fn)sym;
            c->has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved cblas_dsyrk$NEWLAPACK$ILP64\n");
            /* Also try Fortran */
            void *fsym = dlsym(handle, "dsyrk$NEWLAPACK$ILP64");
            if (fsym) {
                c->dsyrk_ilp64 = (jlinalg_dsyrk_ilp64_fn)fsym;
                if (dbg)
                    fprintf(stderr, "jlinalg_dispatch:   also resolved dsyrk$NEWLAPACK$ILP64\n");
            }
            return;
        }
#endif
        /* MKL ILP64 */
        void *sym64 = dlsym(handle, "dsyrk_64_");
        if (sym64) {
            c->dsyrk_ilp64 = (jlinalg_dsyrk_ilp64_fn)sym64;
            c->has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyrk_64_\n");
            return;
        }
        /* OpenBLAS ILP64 */
        void *sym64b = dlsym(handle, "dsyrk64_");
        if (sym64b) {
            c->dsyrk_ilp64 = (jlinalg_dsyrk_ilp64_fn)sym64b;
            c->has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyrk64_\n");
            return;
        }
    }

    /* LP64 dsyrk symbols */
    void *csym = dlsym(handle, "cblas_dsyrk");
    if (csym && !c->is_ilp64) {
        c->cblas_dsyrk = (jlinalg_cblas_dsyrk_fn)csym;
        c->has_dsyrk = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved cblas_dsyrk (LP64)\n");
        return;
    }
    void *fsym = dlsym(handle, "dsyrk_");
    if (fsym && !c->is_ilp64) {
        c->dsyrk_lp64 = (jlinalg_dsyrk_lp64_fn)fsym;
        c->has_dsyrk = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyrk_ (LP64)\n");
        return;
    }
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dsyevd
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dsyevd(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
#ifdef __APPLE__
        /* Accelerate ILP64: Fortran only (no LAPACKE in Accelerate) */
        void *sym = dlsym(handle, "dsyevd$NEWLAPACK$ILP64");
        if (sym) {
            c->dsyevd_ilp64 = (jlinalg_dsyevd_ilp64_fn)sym;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevd$NEWLAPACK$ILP64\n");
        }
        /* No LAPACKE on Accelerate — skip LAPACKE resolution */
        return;
#endif
        /* MKL/OpenBLAS ILP64: try LAPACKE first (C interface, row-major) */
        void *le64 = dlsym(handle, "LAPACKE_dsyevd");
        if (le64) {
            /* When loaded from an ILP64 library, LAPACKE_dsyevd uses
             * lapack_int = long long.  Cast to our ILP64 typedef. */
            c->lapacke_dsyevd_ilp64 = (jlinalg_lapacke_dsyevd_ilp64_fn)le64;
            c->has_lapacke_dsyevd = 1;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved LAPACKE_dsyevd (ILP64)\n");
        }

        /* Also resolve Fortran dsyevd as fallback */
        void *sym64 = dlsym(handle, "dsyevd_64_");
        if (sym64) {
            c->dsyevd_ilp64 = (jlinalg_dsyevd_ilp64_fn)sym64;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevd_64_\n");
            return;
        }
        /* OpenBLAS ILP64 */
        void *sym64b = dlsym(handle, "dsyevd64_");
        if (sym64b) {
            c->dsyevd_ilp64 = (jlinalg_dsyevd_ilp64_fn)sym64b;
            c->has_lapack = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevd64_\n");
            return;
        }
        return;
    }

    /* LP64: try LAPACKE first */
    void *le = dlsym(handle, "LAPACKE_dsyevd");
    if (le) {
        c->lapacke_dsyevd_lp64 = (jlinalg_lapacke_dsyevd_lp64_fn)le;
        c->has_lapacke_dsyevd = 1;
        c->has_lapack = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved LAPACKE_dsyevd (LP64)\n");
    }

    /* LP64 Fortran dsyevd */
    void *fsym = dlsym(handle, "dsyevd_");
    if (fsym) {
        c->dsyevd_lp64 = (jlinalg_dsyevd_lp64_fn)fsym;
        c->has_lapack = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevd_ (LP64)\n");
    }
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dsyevr (memory-pressure fallback for dsyevd)
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dsyevr(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
#ifdef __APPLE__
        /* Accelerate ILP64 */
        void *sym = dlsym(handle, "dsyevr$NEWLAPACK$ILP64");
        if (sym) {
            c->dsyevr_ilp64 = (jlinalg_dsyevr_ilp64_fn)sym;
            c->has_dsyevr = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevr$NEWLAPACK$ILP64\n");
        }
        return;
#endif
        /* MKL ILP64 */
        void *sym64 = dlsym(handle, "dsyevr_64_");
        if (sym64) {
            c->dsyevr_ilp64 = (jlinalg_dsyevr_ilp64_fn)sym64;
            c->has_dsyevr = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevr_64_\n");
            return;
        }
        /* OpenBLAS ILP64 */
        void *sym64b = dlsym(handle, "dsyevr64_");
        if (sym64b) {
            c->dsyevr_ilp64 = (jlinalg_dsyevr_ilp64_fn)sym64b;
            c->has_dsyevr = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevr64_\n");
            return;
        }
        return;
    }

    /* LP64 Fortran dsyevr */
    void *fsym = dlsym(handle, "dsyevr_");
    if (fsym) {
        c->dsyevr_lp64 = (jlinalg_dsyevr_lp64_fn)fsym;
        c->has_dsyevr = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dsyevr_ (LP64)\n");
    }
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dgeqrf (QR factorization)
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dgeqrf(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
#ifdef __APPLE__
        void *sym = dlsym(handle, "dgeqrf$NEWLAPACK$ILP64");
        if (sym) {
            c->dgeqrf_ilp64 = (jlinalg_dgeqrf_ilp64_fn)sym;
            c->has_dgeqrf = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgeqrf$NEWLAPACK$ILP64\n");
        }
        return;
#endif
        void *sym64 = dlsym(handle, "dgeqrf_64_");
        if (sym64) {
            c->dgeqrf_ilp64 = (jlinalg_dgeqrf_ilp64_fn)sym64;
            c->has_dgeqrf = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgeqrf_64_\n");
            return;
        }
        void *sym64b = dlsym(handle, "dgeqrf64_");
        if (sym64b) {
            c->dgeqrf_ilp64 = (jlinalg_dgeqrf_ilp64_fn)sym64b;
            c->has_dgeqrf = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgeqrf64_\n");
            return;
        }
        return;
    }

    /* LP64 Fortran dgeqrf */
    void *fsym = dlsym(handle, "dgeqrf_");
    if (fsym) {
        c->dgeqrf_lp64 = (jlinalg_dgeqrf_lp64_fn)fsym;
        c->has_dgeqrf = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgeqrf_ (LP64)\n");
    }
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dorgqr (generate Q from Householder reflectors)
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dorgqr(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
#ifdef __APPLE__
        void *sym = dlsym(handle, "dorgqr$NEWLAPACK$ILP64");
        if (sym) {
            c->dorgqr_ilp64 = (jlinalg_dorgqr_ilp64_fn)sym;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dorgqr$NEWLAPACK$ILP64\n");
        }
        return;
#endif
        void *sym64 = dlsym(handle, "dorgqr_64_");
        if (sym64) {
            c->dorgqr_ilp64 = (jlinalg_dorgqr_ilp64_fn)sym64;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dorgqr_64_\n");
            return;
        }
        void *sym64b = dlsym(handle, "dorgqr64_");
        if (sym64b) {
            c->dorgqr_ilp64 = (jlinalg_dorgqr_ilp64_fn)sym64b;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dorgqr64_\n");
            return;
        }
        return;
    }

    void *fsym = dlsym(handle, "dorgqr_");
    if (fsym) {
        c->dorgqr_lp64 = (jlinalg_dorgqr_lp64_fn)fsym;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dorgqr_ (LP64)\n");
    }
}

/* ---------------------------------------------------------------------------
 * Symbol resolution — dgesvd (SVD)
 * ---------------------------------------------------------------------------
 */
static void try_resolve_dgesvd(void *handle, blas_candidate_t *c) {
    int dbg = _debug_enabled();

    if (c->is_ilp64) {
#ifdef __APPLE__
        void *sym = dlsym(handle, "dgesvd$NEWLAPACK$ILP64");
        if (sym) {
            c->dgesvd_ilp64 = (jlinalg_dgesvd_ilp64_fn)sym;
            c->has_dgesvd = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgesvd$NEWLAPACK$ILP64\n");
        }
        return;
#endif
        void *sym64 = dlsym(handle, "dgesvd_64_");
        if (sym64) {
            c->dgesvd_ilp64 = (jlinalg_dgesvd_ilp64_fn)sym64;
            c->has_dgesvd = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgesvd_64_\n");
            return;
        }
        void *sym64b = dlsym(handle, "dgesvd64_");
        if (sym64b) {
            c->dgesvd_ilp64 = (jlinalg_dgesvd_ilp64_fn)sym64b;
            c->has_dgesvd = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgesvd64_\n");
            return;
        }
        return;
    }

    void *fsym = dlsym(handle, "dgesvd_");
    if (fsym) {
        c->dgesvd_lp64 = (jlinalg_dgesvd_lp64_fn)fsym;
        c->has_dgesvd = 1;
        if (dbg) fprintf(stderr, "jlinalg_dispatch:   resolved dgesvd_ (LP64)\n");
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
            try_resolve_dsyrk(handle, c);
            try_resolve_dsyevd(handle, c);
            try_resolve_dsyevr(handle, c);
            try_resolve_dgeqrf(handle, c);
            try_resolve_dorgqr(handle, c);
            try_resolve_dgesvd(handle, c);
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
            try_resolve_dsyrk(handle, c);
            try_resolve_dsyevd(handle, c);
            try_resolve_dsyevr(handle, c);
            try_resolve_dgeqrf(handle, c);
            try_resolve_dorgqr(handle, c);
            try_resolve_dgesvd(handle, c);
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
 * discover_system_blas -- Full system BLAS discovery (4-step pattern)
 * Populates a blas_candidate_t instead of setting globals.
 * ---------------------------------------------------------------------------
 */
static void discover_system_blas(blas_candidate_t *c) {
    int dbg = _debug_enabled();

    /* Step 1: RTLD_DEFAULT (catches macOS Accelerate, LD_PRELOAD) */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: step 1 -- RTLD_DEFAULT\n");
    if (try_resolve_dgemm_candidate(RTLD_DEFAULT, NULL, c)) {
        if (dbg)
            fprintf(stderr, "jlinalg_dispatch: found via RTLD_DEFAULT (ilp64=%d, backend=%s)\n",
                    c->is_ilp64, c->name);
        try_resolve_dsyrk(RTLD_DEFAULT, c);
        try_resolve_dsyevd(RTLD_DEFAULT, c);
        try_resolve_dsyevr(RTLD_DEFAULT, c);
        try_resolve_dgeqrf(RTLD_DEFAULT, c);
        try_resolve_dorgqr(RTLD_DEFAULT, c);
        try_resolve_dgesvd(RTLD_DEFAULT, c);
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
        try_resolve_dsyrk(RTLD_DEFAULT, c);
        try_resolve_dsyevd(RTLD_DEFAULT, c);
        try_resolve_dsyevr(RTLD_DEFAULT, c);
        try_resolve_dgeqrf(RTLD_DEFAULT, c);
        try_resolve_dorgqr(RTLD_DEFAULT, c);
        try_resolve_dgesvd(RTLD_DEFAULT, c);
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

    /* Step 4: Scan numpy's lib directories */
    if (dbg) fprintf(stderr, "jlinalg_dispatch: step 4 -- numpy dir scan\n");
    PyObject *np2 = PyImport_ImportModule("numpy");
    if (!np2) {
        PyErr_Clear();
        return;
    }

    PyObject *np_file = PyObject_GetAttrString(np2, "__file__");
    if (!np_file) {
        PyErr_Clear();
        Py_DECREF(np2);
        return;
    }

    PyObject *pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) {
        PyErr_Clear();
        Py_DECREF(np_file);
        Py_DECREF(np2);
        return;
    }

    PyObject *Path = PyObject_GetAttrString(pathlib, "Path");
    if (!Path) {
        PyErr_Clear();
        Py_DECREF(pathlib);
        Py_DECREF(np_file);
        Py_DECREF(np2);
        return;
    }

    PyObject *p = PyObject_CallFunctionObjArgs(Path, np_file, NULL);
    Py_DECREF(np_file);
    if (!p) {
        PyErr_Clear();
        Py_DECREF(Path);
        Py_DECREF(pathlib);
        Py_DECREF(np2);
        return;
    }

    PyObject *resolved = PyObject_CallMethod(p, "resolve", NULL);
    Py_DECREF(p);
    if (!resolved) {
        PyErr_Clear();
        Py_DECREF(Path);
        Py_DECREF(pathlib);
        Py_DECREF(np2);
        return;
    }

    PyObject *np_dir = PyObject_GetAttrString(resolved, "parent");
    Py_DECREF(resolved);
    if (!np_dir) {
        PyErr_Clear();
        Py_DECREF(Path);
        Py_DECREF(pathlib);
        Py_DECREF(np2);
        return;
    }

    const char *subpaths[] = {".libs", "_core/.libs", NULL};
    for (int si = 0; subpaths[si]; si++) {
        PyObject *candidate = PyObject_CallMethod(np_dir, "__truediv__", "s", subpaths[si]);
        if (!candidate) {
            PyErr_Clear();
            continue;
        }
        PyObject *cstr = PyObject_Str(candidate);
        Py_DECREF(candidate);
        if (!cstr) {
            PyErr_Clear();
            continue;
        }
        const char *dirpath = PyUnicode_AsUTF8(cstr);
        if (dirpath && scan_dir_for_blas_candidate(dirpath, c)) {
            Py_DECREF(cstr);
            Py_DECREF(np_dir);
            Py_DECREF(Path);
            Py_DECREF(pathlib);
            Py_DECREF(np2);
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
                    Py_DECREF(cstr);
                    Py_DECREF(np_parent);
                    Py_DECREF(np_dir);
                    Py_DECREF(Path);
                    Py_DECREF(pathlib);
                    Py_DECREF(np2);
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

    Py_DECREF(np_dir);
    Py_DECREF(Path);
    Py_DECREF(pathlib);
    Py_DECREF(np2);
}

/* ---------------------------------------------------------------------------
 * discover_pip_mkl -- Look for pip-installed MKL (site-packages/mkl)
 * ---------------------------------------------------------------------------
 */
static void discover_pip_mkl(blas_candidate_t *c) {
    int dbg = _debug_enabled();
    if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- trying import mkl\n");

    PyObject *mkl = PyImport_ImportModule("mkl");
    if (!mkl) {
        PyErr_Clear();
        if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- mkl module not found\n");
        return;
    }

    PyObject *mkl_file = PyObject_GetAttrString(mkl, "__file__");
    Py_DECREF(mkl);
    if (!mkl_file) {
        PyErr_Clear();
        if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- mkl.__file__ not found\n");
        return;
    }

    PyObject *pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) {
        PyErr_Clear();
        Py_DECREF(mkl_file);
        return;
    }

    PyObject *Path = PyObject_GetAttrString(pathlib, "Path");
    Py_DECREF(pathlib);
    if (!Path) {
        PyErr_Clear();
        Py_DECREF(mkl_file);
        return;
    }

    PyObject *mkl_path = PyObject_CallFunctionObjArgs(Path, mkl_file, NULL);
    Py_DECREF(mkl_file);
    if (!mkl_path) {
        PyErr_Clear();
        Py_DECREF(Path);
        return;
    }

    PyObject *mkl_dir = PyObject_GetAttrString(mkl_path, "parent");
    Py_DECREF(mkl_path);
    if (!mkl_dir) {
        PyErr_Clear();
        Py_DECREF(Path);
        return;
    }

    /* Try mkl_dir / 'mkl.libs' and mkl_dir.parent / 'mkl.libs' */
    const char *mkl_lib_paths[] = {"mkl.libs", NULL};

    for (int attempt = 0; attempt < 2; attempt++) {
        PyObject *base = attempt == 0 ? mkl_dir : PyObject_GetAttrString(mkl_dir, "parent");
        if (!base) {
            PyErr_Clear();
            continue;
        }

        for (int pi = 0; mkl_lib_paths[pi]; pi++) {
            PyObject *libs_dir = PyObject_CallMethod(base, "__truediv__", "s", mkl_lib_paths[pi]);
            if (!libs_dir) {
                PyErr_Clear();
                continue;
            }
            PyObject *libs_str = PyObject_Str(libs_dir);
            Py_DECREF(libs_dir);
            if (!libs_str) {
                PyErr_Clear();
                continue;
            }
            const char *dirpath = PyUnicode_AsUTF8(libs_str);
            if (!dirpath) {
                Py_DECREF(libs_str);
                continue;
            }

            if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- trying dir: %s\n", dirpath);

            /* MKL libraries must be loaded in dependency order:
             * core first, then sequential, then ilp64 */
            const char *mkl_libs[] = {"libmkl_core", "libmkl_sequential", "libmkl_intel_ilp64",
                                      NULL};
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
                        if (dbg)
                            fprintf(stderr, "jlinalg_dispatch: pip-mkl -- loaded %s\n",
                                    entry->d_name);
                    } else {
                        if (dbg)
                            fprintf(stderr, "jlinalg_dispatch: pip-mkl -- dlopen failed: %s\n",
                                    dlerror());
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
                        if (dbg)
                            fprintf(stderr,
                                    "jlinalg_dispatch: pip-mkl -- "
                                    "WARNING: resolved LP64 dgemm from ILP64 MKL path, skipping\n");
                        c->found = 0;
                        Py_DECREF(libs_str);
                        if (attempt == 1) Py_DECREF(base);
                        Py_DECREF(mkl_dir);
                        Py_DECREF(Path);
                        return;
                    }
                    c->name = "MKL-ILP64";
                    try_resolve_dsyrk(RTLD_DEFAULT, c);
                    try_resolve_dsyevd(RTLD_DEFAULT, c);
                    try_resolve_dsyevr(RTLD_DEFAULT, c);
                    try_resolve_dgeqrf(RTLD_DEFAULT, c);
                    try_resolve_dorgqr(RTLD_DEFAULT, c);
                    try_resolve_dgesvd(RTLD_DEFAULT, c);
                    if (dbg)
                        fprintf(stderr,
                                "jlinalg_dispatch: pip-mkl -- resolved (ilp64=%d, lapack=%d)\n",
                                c->is_ilp64, c->has_lapack);
                    Py_DECREF(libs_str);
                    if (attempt == 1) Py_DECREF(base);
                    Py_DECREF(mkl_dir);
                    Py_DECREF(Path);
                    return;
                }
            }
            Py_DECREF(libs_str);
        }
        if (attempt == 1) Py_DECREF(base);
    }

    Py_DECREF(mkl_dir);
    Py_DECREF(Path);
    if (dbg) fprintf(stderr, "jlinalg_dispatch: pip-mkl -- not found\n");
}

/* ---------------------------------------------------------------------------
 * Candidate validation and selection
 *
 * _validate_candidate: ensures capability flags match resolved pointers.
 * _score_candidate:    ILP64 + LAPACK = 4, ILP64 BLAS-only = 3, LP64 = 1.
 * select_best_backend: returns highest-scoring candidate (NULL if none).
 *   LP64 candidates are returned for logging but not wired for dgemm.
 * ---------------------------------------------------------------------------
 */

/* Validate candidate invariants.  Returns 1 if valid, 0 if inconsistent.
 * When invalid, zeros out the candidate (found=0) so it cannot be selected
 * — prevents NULL function pointer dereferences from broken discovery. */
static int _validate_candidate(blas_candidate_t *c, const char *label) {
    if (!c->found) return 1; /* not-found is always valid */
    int valid = 1;
    /* found=1 requires at least one dgemm pointer */
    if (!c->dgemm_lp64 && !c->dgemm_ilp64 && !c->cblas_dgemm && !c->cblas_dgemm_ilp64) {
        fprintf(stderr, "jlinalg_dispatch: WARN: %s found=1 but no dgemm pointers — disabling\n",
                label);
        valid = 0;
    }
    /* has_lapack requires at least one dsyevd pointer */
    if (c->has_lapack && !c->dsyevd_lp64 && !c->dsyevd_ilp64 && !c->lapacke_dsyevd_lp64 &&
        !c->lapacke_dsyevd_ilp64) {
        fprintf(stderr,
                "jlinalg_dispatch: WARN: %s has_lapack=1 but no dsyevd pointers — disabling\n",
                label);
        valid = 0;
    }
    /* has_lapacke_dsyevd requires has_lapack */
    if (c->has_lapacke_dsyevd && !c->has_lapack) {
        fprintf(stderr,
                "jlinalg_dispatch: WARN: %s has_lapacke_dsyevd=1 but has_lapack=0 — disabling\n",
                label);
        valid = 0;
    }
    /* has_dsyrk requires at least one dsyrk pointer */
    if (c->has_dsyrk && !c->cblas_dsyrk && !c->cblas_dsyrk_ilp64 && !c->dsyrk_lp64 &&
        !c->dsyrk_ilp64) {
        fprintf(stderr,
                "jlinalg_dispatch: WARN: %s has_dsyrk=1 but no dsyrk pointers — disabling\n",
                label);
        valid = 0;
    }
    /* has_dsyevr requires at least one dsyevr pointer */
    if (c->has_dsyevr && !c->dsyevr_lp64 && !c->dsyevr_ilp64) {
        fprintf(stderr,
                "jlinalg_dispatch: WARN: %s has_dsyevr=1 but no dsyevr pointers — disabling\n",
                label);
        valid = 0;
    }
    if (!valid) {
        /* Zero out the candidate so it cannot be selected */
        const char *saved_name = c->name;
        memset(c, 0, sizeof(*c));
        c->name = saved_name; /* preserve for diagnostic logging */
    }
    return valid;
}

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
 * LP64 overflow guard — shared by both the simplified and full-signature
 * dispatch wrappers.  Returns 1 if overflow detected (LP64 vendor BLAS
 * cannot handle these dimensions), 0 if dimensions fit in int32.
 * ---------------------------------------------------------------------------
 */
static int _lp64_overflow_guard(npy_intp M, npy_intp N, npy_intp K, npy_intp lda, npy_intp ldb,
                                npy_intp ldc) {
    if (g_is_ilp64) return 0;
    if (M <= LP64_DIM_MAX && N <= LP64_DIM_MAX && K <= LP64_DIM_MAX && lda <= LP64_DIM_MAX &&
        ldb <= LP64_DIM_MAX && ldc <= LP64_DIM_MAX)
        return 0;

    __atomic_add_fetch(&g_lp64_overflow_count, 1, __ATOMIC_RELAXED);
    static int warned = 0;
    if (!warned) {
        warned = 1;
        fprintf(stderr,
                "jlinalg_dispatch: WARNING: LP64 overflow guard triggered "
                "(M=%ld N=%ld K=%ld > %d). Result zeroed — install ILP64 numpy "
                "for large matrices.\n",
                (long)M, (long)N, (long)K, LP64_DIM_MAX);
    }
    return 1;
}

static int g_has_vendor_dgemm = 0; /* set to 1 when vendor dgemm is wired */

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

    /* Validate invariants — invalid candidates are zeroed out (found=0)
     * so they cannot be selected, preventing NULL dereferences. */
    _validate_candidate(&system, "system");
    _validate_candidate(&pip_mkl, "pip_mkl");

    blas_candidate_t *best = select_best_backend(&system, &pip_mkl);

    if (best && best->is_ilp64) {
        /* ILP64 backend — wire dgemm */
        if (_no_vendor_dgemm()) {
            fprintf(stderr, "jlinalg_dispatch: INFO: JLINALG_NO_VENDOR_DGEMM set -- "
                            "vendor dgemm left unwired, numpy fallback in use.\n");
        } else {
            if (dbg) fprintf(stderr, "jlinalg_dispatch: using %s (ILP64) for dgemm\n", best->name);
            g_dgemm_ilp64 = best->dgemm_ilp64;
            g_dgemm_lp64 = best->dgemm_lp64;
            g_cblas_dgemm = best->cblas_dgemm;
            g_cblas_dgemm_ilp64 = best->cblas_dgemm_ilp64;
            g_has_vendor_dgemm = 1;
        }
        g_is_ilp64 = 1;
        g_backend_name = best->name;
        g_blas_handle = best->handle;

        /* Wire dsyrk — only ILP64 pointers (LP64 dsyrk is not dispatched,
         * same policy as dgemm: LP64 is not wired for numerical consistency) */
        if (best->has_dsyrk) {
            g_cblas_dsyrk_ilp64 = best->cblas_dsyrk_ilp64;
            g_dsyrk_ilp64 = best->dsyrk_ilp64;
            g_has_dsyrk = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch: vendor dsyrk wired\n");
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
                if (dbg)
                    fprintf(stderr, "jlinalg_dispatch: vendor LAPACKE dsyevd wired (row-major)\n");
            } else {
                if (dbg)
                    fprintf(
                        stderr,
                        "jlinalg_dispatch: vendor dsyevd wired (Fortran, transpose required)\n");
            }
        }

        /* Wire dsyevr — memory-pressure fallback (O(N) workspace) */
        if (best->has_dsyevr) {
            g_dsyevr_lp64 = best->dsyevr_lp64;
            g_dsyevr_ilp64 = best->dsyevr_ilp64;
            g_has_dsyevr = 1;
            if (dbg)
                fprintf(stderr,
                        "jlinalg_dispatch: vendor dsyevr wired (memory-pressure fallback)\n");
        }

        /* Wire dgeqrf + dorgqr — only if BOTH are available */
        if (best->has_dgeqrf && (best->dorgqr_ilp64 || best->dorgqr_lp64)) {
            g_dgeqrf_lp64 = best->dgeqrf_lp64;
            g_dgeqrf_ilp64 = best->dgeqrf_ilp64;
            g_dorgqr_lp64 = best->dorgqr_lp64;
            g_dorgqr_ilp64 = best->dorgqr_ilp64;
            g_has_dgeqrf = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch: vendor dgeqrf + dorgqr wired\n");
        }

        /* Wire dgesvd */
        if (best->has_dgesvd) {
            g_dgesvd_lp64 = best->dgesvd_lp64;
            g_dgesvd_ilp64 = best->dgesvd_ilp64;
            g_has_dgesvd = 1;
            if (dbg) fprintf(stderr, "jlinalg_dispatch: vendor dgesvd wired\n");
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
        /* Reset backend name -- LP64 is available but not active */
        g_backend_name = "numpy-fallback";
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
    return g_is_ilp64;
}

int blas_has_external(void) {
    /* Only true when external BLAS is actually wired (i.e., ILP64 found).
     * LP64-only discovery does not wire dispatch. */
    return g_has_vendor_dgemm;
}

int blas_has_dsyrk(void) {
    return g_has_dsyrk;
}
int blas_has_dsyevd(void) {
    return g_has_dsyevd;
}
int blas_has_lapacke_dsyevd(void) {
    return g_has_lapacke_dsyevd;
}
int blas_has_dsyevr(void) {
    return g_has_dsyevr && g_is_ilp64;
}
int blas_has_dgeqrf(void) {
    return g_has_dgeqrf && g_is_ilp64;
}
int blas_has_dgesvd(void) {
    return g_has_dgesvd && g_is_ilp64;
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
    if (g_has_dsyrk && g_is_ilp64) {
        if (g_cblas_dsyrk_ilp64) {
            /* Row-major, lower, no-trans: C = X @ X.T + beta * C */
            g_cblas_dsyrk_ilp64(JLINALG_CblasRowMajor, JLINALG_CblasLower, JLINALG_CblasNoTrans,
                                (long)N, (long)K, 1.0, X, (long)ldx, beta, C, (long)ldc);
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
            const double alpha = 1.0;
            g_dsyrk_ilp64("U", "T", &n, &k, &alpha, X, &lda, &beta, C, &ldc_f);
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
    if (!g_has_dsyevd || !g_is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    /* --- LAPACKE path (MKL): row-major natively, no transpose needed.
     * Only used when Fortran ILP64 dsyevd is NOT available.  When both
     * exist, we prefer Fortran because dsyevd_64_ is an unambiguous ILP64
     * symbol, whereas LAPACKE_dsyevd is unsuffixed and could resolve to
     * the LP64 variant on systems with mixed LP64/ILP64 MKL. --- */
    if (g_has_lapacke_dsyevd && g_lapacke_dsyevd_ilp64 && !g_dsyevd_ilp64) {
        long long info = g_lapacke_dsyevd_ilp64(JLINALG_LAPACK_ROW_MAJOR, 'V', 'L', (long long)N, K,
                                                (long long)ldk, eigenvalues);
        if (info != 0) return _info_to_int(info, N);
        return JLINALG_EXT_SUCCESS;
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
        g_dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues, &work_query, &lwork, &iwork_query,
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
        g_dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues, work, &lwork, iwork, &liwork, &info);
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
    if (!g_has_dsyevr || !g_is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    if (g_dsyevr_ilp64) {
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
        g_dsyevr_ilp64("V", "A", "U", &n, K, &lda, &vl, &vu, &il, &iu, &abstol, &m_out, eigenvalues,
                       eigenvectors, &ldz_f, isuppz_dummy, &work_query, &lwork, &iwork_query,
                       &liwork, &info);
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
        g_dsyevr_ilp64("V", "A", "U", &n, K, &lda, &vl, &vu, &il, &iu, &abstol, &m_out, eigenvalues,
                       Z_col, &ldz_f, isuppz, work, &lwork, iwork, &liwork, &info);
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
 * jlinalg_dgeqrf_ext — Vendor-dispatch QR factorization (dgeqrf)
 * ---------------------------------------------------------------------------
 */
int jlinalg_dgeqrf_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, double *tau) {
    if (!g_has_dgeqrf || !g_is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    if (g_dgeqrf_ilp64) {
        long long lm = (long long)m, ln = (long long)n;
        long long llda = (long long)lda;
        long long info = 0;

        /* Workspace query */
        long long lwork = -1;
        double work_query;
        g_dgeqrf_ilp64(&lm, &ln, A_col, &llda, tau, &work_query, &lwork, &info);
        if (info != 0) return _info_to_int(info, m);

        lwork = (long long)work_query + 1;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work) return JLINALG_EXT_ALLOC_FAIL;

        g_dgeqrf_ilp64(&lm, &ln, A_col, &llda, tau, work, &lwork, &info);
        free(work);
        if (info != 0) return _info_to_int(info, m);
        return JLINALG_EXT_SUCCESS;
    }
    return JLINALG_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * jlinalg_dorgqr_ext — Vendor-dispatch generate Q from Householder (dorgqr)
 * ---------------------------------------------------------------------------
 */
int jlinalg_dorgqr_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, const double *tau) {
    if (!g_has_dgeqrf || !g_is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    if (g_dorgqr_ilp64) {
        long long lm = (long long)m, ln = (long long)n, lk = (long long)n;
        long long llda = (long long)lda;
        long long info = 0;

        /* Workspace query */
        long long lwork = -1;
        double work_query;
        g_dorgqr_ilp64(&lm, &ln, &lk, A_col, &llda, tau, &work_query, &lwork, &info);
        if (info != 0) return _info_to_int(info, m);

        lwork = (long long)work_query + 1;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work) return JLINALG_EXT_ALLOC_FAIL;

        g_dorgqr_ilp64(&lm, &ln, &lk, A_col, &llda, tau, work, &lwork, &info);
        free(work);
        if (info != 0) return _info_to_int(info, m);
        return JLINALG_EXT_SUCCESS;
    }
    return JLINALG_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * jlinalg_dgesvd_ext — Vendor-dispatch SVD (dgesvd)
 * ---------------------------------------------------------------------------
 */
int jlinalg_dgesvd_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, double *s,
                       double *U_col, npy_intp ldu, double *Vt_col, npy_intp ldvt, int compute_uv) {
    if (!g_has_dgesvd || !g_is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    if (g_dgesvd_ilp64) {
        const char *jobu = compute_uv ? "S" : "N";
        const char *jobvt = compute_uv ? "S" : "N";
        long long lm = (long long)m, ln = (long long)n;
        long long llda = (long long)lda;
        long long lldu = (long long)ldu;
        long long lldvt = (long long)ldvt;
        long long info = 0;

        /* Workspace query */
        long long lwork = -1;
        double work_query;
        g_dgesvd_ilp64(jobu, jobvt, &lm, &ln, A_col, &llda, s, U_col, &lldu, Vt_col, &lldvt,
                       &work_query, &lwork, &info);
        if (info != 0) return _info_to_int(info, m);

        lwork = (long long)work_query + 1;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work) return JLINALG_EXT_ALLOC_FAIL;

        g_dgesvd_ilp64(jobu, jobvt, &lm, &ln, A_col, &llda, s, U_col, &lldu, Vt_col, &lldvt, work,
                       &lwork, &info);
        free(work);
        if (info != 0) return _info_to_int(info, m);
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
    if (_lp64_overflow_guard(M, N, K, lda, ldb, ldc)) return 0;

    if (g_cblas_dgemm_ilp64) {
        int ta = transa ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        int tb = transb ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        long llda = (long)(lda > 0 ? lda : 1);
        long lldb = (long)(ldb > 0 ? ldb : 1);
        long lldc = (long)(ldc > 0 ? ldc : 1);
        g_cblas_dgemm_ilp64(JLINALG_CblasRowMajor, ta, tb, (long)M, (long)N, (long)K, alpha, A,
                            llda, B, lldb, beta, C, lldc);
        return 1;
    }
    if (g_cblas_dgemm) {
        int ta = transa ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        int tb = transb ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        int ilda = (int)(lda > 0 ? lda : 1);
        int ildb = (int)(ldb > 0 ? ldb : 1);
        int ildc = (int)(ldc > 0 ? ldc : 1);
        g_cblas_dgemm(JLINALG_CblasRowMajor, ta, tb, (int)M, (int)N, (int)K, alpha, A, ilda, B,
                      ildb, beta, C, ildc);
        return 1;
    }

    /* Fortran interface fallback: row-major -> column-major swap */
    const char *transa_f = transb ? "T" : "N";
    const char *transb_f = transa ? "T" : "N";

    if (g_is_ilp64) {
        const long long lM = (long long)M, lN = (long long)N, lK = (long long)K;
        const long long llda = (long long)lda, lldb = (long long)ldb;
        const long long lldc = (long long)ldc;
        g_dgemm_ilp64(transa_f, transb_f, &lN, &lM, &lK, &alpha, B, &lldb, A, &llda, &beta, C,
                      &lldc);
    } else {
        const int iM = (int)M, iN = (int)N, iK = (int)K;
        const int ilda = (int)lda, ildb = (int)ldb, ildc = (int)ldc;
        g_dgemm_lp64(transa_f, transb_f, &iN, &iM, &iK, &alpha, B, &ildb, A, &ilda, &beta, C,
                     &ildc);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * Public full-signature dispatch API
 * ---------------------------------------------------------------------------
 */

void jlinalg_dgemm_ext(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                       const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                       int transb) {
    if ((g_dgemm_lp64 || g_dgemm_ilp64) &&
        _dgemm_external_full(M, N, K, A, lda, B, ldb, C, ldc, transa, transb, 1.0, 0.0)) {
        return;
    }
    /* No external BLAS, or LP64 overflow guard triggered.
     * Caller should check blas_has_external() and use numpy fallback. */
    fprintf(stderr, "FATAL: jlinalg_dgemm_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

void jlinalg_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                          const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                          int transb, double alpha, double beta) {
    if ((g_dgemm_lp64 || g_dgemm_ilp64) &&
        _dgemm_external_full(M, N, K, A, lda, B, ldb, C, ldc, transa, transb, alpha, beta)) {
        return;
    }
    /* No external BLAS -- caller should use numpy fallback. */
    fprintf(stderr, "FATAL: jlinalg_dgemm_ext_ws called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

#else /* _WIN32 */

/* Windows: no external dispatch -- numpy fallback */
int blas_dispatch_init(void) {
    return 0;
}

const char *blas_backend_name(void) {
    return "numpy-fallback";
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

int blas_has_dsyevr(void) {
    return 0;
}

int blas_has_lapacke_dsyevd(void) {
    return 0;
}

int blas_has_dgeqrf(void) {
    return 0;
}

int blas_has_dgesvd(void) {
    return 0;
}

int jlinalg_dgeqrf_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, double *tau) {
    (void)m;
    (void)n;
    (void)A_col;
    (void)lda;
    (void)tau;
    return JLINALG_EXT_UNAVAILABLE;
}

int jlinalg_dorgqr_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, const double *tau) {
    (void)m;
    (void)n;
    (void)A_col;
    (void)lda;
    (void)tau;
    return JLINALG_EXT_UNAVAILABLE;
}

int jlinalg_dgesvd_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, double *s,
                       double *U_col, npy_intp ldu, double *Vt_col, npy_intp ldvt, int compute_uv) {
    (void)m;
    (void)n;
    (void)A_col;
    (void)lda;
    (void)s;
    (void)U_col;
    (void)ldu;
    (void)Vt_col;
    (void)ldvt;
    (void)compute_uv;
    return JLINALG_EXT_UNAVAILABLE;
}

int jlinalg_dsyevr_ext(npy_intp N, double *K, npy_intp ldk, double *eigenvalues,
                       double *eigenvectors, npy_intp ldz) {
    (void)N;
    (void)K;
    (void)ldk;
    (void)eigenvalues;
    (void)eigenvectors;
    (void)ldz;
    return JLINALG_EXT_UNAVAILABLE;
}

void jlinalg_dsyrk_ext(npy_intp N, npy_intp K, const double *X, npy_intp ldx, double *C,
                       npy_intp ldc, double beta) {
    (void)N;
    (void)K;
    (void)X;
    (void)ldx;
    (void)C;
    (void)ldc;
    (void)beta;
    fprintf(stderr, "FATAL: jlinalg_dsyrk_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

int jlinalg_dsyevd_ext(npy_intp N, double *K, npy_intp ldk, double *eigenvalues) {
    (void)N;
    (void)K;
    (void)ldk;
    (void)eigenvalues;
    return JLINALG_EXT_UNAVAILABLE;
}

void jlinalg_dgemm_ext(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                       const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                       int transb) {
    (void)A;
    (void)B;
    (void)K;
    (void)lda;
    (void)ldb;
    (void)transa;
    (void)transb;
    (void)M;
    (void)N;
    (void)C;
    (void)ldc;
    fprintf(stderr, "FATAL: jlinalg_dgemm_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

void jlinalg_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                          const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                          int transb, double alpha, double beta) {
    (void)M;
    (void)N;
    (void)K;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)transa;
    (void)transb;
    (void)alpha;
    (void)beta;
    fprintf(stderr, "FATAL: jlinalg_dgemm_ext_ws called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

int blas_dispatch_lp64_overflow_count(void) {
    return 0;
}

void blas_dispatch_reset_lp64_overflow(void) {}

#endif /* !_WIN32 */
