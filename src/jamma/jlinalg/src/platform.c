/**
 * jlinalg_init() is idempotent: repeated calls are safe and cheap
 * (guarded by a static flag).
 */

#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> /* npy_intp */
#include "jlinalg.h"

/* Initialisation guard */
static int _initialized = 0;

/* Thread count -- exposed via get/set API */
static int _n_threads = 1;

/* ---------------------------------------------------------------------------
 * Thread safety: called from PyInit__jlinalg under the GIL during module import.
 * No additional synchronization is needed; fork() children inherit the
 * already-initialized state.
 * ---------------------------------------------------------------------------
 */
int jlinalg_init(void) {
    if (_initialized) return 0;

    /* Determine default thread count from environment */
    const char *omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        int t = atoi(omp_threads);
        if (t > 0) _n_threads = t;
    }

    /* Discover and wire vendor BLAS/LAPACK dispatch.
     * blas_dispatch_init() may wire external dgemm, dsyrk, dsyevd, etc.
     * Falls through to numpy-fallback on failure. */
    blas_dispatch_init();

    _initialized = 1;
    return 0;
}

const char *jlinalg_isa_name(void) {
#if defined(__AVX2__)
    return "AVX2";
#elif defined(__ARM_NEON)
    return "NEON";
#else
    return "generic";
#endif
}

/* ---------------------------------------------------------------------------
 * Thread control API
 * ---------------------------------------------------------------------------
 */

int jlinalg_get_n_threads(void) {
    return __atomic_load_n(&_n_threads, __ATOMIC_RELAXED);
}

int jlinalg_set_n_threads(int n) {
    if (n < 1) return -1;
    int old = __atomic_load_n(&_n_threads, __ATOMIC_RELAXED);
    __atomic_store_n(&_n_threads, n, __ATOMIC_RELAXED);
    return old;
}
