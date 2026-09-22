/**
 * eigh.c -- Driver for jlinalg symmetric eigendecomposition (vendor-only).
 *
 * Implements jlinalg_eigh_c: computes all eigenvalues and eigenvectors of a
 * symmetric N x N matrix K (row-major, lower triangle) using vendor LAPACK:
 *
 *   1. Try vendor DSYEVD (O(N^2) workspace, fast)
 *   2. Fall back to vendor DSYEVR (O(N) workspace, memory-pressure fallback)
 *   3. Return JLINALG_EXT_UNAVAILABLE if neither vendor routine is available
 *
 * On exit:
 *   eigenvalues[k]    = k-th eigenvalue (ascending)
 *   eigenvectors[i,j] = i-th component of j-th eigenvector, column j of Z
 *                       (row-major: eigenvectors[row*ldz + col])
 *
 * Error propagation:
 *   Returns 0 on success.
 *   Returns JLINALG_EXT_UNAVAILABLE if no vendor LAPACK available.
 *   Returns JLINALG_EXT_ALLOC_FAIL if workspace allocation fails.
 *   Returns JLINALG_EXT_BAD_STRIDE if ldk != N or ldz != N (unsupported).
 *   Returns positive i on LAPACK convergence failure.
 *   Returns negative -i on LAPACK illegal-argument error (surfaced, not swallowed).
 *
 * See jlinalg_eigh_status_t in jlinalg.h for status->driver_used.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jlinalg.h"

static int eigh_dsyevr(npy_intp N, double *K, double *eigenvalues, double *eigenvectors,
                       jlinalg_eigh_status_t *status) {
    int ret = jlinalg_dsyevr_ext(N, K, N, eigenvalues, eigenvectors, N);
    if (ret == JLINALG_EXT_SUCCESS) {
        if (status) status->driver_used = JLINALG_DRIVER_DSYEVR;
        return 0;
    }
    if (ret == JLINALG_EXT_ALLOC_FAIL) {
        fprintf(stderr,
                "jlinalg_eigh_c: vendor dsyevr workspace allocation failed "
                "(N=%ld) -- no smaller-footprint driver left\n",
                (long)N);
        if (status) status->vendor_lapack_skipped = 1;
    }
    return ret;
}

/* ---------------------------------------------------------------------------
 * jlinalg_eigh_c -- Full symmetric eigensolver (public API, vendor-only).
 *
 * Parameters: see jlinalg.h
 * Returns: 0 on success, JLINALG_EXT_UNAVAILABLE if no vendor LAPACK,
 *          JLINALG_EXT_ALLOC_FAIL on allocation failure, positive i on
 *          convergence failure, negative -i on LAPACK illegal-argument error.
 * ---------------------------------------------------------------------------
 */
int jlinalg_eigh_c(npy_intp N, double *K, npy_intp ldk, double *eigenvalues, double *eigenvectors,
                   npy_intp ldz, int require_dsyevr, jlinalg_eigh_status_t *status) {
    if (N <= 0) return 0;

    /* Every caller passes ldk == ldz == N (tight row-major storage). A padded
     * stride has no exerciser in this tree and no test, so it is a contract
     * violation rather than a case to service. */
    if (ldk != N || ldz != N) return JLINALG_EXT_BAD_STRIDE;

    if (N == 1) {
        eigenvalues[0] = K[0];
        eigenvectors[0] = 1.0;
        if (status) status->driver_used = JLINALG_DRIVER_NONE;
        return 0;
    }

    if (require_dsyevr) return eigh_dsyevr(N, K, eigenvalues, eigenvectors, status);

    /* --- Vendor dsyevd fast path ---
     * When vendor LAPACK dsyevd is available (Accelerate, MKL), call it
     * directly.  Vendor dsyevd is production-hardened, multithreaded, and
     * handles edge cases robustly.
     *
     * jlinalg_dsyevd_ext returns:
     *   JLINALG_EXT_SUCCESS:     K holds eigenvectors (row-major), eigenvalues filled
     *   JLINALG_EXT_UNAVAILABLE: no vendor dsyevd -- fall through to dsyevr
     *   JLINALG_EXT_ALLOC_FAIL:  workspace allocation failure -- fall through
     *   >0:                      LAPACK convergence failure -- return
     *   <0 (near -i):            LAPACK argument i illegal -- return, don't swallow
     *
     * K and eigenvectors are both tightly packed (ldk == ldz == N, checked
     * above), so the caller-owned eigenvectors buffer doubles as the
     * in-place vendor workspace.
     */
    if (K != eigenvectors) memcpy(eigenvectors, K, (size_t)N * (size_t)N * sizeof(double));
    int ext_ret = jlinalg_dsyevd_ext(N, eigenvectors, N, eigenvalues);
    if (ext_ret == JLINALG_EXT_SUCCESS) {
        if (status) status->driver_used = JLINALG_DRIVER_DSYEVD;
        return 0;
    }
    if (ext_ret == JLINALG_EXT_ALLOC_FAIL) {
        /* DSYEVD workspace alloc failed -- fall through to DSYEVR.
         * ALLOC_FAIL only occurs before K is touched, so
         * K/eigenvectors are still pristine. */
        if (K != eigenvectors) memcpy(eigenvectors, K, (size_t)N * (size_t)N * sizeof(double));
        fprintf(stderr,
                "jlinalg_eigh_c: vendor dsyevd workspace allocation failed "
                "(N=%ld) -- falling through to dsyevr\n",
                (long)N);
        if (status) status->vendor_lapack_skipped = 1;
    } else if (ext_ret != JLINALG_EXT_UNAVAILABLE) {
        /* Convergence or argument failure -- don't fall through */
        return ext_ret;
    }

    return eigh_dsyevr(N, K, eigenvalues, eigenvectors, status);
}
