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
 *   Returns positive i on LAPACK convergence failure.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jlinalg.h"

/* ---------------------------------------------------------------------------
 * jlinalg_eigh_c -- Full symmetric eigensolver (public API, vendor-only).
 *
 * Parameters: see jlinalg.h
 * Returns: 0 on success, JLINALG_EXT_UNAVAILABLE if no vendor LAPACK,
 *          JLINALG_EXT_ALLOC_FAIL on allocation failure, positive i on
 *          convergence failure.
 * ---------------------------------------------------------------------------
 */
int jlinalg_eigh_c(npy_intp N, double *K, npy_intp ldk, double *eigenvalues, double *eigenvectors,
                   npy_intp ldz, jlinalg_eigh_status_t *status) {
    if (N <= 0) return 0;

    if (N == 1) {
        eigenvalues[0] = K[0];
        eigenvectors[0] = 1.0;
        return 0;
    }

    /* --- Vendor dsyevd fast path ---
     * When vendor LAPACK dsyevd is available (Accelerate, MKL), call it
     * directly.  Vendor dsyevd is production-hardened, multithreaded, and
     * handles edge cases robustly.
     *
     * jlinalg_dsyevd_ext returns:
     *   0:  success -- K contains eigenvectors (row-major), eigenvalues filled
     *  -2:  no vendor dsyevd available -- fall through to dsyevr
     *  -1:  allocation failure
     *  >0:  LAPACK dsyevd convergence failure
     *
     * When both input and output are tightly packed (ldk == ldz == N), reuse
     * the caller-owned eigenvectors buffer as the in-place vendor workspace.
     * Padded rows still require a tight staging copy.
     */
    if (ldk == N && ldz == N) {
        if (K != eigenvectors) memcpy(eigenvectors, K, (size_t)N * (size_t)N * sizeof(double));
        int ext_ret = jlinalg_dsyevd_ext(N, eigenvectors, N, eigenvalues);
        if (ext_ret == JLINALG_EXT_SUCCESS) return 0;
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
    } else {
        double *K_work = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
        if (K_work) {
            /* Stride-aware copy: K (ldk stride) -> K_work (N stride) */
            if (ldk == N) {
                memcpy(K_work, K, (size_t)N * (size_t)N * sizeof(double));
            } else {
                for (npy_intp i = 0; i < N; i++)
                    memcpy(K_work + i * N, K + i * ldk, (size_t)N * sizeof(double));
            }
            int ext_ret = jlinalg_dsyevd_ext(N, K_work, N, eigenvalues);
            if (ext_ret == JLINALG_EXT_SUCCESS) {
                /* Success: K_work now contains row-major eigenvectors (N stride).
                 * Copy to eigenvectors output buffer (ldz stride). */
                if (ldz == N) {
                    memcpy(eigenvectors, K_work, (size_t)N * (size_t)N * sizeof(double));
                } else {
                    for (npy_intp i = 0; i < N; i++)
                        memcpy(eigenvectors + i * ldz, K_work + i * N, (size_t)N * sizeof(double));
                }
                free(K_work);
                return 0;
            }
            if (ext_ret == JLINALG_EXT_ALLOC_FAIL) {
                /* DSYEVD workspace alloc failed -- fall through to DSYEVR. */
                free(K_work);
                fprintf(stderr,
                        "jlinalg_eigh_c: vendor dsyevd workspace allocation failed "
                        "(N=%ld, padded stride) -- falling through to dsyevr\n",
                        (long)N);
                if (status) status->vendor_lapack_skipped = 1;
            } else if (ext_ret != JLINALG_EXT_UNAVAILABLE) {
                /* Convergence or argument failure -- don't fall through */
                free(K_work);
                return ext_ret;
            } else {
                free(K_work);
                /* ext_ret == -2: no vendor dsyevd -- fall through to dsyevr */
            }
        } else {
            /* K_work malloc failed -- fall through to dsyevr. */
            fprintf(stderr,
                    "jlinalg_eigh_c: vendor dsyevd work copy allocation failed "
                    "(N=%ld, %.1f GB needed) -- trying dsyevr\n",
                    (long)N,
                    (double)((size_t)N * (size_t)N * sizeof(double)) / (1024.0 * 1024 * 1024));
            if (status) status->vendor_lapack_skipped = 1;
        }
    }

    /* --- Vendor dsyevr fast path (memory-pressure fallback) ---
     * DSYEVR uses O(N) workspace (vs O(N^2) for DSYEVD).  Try it when:
     *   - vendor dsyevd was unavailable, OR
     *   - dsyevd workspace allocation failed
     * DSYEVR reads K directly (no work copy needed). */
    if (blas_has_dsyevr()) {
        int evr_ret = jlinalg_dsyevr_ext(N, K, ldk, eigenvalues, eigenvectors, ldz);
        if (evr_ret == JLINALG_EXT_SUCCESS) return 0;
        if (evr_ret == JLINALG_EXT_ALLOC_FAIL) {
            fprintf(stderr,
                    "jlinalg_eigh_c: vendor dsyevr workspace allocation failed "
                    "(N=%ld) -- both DSYEVD and DSYEVR workspace allocs failed\n",
                    (long)N);
            if (status) status->vendor_lapack_skipped = 1;
            return JLINALG_EXT_ALLOC_FAIL;
        }
        if (evr_ret != JLINALG_EXT_UNAVAILABLE) {
            /* Convergence or argument failure -- return error */
            return evr_ret;
        }
    }

    /* No vendor LAPACK available -- caller must use numpy.linalg.eigh */
    return JLINALG_EXT_UNAVAILABLE;
}
