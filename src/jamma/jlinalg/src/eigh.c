/**
 * eigh.c — Driver for jlinalg symmetric eigendecomposition.
 *
 * Implements jlinalg_eigh_c: computes all eigenvalues and eigenvectors of a
 * symmetric N x N matrix K (row-major, lower triangle) using the three-step
 * LAPACK pipeline:
 *
 *   1. jlinalg_dsytrd_c — Householder tridiagonalization: K -> T (tridiagonal)
 *   2. jlinalg_dstedc_c — D&C tridiagonal eigensolver: T -> eigenvectors of T
 *   3. jlinalg_dormtr_c — Back-transformation: eigenvectors of T -> eigenvectors of K
 *
 * On exit:
 *   eigenvalues[k]    = k-th eigenvalue (ascending)
 *   eigenvectors[i,j] = i-th component of j-th eigenvector, column j of Z
 *                       (row-major: eigenvectors[row*ldz + col])
 *
 * Memory:
 *   Workspace: d[N], e[N], tau[N-1] — allocated and freed here.
 *   K is overwritten with the Householder vectors from dsytrd.
 *   dstedc allocates its own internal workspace: 2*N*N (work + merge_scratch)
 *   + O(N) (d_orig, e_orig, iwork) — all malloc'd and freed inside.
 *   eigh does NOT allocate the merge buffer — it belongs to dstedc.
 *
 * Error propagation:
 *   Returns 0 on success.
 *   Returns -1 if workspace allocation fails.
 *   Returns positive i if dstedc failed to converge for eigenvalue i.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jlinalg.h"

/* ---------------------------------------------------------------------------
 * jlinalg_eigh_c — Full symmetric eigensolver (public API).
 *
 * Parameters: see jlinalg.h
 * Returns: 0 on success, -1 on allocation failure, positive i on convergence
 *          failure from dstedc for eigenvalue i.
 * ---------------------------------------------------------------------------
 */
int jlinalg_eigh_c(npy_intp N,
                 double *K, npy_intp ldk,
                 double *eigenvalues,
                 double *eigenvectors, npy_intp ldz,
                 jlinalg_eigh_status_t *status)
{
    if (N <= 0) return 0;

    if (N == 1) {
        eigenvalues[0]       = K[0];
        eigenvectors[0]      = 1.0;
        return 0;
    }

    /* --- Vendor dsyevd fast path ---
     * When vendor LAPACK dsyevd is available (Accelerate, MKL), call it
     * directly instead of the three-step dsytrd+dstedc+dormtr pipeline.
     * Vendor dsyevd is production-hardened, multithreaded, and handles
     * edge cases that our D&C implementation may not.
     *
     * jlinalg_dsyevd_ext returns:
     *   0:  success — K contains eigenvectors (row-major), eigenvalues filled
     *  -2:  no vendor dsyevd available — fall through to jlinalg pipeline
     *  -1:  allocation failure
     *  >0:  LAPACK dsyevd convergence failure
     *
     * When both input and output are tightly packed (ldk == ldz == N), reuse
     * the caller-owned eigenvectors buffer as the in-place vendor workspace.
     * This avoids the extra N×N K_work allocation on the Python path.
     * Padded rows still require a tight staging copy because row-major with
     * ldk/ldz > N cannot be reinterpreted safely as the vendor's packed
     * column-major layout.
     */
    {
        if (ldk == N && ldz == N) {
            if (K != eigenvectors)
                memcpy(eigenvectors, K, (size_t)N * (size_t)N * sizeof(double));
            int ext_ret = jlinalg_dsyevd_ext(N, eigenvectors, N, eigenvalues);
            if (ext_ret == JLINALG_EXT_SUCCESS)
                return 0;
            if (ext_ret == JLINALG_EXT_ALLOC_FAIL) {
                /* DSYEVD workspace alloc failed — fall through to DSYEVR.
                 * ALLOC_FAIL only occurs on the Fortran path before K is
                 * touched (workspace query fails), so K/eigenvectors are
                 * still pristine.  When K == eigenvectors the memcpy is a
                 * no-op (src==dst). */
                if (K != eigenvectors)
                    memcpy(eigenvectors, K, (size_t)N * (size_t)N * sizeof(double));
                fprintf(stderr,
                    "jlinalg_eigh_c: vendor dsyevd workspace allocation failed "
                    "(N=%ld) — falling through to dsyevr\n", (long)N);
                if (status) status->vendor_lapack_skipped = 1;
            } else if (ext_ret != JLINALG_EXT_UNAVAILABLE) {
                /* Convergence or argument failure — don't fall through */
                return ext_ret;
            }
        } else {
            double *K_work = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
            if (K_work) {
                /* Stride-aware copy: K (ldk stride) → K_work (N stride) */
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
                            memcpy(eigenvectors + i * ldz, K_work + i * N,
                                   (size_t)N * sizeof(double));
                    }
                    free(K_work);
                    return 0;
                }
                free(K_work);
                if (ext_ret != JLINALG_EXT_UNAVAILABLE) {
                    /* Vendor dsyevd failed (not "unavailable") — return error.
                     * Don't fall through: if vendor LAPACK fails, our D&C
                     * would likely also fail on the same matrix. */
                    return ext_ret;
                }
                /* ext_ret == -2: no vendor dsyevd — fall through to jlinalg pipeline */
            } else {
                /* K_work malloc failed — fall through to jlinalg pipeline.
                 * NOTE: dstedc needs 2*N*N internally, so it will likely also
                 * fail for truly memory-constrained cases. */
                fprintf(stderr,
                    "jlinalg_eigh_c: vendor dsyevd work copy allocation failed "
                    "(N=%ld, %.1f GB needed) — trying jlinalg pipeline\n",
                    (long)N,
                    (double)((size_t)N * (size_t)N * sizeof(double)) / (1024.0*1024*1024));
                if (status) status->vendor_lapack_skipped = 1;
            }
        }
    }

    /* --- Vendor dsyevr fast path (memory-pressure fallback) ---
     * DSYEVR uses O(N) workspace (vs O(N^2) for DSYEVD).  Try it when:
     *   - vendor dsyevd was unavailable, OR
     *   - dsyevd workspace allocation failed (tight-pack or K_work path)
     * DSYEVR reads K directly (no work copy needed).  When K != eigenvectors,
     * it writes into the caller's eigenvectors buffer; when K == eigenvectors,
     * jlinalg_dsyevr_ext allocates a temporary Z_col internally. */
    if (blas_has_dsyevr()) {
        int evr_ret = jlinalg_dsyevr_ext(N, K, ldk, eigenvalues, eigenvectors, ldz);
        if (evr_ret == JLINALG_EXT_SUCCESS)
            return 0;
        if (evr_ret == JLINALG_EXT_ALLOC_FAIL) {
            /* DSYEVR workspace alloc failed — fall through to jlinalg D&C */
            fprintf(stderr,
                "jlinalg_eigh_c: vendor dsyevr workspace allocation failed "
                "(N=%ld) — trying jlinalg pipeline\n", (long)N);
            if (status) status->vendor_lapack_skipped = 1;
        } else if (evr_ret != JLINALG_EXT_UNAVAILABLE) {
            /* Convergence or argument failure — return error */
            return evr_ret;
        }
    }

    /* --- jlinalg three-step pipeline (fallback) --- */

    /* Guard: K == eigenvectors (inplace mode) is NOT safe for the D&C pipeline.
     * dsytrd writes Householder reflectors into K, then dstedc overwrites
     * eigenvectors (same buffer) with the tridiagonal eigenvectors, destroying
     * the reflectors that dormtr needs for back-transformation.
     * Inplace is only safe on the vendor dsyevd/dsyevr paths above. */
    if (K == eigenvectors) {
        fprintf(stderr,
            "jlinalg_eigh_c: inplace mode (K==eigenvectors) reached D&C pipeline "
            "(vendor DSYEVD and DSYEVR both unavailable or failed, N=%ld). "
            "Inplace is only safe with vendor LAPACK drivers.\n", (long)N);
        return JLINALG_EXT_INPLACE_UNSUPPORTED;
    }

    /* Guard: workspace must be initialized (jlinalg_init() called) */
    if (!jlinalg_packed_A) {
        fprintf(stderr,
            "jlinalg_eigh_c: workspace not allocated "
            "(jlinalg_dgemm_init() not called or failed)\n");
        return -1;
    }

    /* Step 1: Allocate workspace d[N], e[N], tau[N-1] */
    double *d   = (double *)malloc((size_t)N * sizeof(double));
    double *e   = (double *)malloc((size_t)N * sizeof(double));    /* length N; only [0..N-2] used */
    double *tau = (double *)malloc((size_t)(N - 1) * sizeof(double));

    if (!d || !e || !tau) {
        free(d); free(e); free(tau);
        return -1;
    }

    /* Allocate a GEMM workspace shared by dsytrd, dstedc, and dormtr.
     * This avoids mutex serialisation in dsytrd's trailing dsyr2k updates
     * and in dstedc/dormtr merge-level GEMMs.  Thread count matches
     * init-time jlinalg_n_threads.  If allocation fails, pass NULL (all
     * three stages fall back to the global mutex path). */
    jlinalg_workspace_t gemm_ws;
    int ws_ok = jlinalg_workspace_alloc(&gemm_ws, jlinalg_n_threads);
    if (ws_ok != 0) {
        fprintf(stderr, "jlinalg eigh: GEMM workspace allocation failed "
                "(N=%ld, %d threads) — using global mutex path (slower)\n",
                (long)N, jlinalg_n_threads);
        if (status) status->dstedc_ws_fallback = 1;
    }
    jlinalg_workspace_t *ws_ptr = ws_ok == 0 ? &gemm_ws : NULL;

    /* Step 2: Tridiagonalization: K -> T, Householder vectors in K's lower triangle */
    int ret = jlinalg_dsytrd_c(N, K, ldk, d, e, tau, ws_ptr, status);
    if (ret != 0) {
        if (ws_ok == 0) jlinalg_workspace_free(&gemm_ws);
        free(d); free(e); free(tau);
        return ret;
    }

    /* Step 3: D&C tridiagonal eigensolver
     * On input: d[N] diagonal, e[N-1] off-diagonal.
     * On output: d[N] eigenvalues (ascending), Z columns = eigenvectors of T.
     * dstedc initializes Z to identity internally. */
    ret = jlinalg_dstedc_c(N, d, e, eigenvectors, ldz, ws_ptr, status);
    if (ret != 0) {
        if (ws_ok == 0) jlinalg_workspace_free(&gemm_ws);
        free(d); free(e); free(tau);
        return ret;
    }

    /* Step 4: Back-transformation: eigenvectors of T -> eigenvectors of K
     * C = Q @ C  where Q is encoded in K's lower triangle + tau.
     * Reuses the GEMM workspace from dstedc so dormtr routes through external
     * BLAS dispatch (MKL/Accelerate) instead of the global-mutex path. */
    ret = jlinalg_dormtr_c(N, N, K, ldk, tau, eigenvectors, ldz, ws_ptr);
    if (ws_ok == 0) jlinalg_workspace_free(&gemm_ws);
    if (ret != 0) {
        free(d); free(e); free(tau);
        return ret;
    }

    /* Step 5: Copy eigenvalues to output */
    memcpy(eigenvalues, d, (size_t)N * sizeof(double));

    /* Step 6: Free workspace (d, e, tau only — dstedc owned its merge buffer) */
    free(d);
    free(e);
    free(tau);

    return 0;
}
