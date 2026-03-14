/**
 * eigh.c — Driver for jblas symmetric eigendecomposition.
 *
 * Implements jblas_eigh_c: computes all eigenvalues and eigenvectors of a
 * symmetric N x N matrix K (row-major, lower triangle) using the three-step
 * LAPACK pipeline:
 *
 *   1. jblas_dsytrd_c — Householder tridiagonalization: K -> T (tridiagonal)
 *   2. jblas_dstedc_c — D&C tridiagonal eigensolver: T -> eigenvectors of T
 *   3. jblas_dormtr_c — Back-transformation: eigenvectors of T -> eigenvectors of K
 *
 * On exit:
 *   eigenvalues[k]    = k-th eigenvalue (ascending)
 *   eigenvectors[i,j] = j-th component of k-th eigenvector, column k of Z
 *                       (row-major: eigenvectors[row*ldz + col])
 *
 * Memory:
 *   Workspace: d[N], e[N], tau[N-1] — allocated and freed here.
 *   K is overwritten with the Householder vectors from dsytrd.
 *   dstedc owns its own internal N x N merge buffer (malloc/free inside).
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
#include "jblas.h"

/* ---------------------------------------------------------------------------
 * jblas_eigh_c — Full symmetric eigensolver (public API).
 *
 * Parameters: see jblas.h
 * Returns: 0 on success, -1 on allocation failure, positive i on convergence
 *          failure from dstedc for eigenvalue i.
 * ---------------------------------------------------------------------------
 */
int jblas_eigh_c(npy_intp N,
                 double *K, npy_intp ldk,
                 double *eigenvalues,
                 double *eigenvectors, npy_intp ldz)
{
    if (N <= 0) return 0;

    /* Guard: workspace must be initialized (jblas_init() called) */
    if (!jblas_packed_A) {
        fprintf(stderr,
            "jblas_eigh_c: workspace not allocated "
            "(jblas_dgemm_init() not called or failed)\n");
        return -1;
    }

    if (N == 1) {
        eigenvalues[0]       = K[0];
        eigenvectors[0]      = 1.0;
        return 0;
    }

    /* Step 1: Allocate workspace d[N], e[N], tau[N-1] */
    double *d   = (double *)malloc((size_t)N * sizeof(double));
    double *e   = (double *)malloc((size_t)N * sizeof(double));    /* length N; only [0..N-2] used */
    double *tau = (double *)malloc((size_t)(N - 1) * sizeof(double));

    if (!d || !e || !tau) {
        free(d); free(e); free(tau);
        return -1;
    }

    /* Step 2: Tridiagonalization: K -> T, Householder vectors in K's lower triangle */
    int ret = jblas_dsytrd_c(N, K, ldk, d, e, tau);
    if (ret != 0) {
        free(d); free(e); free(tau);
        return ret;
    }

    /* Step 3: Initialize eigenvectors to identity */
    memset(eigenvectors, 0, (size_t)N * (size_t)ldz * sizeof(double));
    for (npy_intp k = 0; k < N; k++)
        eigenvectors[k * ldz + k] = 1.0;

    /* Step 4: D&C tridiagonal eigensolver
     * On input: d[N] diagonal, e[N-1] off-diagonal, Z = identity.
     * On output: d[N] eigenvalues (ascending), Z columns = eigenvectors of T.
     * dstedc reinitializes Z to identity internally. */
    ret = jblas_dstedc_c(N, d, e, eigenvectors, ldz);
    if (ret != 0) {
        free(d); free(e); free(tau);
        return ret;
    }

    /* Step 5: Back-transformation: eigenvectors of T -> eigenvectors of K
     * C = Q @ C  where Q is encoded in K's lower triangle + tau. */
    ret = jblas_dormtr_c(N, N, K, ldk, tau, eigenvectors, ldz);
    if (ret != 0) {
        free(d); free(e); free(tau);
        return ret;
    }

    /* Step 6: Copy eigenvalues to output */
    memcpy(eigenvalues, d, (size_t)N * sizeof(double));

    /* Step 7: Free workspace (d, e, tau only — dstedc owned its merge buffer) */
    free(d);
    free(e);
    free(tau);

    return 0;
}
