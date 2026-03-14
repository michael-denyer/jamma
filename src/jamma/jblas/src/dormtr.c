/**
 * dormtr.c — Blocked Householder back-transformation for jblas.
 *
 * Implements jblas_dormtr_c: applies Q (from dsytrd's Householder vectors
 * stored in the lower triangle of A) to the eigenvector matrix C:
 *
 *   C = Q @ C
 *
 * where Q = H_1 * H_2 * ... * H_{N-2} is the accumulated orthogonal
 * transformation from jblas_dsytrd_c.
 *
 * Each Householder reflector H_j = I - tau[j] * v_j * v_j^T where:
 *   v_j[k] = 1       for k = j+1 (implicit, not stored)
 *   v_j[k] = A[(j+1+k)*lda + j]  for k = 1..N-j-2 (stored in lower A)
 *   In 0-based indexing: v stored in column j of A, rows j+1..N-1.
 *
 * Algorithm (column-by-column with dgemm for the rank-1 update):
 *   Apply reflectors right-to-left (H_{N-2} first, H_1 last):
 *   for j = N-2, N-3, ..., 0:
 *     C = H_j @ C = C - tau[j] * v_j @ (v_j^T @ C)
 *
 * The inner product v_j^T @ C (M-vector) is computed via a dgemv-like loop.
 * The outer product update v_j @ w^T is a rank-1 update on a (vlen x M)
 * subblock.  For large M, we delegate to jblas_dgemm_c for cache efficiency.
 *
 * Specifically the update for reflector j is:
 *   Let v = v_j (length vlen = N-j-1), w = v^T @ C (length M).
 *   C_sub = C[j+1:N, :] (vlen x M subblock)
 *   C_sub -= tau * v @ w^T
 *   This is: C_sub -= tau * (v [vlen x 1]) @ (w [1 x M])
 *   Which we compute as: jblas_dgemm_c(vlen, M, 1, v, 1, w, M, C_sub, ldc, 0, 0)
 *   with a -tau scale applied to v beforehand.
 *
 * Row-major layout: A[i,j] = A[i * lda + j], C[i,j] = C[i * ldc + j].
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

/* ---------------------------------------------------------------------------
 * jblas_dormtr_c — Apply Q (from dsytrd) to C on the left: C = Q @ C.
 *
 * Parameters:
 *   N    : matrix dimension (Q is N x N, C is N x M).
 *   M    : number of columns in C (= N for eigenvector back-transformation).
 *   A    : the N x N matrix from dsytrd, row-major, stride lda.
 *          Lower triangle (below diagonal) holds Householder vectors.
 *          A[j+1, j] = e[j] (treated as implicit 1 in the Householder vector).
 *          A[j+2, j], A[j+3, j], ... hold the tail of v_j.
 *   lda  : leading dimension of A (>= N).
 *   tau  : Householder scalars, length N-1.  tau[j] is the scalar for H_j.
 *   C    : N x M matrix to be multiplied by Q (in-place), row-major, stride ldc.
 *   ldc  : leading dimension of C (>= M).
 *
 * Returns 0 on success, -1 on allocation failure.
 * ---------------------------------------------------------------------------
 */
int jblas_dormtr_c(npy_intp N, npy_intp M,
                   const double *A, npy_intp lda, const double *tau,
                   double *C, npy_intp ldc)
{
    if (N <= 1 || M <= 0)
        return 0;

    /* Allocate workspace:
     *   w [M]:    v_j^T @ C (length-M inner product result)
     *   v [N]:    Householder vector (with leading 1) for dgemm call
     */
    double *w = (double *)malloc((size_t)M * sizeof(double));
    double *v = (double *)malloc((size_t)N * sizeof(double));
    if (!w || !v) {
        free(w); free(v);
        return -1;
    }

    /* Apply reflectors in reverse order: j = N-2, N-3, ..., 0.
     *
     * H_j = I - tau[j] * v_j * v_j^T
     *
     * v_j has length vlen = N - j - 1:
     *   v_j[0] = 1   (implicit; actual e value stored at A[(j+1)*lda + j])
     *   v_j[k] = A[(j+1+k)*lda + j]  for k = 1..vlen-1
     *
     * C = H_j @ C means:
     *   w = v_j^T @ C_sub  (C_sub = C[j+1:N, :])
     *   C_sub -= tau[j] * v_j @ w^T
     *
     * Step A: compute w[col] = sum_{row=0}^{vlen-1} v_j[row] * C[j+1+row, col]
     * Step B: C_sub -= tau * v_outer @ w^T  via jblas_dgemm_c
     *   v_outer[k] = -tau[j] * v_j[k]  (negated for the subtraction)
     *   jblas_dgemm_c(vlen, M, 1, v_outer, 1, w, M, C_sub, ldc, 0, 0)
     *   adds v_outer @ w to C_sub, which equals C_sub -= tau * v @ w^T.
     */
    for (npy_intp j = N - 2; j >= 0; j--) {
        if (tau[j] == 0.0) continue;

        npy_intp vlen = N - j - 1;  /* length of v_j */
        double t = tau[j];

        /* Step A: Compute w = v_j^T @ C[j+1:N, :] */
        /* w[col] = sum_{row=0}^{vlen-1} v_j[row] * C[j+1+row, col] */
        memset(w, 0, (size_t)M * sizeof(double));
        for (npy_intp row = 0; row < vlen; row++) {
            double v_row = (row == 0) ? 1.0 : A[(j + 1 + row) * lda + j];
            const double *C_row = C + (j + 1 + row) * ldc;
            for (npy_intp col = 0; col < M; col++)
                w[col] += v_row * C_row[col];
        }

        /* Step B: Build v_outer = -tau * v_j for the dgemm call.
         * dgemm_c adds v_outer @ w^T to C_sub:
         *   C_sub += v_outer @ w^T = C_sub + (-tau*v) @ w^T = C_sub - tau*v@w^T
         */
        for (npy_intp row = 0; row < vlen; row++) {
            double v_row = (row == 0) ? 1.0 : A[(j + 1 + row) * lda + j];
            v[row] = -t * v_row;
        }

        /* C_sub = C + (j+1)*ldc (pointer to C[j+1, 0])
         * jblas_dgemm_c(M_dm, N_dm, K_dm, A, lda, B, ldb, C, ldc, transa, transb)
         * Here: M_dm = vlen, N_dm = M, K_dm = 1
         *       A = v (vlen x 1, lda=1), B = w (1 x M, ldb=M)
         *       C = C_sub (vlen x M, ldc=ldc)
         * Adds A @ B to C_sub: C_sub += v_outer @ w (which is vlen x M)
         */
        double *C_sub = C + (j + 1) * ldc;
        jblas_dgemm_c(vlen, M, (npy_intp)1,
                      v, (npy_intp)1,
                      w, ldc,
                      C_sub, ldc,
                      0, 0);
    }

    free(w);
    free(v);
    return 0;
}
