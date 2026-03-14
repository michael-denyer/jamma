/**
 * dormtr.c — Householder back-transformation for jblas.
 *
 * Implements jblas_dormtr_c: applies Q (from dsytrd's Householder vectors
 * stored in the lower triangle of A) to the eigenvector matrix C:
 *
 *   C = Q @ C
 *
 * where Q = H_0 * H_1 * ... * H_{N-3} is the accumulated orthogonal
 * transformation from jblas_dsytrd_c.
 *
 * Each Householder reflector H_j = I - tau[j] * v_j * v_j^T where:
 *   v_j[k] = 1       for k = 0 (implicit, not stored)
 *   v_j[k] = A[(j+1+k)*lda + j]  for k = 1..N-j-2 (stored in lower A)
 *   In 0-based indexing: v stored in column j of A, rows j+1..N-1.
 *
 * Algorithm:
 *   Apply reflectors right-to-left (H_{N-2} first, H_0 last):
 *   for j = N-2, N-3, ..., 0:
 *     w = v_j^T @ C[j+1:N, :]          (length-M inner product)
 *     C[j+1:N, :] -= tau[j] * v_j @ w^T  (rank-1 update)
 *
 * Note: jblas_dgemm_c always zeroes C before accumulating (C = A@B semantics,
 * not C += A@B), so the rank-1 update uses a direct loop rather than dgemm.
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
 *          A[j+1, j] = e[j] (implicit 1 in v_j).
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

    /* Workspace: w[M] = v_j^T @ C[j+1:N, :] (length-M inner product). */
    double *w = (double *)malloc((size_t)M * sizeof(double));
    if (!w)
        return -1;

    /* Apply reflectors in reverse order: j = N-2, N-3, ..., 0.
     *
     * H_j = I - tau[j] * v_j * v_j^T
     *
     * v_j has length vlen = N - j - 1:
     *   v_j[0] = 1   (implicit; A[(j+1)*lda + j] stores e[j], used as v_j[0]=1)
     *   v_j[k] = A[(j+1+k)*lda + j]  for k = 1..vlen-1
     *
     * C = H_j @ C means:
     *   w = v_j^T @ C[j+1:N, :]
     *   C[j+1:N, :] -= tau[j] * v_j @ w^T
     */
    for (npy_intp j = N - 2; j >= 0; j--) {
        if (tau[j] == 0.0) continue;

        npy_intp vlen = N - j - 1;  /* length of v_j */
        double t = tau[j];

        /* Step A: w[col] = sum_{row=0}^{vlen-1} v_j[row] * C[j+1+row, col] */
        memset(w, 0, (size_t)M * sizeof(double));
        for (npy_intp row = 0; row < vlen; row++) {
            double v_row = (row == 0) ? 1.0 : A[(j + 1 + row) * lda + j];
            const double *C_row = C + (j + 1 + row) * ldc;
            for (npy_intp col = 0; col < M; col++)
                w[col] += v_row * C_row[col];
        }

        /* Step B: C[j+1+row, col] -= t * v_j[row] * w[col] */
        for (npy_intp row = 0; row < vlen; row++) {
            double v_row = (row == 0) ? 1.0 : A[(j + 1 + row) * lda + j];
            double scale = t * v_row;
            double *C_row = C + (j + 1 + row) * ldc;
            for (npy_intp col = 0; col < M; col++)
                C_row[col] -= scale * w[col];
        }
    }

    free(w);
    return 0;
}
