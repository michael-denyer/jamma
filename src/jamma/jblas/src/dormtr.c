/**
 * dormtr.c — Blocked Householder back-transformation for jblas.
 *
 * Implements jblas_dormtr_c: applies Q (from dsytrd's Householder vectors
 * stored in the lower triangle of A) to the eigenvector matrix C:
 *   C = Q @ C
 *
 * Algorithm: DLARFT (form triangular T factor) + DLARFB (block application).
 * Processes reflectors in blocks of NB from right to left.
 *
 * DLARFT: Forms upper triangular T[nb x nb] encoding the product
 *   H_j * H_{j+1} * ... * H_{j+nb-1} = I - V * T * V^T
 *
 * DLARFB: Applies (I - V * T * V^T) * C:
 *   1. W = V^T @ C[j+1:N, :]     (nb x M)
 *   2. W = T @ W                   (triangular multiply)
 *   3. C[j+1:N, :] -= V @ W       (rank-nb update via loops)
 *
 * Memory: T[NB x NB] + W[NB x M] + V_block[vlen x NB] + z[NB].
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

#define NB_DORMTR 64

/* dlarft — Form upper triangular T for a block of nb Householder reflectors.
 *
 * V: vlen x nb matrix (row-major, stride nb_alloc). V[:, i] is reflector i.
 * tau: Householder scalars for this block, length nb.
 * T: nb x nb output (row-major, stride T_stride). Upper triangular.
 * z: scratch vector of length nb.
 */
static void dlarft(npy_intp vlen, npy_intp nb,
                    const double *V, npy_intp nb_alloc,
                    const double *tau,
                    double *T, npy_intp T_stride,
                    double *z)
{
    memset(T, 0, (size_t)nb * (size_t)T_stride * sizeof(double));

    for (npy_intp i = 0; i < nb; i++) {
        if (tau[i] == 0.0) {
            T[i * T_stride + i] = 0.0;
            continue;
        }
        T[i * T_stride + i] = tau[i];

        if (i == 0) continue;

        /* z = V[:, 0:i]^T * V[:, i]  (i dot products) */
        for (npy_intp k = 0; k < i; k++) {
            double dot = 0.0;
            for (npy_intp r = 0; r < vlen; r++)
                dot += V[r * nb_alloc + k] * V[r * nb_alloc + i];
            z[k] = dot;
        }

        /* T[0:i, i] = -tau[i] * T[0:i, 0:i] * z
         * T is upper triangular, so T[0:i, 0:i] * z is a triangular matvec. */
        for (npy_intp r = 0; r < i; r++) {
            double s = 0.0;
            for (npy_intp c = r; c < i; c++)  /* upper triangular: c >= r */
                s += T[r * T_stride + c] * z[c];
            T[r * T_stride + i] = -tau[i] * s;
        }
    }
}

int jblas_dormtr_c(npy_intp N, npy_intp M,
                   const double *A, npy_intp lda, const double *tau,
                   double *C, npy_intp ldc)
{
    if (N <= 1 || M <= 0)
        return 0;

    npy_intp nb_alloc = NB_DORMTR;

    /* Allocate workspace */
    double *T_buf = (double *)malloc((size_t)nb_alloc * (size_t)nb_alloc * sizeof(double));
    double *W     = (double *)malloc((size_t)nb_alloc * (size_t)M * sizeof(double));
    npy_intp max_vlen = N - 1;
    double *V_block = (double *)malloc((size_t)max_vlen * (size_t)nb_alloc * sizeof(double));
    double *z     = (double *)malloc((size_t)nb_alloc * sizeof(double));
    if (!T_buf || !W || !V_block || !z) {
        free(T_buf); free(W); free(V_block); free(z);
        return -1;
    }

    npy_intp n_ref = N - 1;  /* total number of reflectors */
    npy_intp j_start = ((n_ref - 1) / NB_DORMTR) * NB_DORMTR;
    for (; j_start >= 0; j_start -= NB_DORMTR) {
        npy_intp nb = n_ref - j_start;
        if (nb > NB_DORMTR) nb = NB_DORMTR;

        npy_intp vlen = N - j_start - 1;

        /* Build V_block[vlen x nb] */
        memset(V_block, 0, (size_t)vlen * (size_t)nb_alloc * sizeof(double));
        for (npy_intp i = 0; i < nb; i++) {
            V_block[i * nb_alloc + i] = 1.0;
            for (npy_intp r = i + 1; r < vlen; r++)
                V_block[r * nb_alloc + i] = A[(j_start + 1 + r) * lda + (j_start + i)];
        }

        /* DLARFT: form T[nb x nb] */
        dlarft(vlen, nb, V_block, nb_alloc, tau + j_start, T_buf, nb_alloc, z);

        /* DLARFB Step 1: W = V^T @ C[j_start+1:N, :] */
        memset(W, 0, (size_t)nb * (size_t)M * sizeof(double));
        for (npy_intp i = 0; i < nb; i++) {
            for (npy_intp r = 0; r < vlen; r++) {
                double v_ri = V_block[r * nb_alloc + i];
                if (v_ri == 0.0) continue;
                const double *C_row = C + (j_start + 1 + r) * ldc;
                double *W_row = W + i * M;
                for (npy_intp c = 0; c < M; c++)
                    W_row[c] += v_ri * C_row[c];
            }
        }

        /* DLARFB Step 2: W = T @ W (upper triangular T, top-to-bottom safe in-place) */
        for (npy_intp i = 0; i < nb; i++) {
            for (npy_intp c = 0; c < M; c++) {
                double s = 0.0;
                for (npy_intp k = i; k < nb; k++)
                    s += T_buf[i * nb_alloc + k] * W[k * M + c];
                W[i * M + c] = s;
            }
        }

        /* DLARFB Step 3: C[j_start+1:N, :] -= V @ W */
        for (npy_intp i = 0; i < nb; i++) {
            const double *W_row = W + i * M;
            for (npy_intp r = 0; r < vlen; r++) {
                double v_ri = V_block[r * nb_alloc + i];
                if (v_ri == 0.0) continue;
                double *C_row = C + (j_start + 1 + r) * ldc;
                for (npy_intp c = 0; c < M; c++)
                    C_row[c] -= v_ri * W_row[c];
            }
        }

        if (j_start == 0) break;
    }

    free(T_buf);
    free(W);
    free(V_block);
    free(z);
    return 0;
}
