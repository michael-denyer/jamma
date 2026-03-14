/**
 * dsytrd.c — Householder tridiagonalization for jblas.
 *
 * Implements jblas_dsytrd_c: reduces a symmetric N x N matrix A (stored
 * row-major, lower triangle used) to tridiagonal form T via orthogonal
 * similarity A = Q T Q^T.
 *
 * Algorithm: Unblocked Householder (dsytd2 style).
 *
 * For each column j = 0..N-2:
 *   1. Generate Householder reflector from A[j+1:N, j].
 *   2. Record d[j], e[j], tau[j].
 *   3. Apply symmetric rank-2 update to trailing A[j+1:N, j+1:N].
 *
 * The symmetric rank-2 update at step j:
 *   p = tau * A_trail * v   (A_trail = A[j+1:N, j+1:N] symmetric, m x m)
 *   alpha2 = (tau / 2) * p^T v
 *   w = p - alpha2 * v
 *   A_trail -= v @ w.T + w @ v.T
 *
 * On exit:
 *   d[i]   = diagonal element i  (i = 0..N-1)
 *   e[i]   = off-diagonal element i  (i = 0..N-2)
 *   tau[i] = Householder scalar for reflector i  (i = 0..N-2)
 *   Lower triangle of A holds the Householder vectors (stored as in LAPACK dsytrd).
 *
 * Key conventions:
 *   Row-major layout: A[i,j] = A[i * lda + j].
 *   Householder vector v for column j:
 *     v[0] = 1 (implicit, not stored), stored tail A[(j+1+k)*lda + j] for k>=1.
 *
 * References: LAPACK Working Note 203 (dsytd2 unblocked algorithm).
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

/* ---------------------------------------------------------------------------
 * dlarfg — Generate a Householder reflector.
 *
 * Given alpha (the leading element) and x[1..n-1] (the trailing part),
 * compute tau and overwrite x with the Householder vector v tail such that:
 *   (I - tau * v * v^T) * [alpha; x] = [beta; 0]
 * where v[0] = 1 (not stored) and beta = -sign(alpha) * ||[alpha; x]||_2.
 *
 * If n <= 1 or xnorm == 0, sets tau = 0 (no reflection needed).
 * ---------------------------------------------------------------------------
 */
static void dlarfg(npy_intp n, double *alpha, double *x, npy_intp incx,
                   double *tau)
{
    if (n <= 1) {
        *tau = 0.0;
        return;
    }

    double xnorm = 0.0;
    for (npy_intp i = 0; i < n - 1; i++)
        xnorm += x[i * incx] * x[i * incx];
    xnorm = sqrt(xnorm);

    if (xnorm == 0.0) {
        *tau = 0.0;
        return;
    }

    double a    = *alpha;
    double beta = -((a >= 0.0) ? 1.0 : -1.0) * sqrt(a * a + xnorm * xnorm);
    *tau        = (beta - a) / beta;
    double scale = 1.0 / (a - beta);

    for (npy_intp i = 0; i < n - 1; i++)
        x[i * incx] *= scale;

    *alpha = beta;
}

/* ---------------------------------------------------------------------------
 * jblas_dsytrd_c — Unblocked Householder tridiagonalization (public API).
 *
 * Reduces symmetric N x N matrix A (row-major, lower triangle) to
 * tridiagonal form T = Q^T A Q by applying a sequence of Householder
 * reflectors.  The reflectors are stored in the lower triangle of A.
 *
 * Returns: 0 on success, -1 on allocation failure.
 * ---------------------------------------------------------------------------
 */
int jblas_dsytrd_c(npy_intp N, double *A, npy_intp lda,
                   double *d, double *e, double *tau)
{
    if (N <= 0)
        return 0;

    if (N == 1) {
        d[0] = A[0];
        return 0;
    }

    /* Initialise tau to 0 (guards against uninitialized reads in dormtr) */
    memset(tau, 0, (size_t)(N - 1) * sizeof(double));

    /* Temporary workspace: p[m] and w[m] for the symmetric rank-2 update.
     * Allocate once at max size (N). */
    double *p = (double *)malloc((size_t)N * sizeof(double));
    double *w = (double *)malloc((size_t)N * sizeof(double));
    if (!p || !w) {
        free(p); free(w);
        return -1;
    }

    for (npy_intp j = 0; j < N - 1; j++) {
        /* Diagonal element */
        d[j] = A[j * lda + j];

        /* m = number of rows/cols in trailing submatrix (= length of Householder v) */
        npy_intp m = N - j - 1;

        /* Householder reflector from A[j+1:N, j] (stored in column j, rows j+1..N-1).
         * alpha = A[j+1, j], x_tail = A[j+2:N, j] with stride lda.
         * After dlarfg: A[j+1, j] = beta (= e[j]), A[j+2:N, j] = v_tail. */
        double *alpha_ptr = A + (j + 1) * lda + j;
        double *x_tail    = A + (j + 2) * lda + j;  /* stride lda, length m-1 */
        double alpha_val  = *alpha_ptr;

        dlarfg(m, &alpha_val, x_tail, lda, &tau[j]);

        e[j]       = alpha_val;
        *alpha_ptr = alpha_val;  /* store beta back (= e[j]) */

        if (tau[j] == 0.0)
            continue;

        /* Householder vector v of length m:
         *   v[0] = 1 (implicit)
         *   v[k] = A[(j+1+k)*lda + j]  for k = 1..m-1
         *
         * Compute p = tau * A_trail * v  where A_trail = A[j+1:N, j+1:N] (m x m).
         * Access uses symmetry: A_trail[i,k] = (ri<=ck) ? A[ri,ck] : A[ck,ri].
         */
        double t = tau[j];
        for (npy_intp i = 0; i < m; i++) {
            double s = 0.0;
            for (npy_intp k = 0; k < m; k++) {
                double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
                npy_intp ri = j + 1 + i;
                npy_intp ck = j + 1 + k;
                double A_ik = (ri <= ck) ? A[ri * lda + ck] : A[ck * lda + ri];
                s += A_ik * v_k;
            }
            p[i] = t * s;
        }

        /* alpha2 = (tau / 2) * p^T v */
        double dot = 0.0;
        for (npy_intp k = 0; k < m; k++) {
            double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
            dot += p[k] * v_k;
        }
        double alpha2 = (t / 2.0) * dot;

        /* w = p - alpha2 * v */
        for (npy_intp k = 0; k < m; k++) {
            double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
            w[k] = p[k] - alpha2 * v_k;
        }

        /* Symmetric rank-2 update: A_trail -= v @ w.T + w @ v.T
         * Update only lower triangle (and mirror to upper) for the full m x m block.
         * A[j+1+i, j+1+k] for i >= k (lower triangle of the trailing block). */
        for (npy_intp i = 0; i < m; i++) {
            double v_i = (i == 0) ? 1.0 : A[(j + 1 + i) * lda + j];
            for (npy_intp k = 0; k <= i; k++) {
                double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
                double upd = v_i * w[k] + w[i] * v_k;
                A[(j + 1 + i) * lda + (j + 1 + k)] -= upd;
                if (i != k)
                    A[(j + 1 + k) * lda + (j + 1 + i)] -= upd;
            }
        }
    }

    /* Capture final diagonal element */
    d[N - 1] = A[(N - 1) * lda + (N - 1)];

    free(p);
    free(w);
    return 0;
}
