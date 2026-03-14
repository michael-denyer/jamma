/**
 * dsytrd.c — Blocked Householder tridiagonalization for jblas.
 *
 * Implements jblas_dsytrd_c: reduces a symmetric N x N matrix A (stored
 * row-major, lower triangle used) to tridiagonal form T via orthogonal
 * similarity A = Q T Q^T.
 *
 * Algorithm: Blocked Householder with NB=64.
 *
 * The matrix is processed in blocks of NB columns.  Within each block,
 * the unblocked inner loop (dsytd2_panel) accumulates:
 *   - Householder vectors in V (n_trail x nb), stored also in A's lower triangle
 *   - Corresponding W matrix (n_trail x nb) for the WY representation
 *
 * After the block, the WY update is applied to the trailing submatrix via:
 *   A_trail -= V @ W.T + W @ V.T
 * using jblas_dsyr2k_c.
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
 *     v[0] = 1 (implicit, not stored), v[k] = A[(j+1+k)*lda + j] for k>=1.
 *
 * References: LAPACK Working Note 203 (dsytd2 + dsytrd), LAPACK dsytrd.f.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

/* Block size for the outer blocking loop. */
#define NB 64

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
 * dsytd2_panel — Unblocked reduction of one block panel.
 *
 * Reduces columns [jb..jb+nb-1] of the global matrix (relative pointer A
 * points to A[jb, jb] in the global matrix).
 *
 * For each column j in 0..nb-1:
 *   1. Generate Householder reflector from A[jb+j+1:N, jb+j].
 *   2. Record d[jb+j], e[jb+j], tau[jb+j].
 *   3. Apply the reflector to the remaining trailing submatrix
 *      A[jb+j+1:N, jb+j+1:N] (symmetric rank-2 update).
 *   4. Record the Householder vector column in V[:,j] and
 *      the corresponding W vector in W[:,j].
 *
 * V (n_trail x nb) and W (n_trail x nb) accumulate the block WY factors.
 * These will be used by the caller to apply jblas_dsyr2k_c to the trailing
 * submatrix A[jb+nb:N, jb+nb:N] (if any).
 *
 * Note: The symmetric update applied here only updates the part of the
 * trailing matrix within the current block window (A[jb+j+1:jb+nb, ...]).
 * The update to A[jb+nb:N, jb+nb:N] is deferred to the jblas_dsyr2k_c call.
 *
 * Parameters:
 *   N_full: global matrix size.
 *   nb:     block width.
 *   jb:     column offset of this block.
 *   A:      pointer to A[jb,jb] in the global matrix (row-major, stride lda).
 *   lda:    global leading dimension.
 *   d, e, tau: global output arrays.
 *   V:      n_trail x nb output (row-major, stride ldv). n_trail = N_full-jb-nb.
 *   ldv:    leading dimension of V (>= nb). Ignored if n_trail <= 0.
 *   W:      n_trail x nb output (row-major, stride ldw).
 *   ldw:    leading dimension of W (>= nb). Ignored if n_trail <= 0.
 * ---------------------------------------------------------------------------
 */
static int dsytd2_panel(npy_intp N_full, npy_intp nb, npy_intp jb,
                        double *A, npy_intp lda,
                        double *d, double *e, double *tau,
                        double *V, npy_intp ldv,
                        double *W, npy_intp ldw)
{
    npy_intp n_trail_full = N_full - jb - nb;  /* rows in the trailing submatrix */

    /* Temporary arrays for the symmetric rank-2 update within the panel */
    double *p = (double *)malloc((size_t)(N_full - jb) * sizeof(double));
    double *w_tmp = (double *)malloc((size_t)(N_full - jb) * sizeof(double));
    if (!p || !w_tmp) {
        free(p); free(w_tmp);
        return -1;
    }

    for (npy_intp j = 0; j < nb; j++) {
        npy_intp gj = jb + j;  /* global column index */

        if (gj >= N_full - 1) {
            /* Last column — only capture diagonal, no reflector needed */
            d[gj] = A[j * lda + j];
            if (gj < N_full - 1) tau[gj] = 0.0;
            break;
        }

        /* Diagonal element */
        d[gj] = A[j * lda + j];

        /* Number of rows remaining in the global matrix below row gj */
        npy_intp m = N_full - gj - 1;

        /* Householder reflector from A[gj+1:N_full, gj]
         * (= A[j+1:end, j] relative to this panel pointer A[0,0] = A[jb,jb]).
         * In the panel, A[row, col] at global (jb+row, jb+col) = A_global. */

        /* alpha = A[j+1, j] (relative pointer) */
        double *alpha_ptr = A + (j + 1) * lda + j;
        double *x_tail    = A + (j + 2) * lda + j;  /* A[j+2:, j] with stride lda */
        double alpha_val  = *alpha_ptr;

        dlarfg(m, &alpha_val, x_tail, lda, &tau[gj]);

        e[gj]       = alpha_val;
        *alpha_ptr  = alpha_val;  /* store beta (e[gj]) back */

        if (tau[gj] != 0.0) {
            /* Compute the rank-1 correction for the WY representation.
             * Householder vector v (implicit 1 at position j+1, stored tail in A[j+2:,j]):
             *   v[k]: k=0 -> 1; k>0 -> A[(j+1+k)*lda + j]
             * All indices relative to the panel origin A[0,0] = A_global[jb,jb].
             *
             * Apply H to the trailing symmetric part of A (within the panel):
             * A_sub = A[j+1:nb, j+1:nb] (within block)
             * p = tau * A_sub * v[0..nb-j-2]  (sub-block)
             * then extend p to m elements including the trailing submatrix portion.
             * w_tmp = p - (tau/2 * p^T v) * v
             * A_sub -= v * w_tmp^T + w_tmp * v^T
             */

            npy_intp vsz = m;  /* full reflector length */

            /* Compute p = tau[gj] * A_trail * v (A_trail = lower-right (m x m) block) */
            double t = tau[gj];
            for (npy_intp i = 0; i < vsz; i++) {
                double s = 0.0;
                for (npy_intp k = 0; k < vsz; k++) {
                    double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
                    double A_ik;
                    /* Use symmetry: A_trail[i,k] = A_global[gj+1+i, gj+1+k] */
                    npy_intp ri = j + 1 + i;  /* row in panel */
                    npy_intp ck = j + 1 + k;  /* col in panel */
                    if (ri <= ck)
                        A_ik = A[ri * lda + ck];
                    else
                        A_ik = A[ck * lda + ri];  /* symmetric */
                    s += A_ik * v_k;
                }
                p[i] = t * s;
            }

            /* alpha2 = (tau/2) * p^T v */
            double dot = 0.0;
            for (npy_intp k = 0; k < vsz; k++) {
                double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
                dot += p[k] * v_k;
            }
            double alpha2 = (t / 2.0) * dot;

            /* w_tmp = p - alpha2 * v */
            for (npy_intp k = 0; k < vsz; k++) {
                double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
                w_tmp[k] = p[k] - alpha2 * v_k;
            }

            /* Update only the panel portion (within this block):
             * A[j+1:nb, j+1:nb] -= v * w_tmp^T + w_tmp * v^T
             * Limit update to the current panel (rows/cols up to nb-1 in block).
             */
            npy_intp panel_sz = nb - j - 1;  /* rows within block that remain */
            for (npy_intp i = 0; i < panel_sz; i++) {
                double v_i = (i == 0) ? 1.0 : A[(j + 1 + i) * lda + j];
                for (npy_intp k = 0; k <= i; k++) {
                    double v_k = (k == 0) ? 1.0 : A[(j + 1 + k) * lda + j];
                    double upd = v_i * w_tmp[k] + w_tmp[i] * v_k;
                    A[(j + 1 + i) * lda + (j + 1 + k)] -= upd;
                    if (i != k)
                        A[(j + 1 + k) * lda + (j + 1 + i)] -= upd;
                }
            }

            /* Store V[:,j] and W[:,j] for the deferred trailing update.
             * V[k, j] = v_j[k] for k = 0..n_trail_full-1 (trailing rows only).
             * These correspond to global rows jb+nb .. N_full-1.
             * In the panel pointer, these are rows j+1+panel_sz .. vsz-1,
             * i.e. rows panel_sz .. vsz-1-0 of the v vector.
             * Wait — panel_sz = nb-j-1; the trail rows in v start at panel_sz.
             * v[panel_sz] = A[(j+1+panel_sz)*lda + j] = A[(nb)*lda + j] etc.
             * So v trail = v[panel_sz .. vsz-1] has length vsz-panel_sz = n_trail_full.
             * But n_trail_full is fixed for the block (N_full-jb-nb), while vsz=m=N_full-gj-1
             * varies per j.  We need consistent V shape (n_trail_full x nb).
             *
             * For the first j in the block: vsz = N_full-jb-1, panel_sz = nb-1,
             *   trail length = vsz - panel_sz = N_full-jb-1 - (nb-1) = N_full-jb-nb = n_trail_full. OK.
             * For later j: vsz = N_full-jb-j-1 < N_full-jb-1, panel_sz = nb-j-1,
             *   trail length = vsz - panel_sz = N_full-jb-j-1 - (nb-j-1) = N_full-jb-nb = n_trail_full. OK.
             *
             * So V[k, j] = v_j[panel_sz + k] for k=0..n_trail_full-1.
             * v_j[panel_sz + k] = A[(j+1+panel_sz+k)*lda + j] = A[(nb+k)*lda + j] (k>=0).
             * (for k=0 if panel_sz=nb-j-1: j+1+panel_sz = j+1+nb-j-1 = nb, so row nb in panel.)
             * For k=0: v[0] could be the implicit 1 (only if j+1+panel_sz+0 = j+1+panel_sz = nb)
             *   and panel_sz+0 > 0 (always for j < nb-1), so this is the stored tail element.
             * Special case: the very first element of the trail (panel_sz = 0 only when j=nb-1,
             *   but then panel_sz = 0 and the trail v starts at v[0] = 1 (implicit).
             *   However j=nb-1 is the last column of the block, and panel_sz = nb-(nb-1)-1 = 0,
             *   so the trail starts at v[0] = 1 (the implicit leading 1).
             *
             * Convention: V[0, nb-1] = 1 (implicit), V[k, nb-1] = A[(nb+k)*lda + (nb-1+jb)] for k>=1.
             *   But relative to panel pointer: A[(nb+k)*lda + (nb-1)] etc.
             * For simplicity, store implicit 1 explicitly in V.
             */
            if (n_trail_full > 0 && V && W) {
                for (npy_intp k = 0; k < n_trail_full; k++) {
                    npy_intp v_idx = panel_sz + k;
                    double v_k_val;
                    if (v_idx == 0) {
                        v_k_val = 1.0;  /* implicit leading 1 */
                    } else {
                        /* A[(j+1+v_idx)*lda + j] in panel coords */
                        v_k_val = A[(j + 1 + v_idx) * lda + j];
                    }
                    V[k * ldv + j] = v_k_val;
                    W[k * ldw + j] = w_tmp[panel_sz + k];
                }
            }
        } else {
            /* tau[gj] == 0: no reflector, V and W columns are zero */
            if (n_trail_full > 0 && V && W) {
                for (npy_intp k = 0; k < n_trail_full; k++) {
                    V[k * ldv + j] = 0.0;
                    W[k * ldw + j] = 0.0;
                }
            }
        }
    }

    free(p);
    free(w_tmp);
    return 0;
}

/* ---------------------------------------------------------------------------
 * jblas_dsytrd_c — Blocked Householder tridiagonalization (public API).
 *
 * Reduces symmetric N x N matrix A (row-major, lower triangle) to
 * tridiagonal form T = Q^T A Q by applying a sequence of Householder
 * reflectors.  The reflectors are stored in the lower triangle of A.
 *
 * For N > NB, uses the blocked algorithm: reduce each NB-column panel
 * with dsytd2_panel, then apply the accumulated WY update to the trailing
 * submatrix via jblas_dsyr2k_c.
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

    /* Blocked outer loop: process NB columns at a time */
    for (npy_intp jb = 0; jb < N - 1; jb += NB) {
        npy_intp nb = (NB < N - 1 - jb) ? NB : (N - 1 - jb);
        if (nb <= 0) break;

        npy_intp n_trail = N - jb - nb;  /* size of trailing submatrix */

        double *V = NULL, *W = NULL;
        npy_intp ldv = nb, ldw = nb;

        if (n_trail > 0) {
            /* Allocate V (n_trail x nb) and W (n_trail x nb) for WY update */
            V = (double *)calloc((size_t)n_trail * (size_t)nb, sizeof(double));
            W = (double *)calloc((size_t)n_trail * (size_t)nb, sizeof(double));
            if (!V || !W) {
                free(V); free(W);
                return -1;
            }
        }

        /* Reduce the block panel [jb..jb+nb-1] and accumulate V, W */
        int ret = dsytd2_panel(N, nb, jb,
                               A + jb * lda + jb, lda,
                               d, e, tau,
                               V, ldv, W, ldw);
        if (ret != 0) {
            free(V); free(W);
            return ret;
        }

        /* Apply WY update to trailing submatrix A[jb+nb:N, jb+nb:N]:
         *   A_trail -= V @ W.T + W @ V.T
         * using jblas_dsyr2k_c(n_trail, nb, V, ldv, W, ldw, A_trail, lda).
         * This is the blocked WY update that amortizes panel work over cache. */
        if (n_trail > 0 && V && W) {
            double *A_trail = A + (jb + nb) * lda + (jb + nb);
            jblas_dsyr2k_c(n_trail, nb, V, ldv, W, ldw, A_trail, lda);
        }

        free(V);
        free(W);
    }

    /* Capture final diagonal element */
    d[N - 1] = A[(N - 1) * lda + (N - 1)];

    return 0;
}
