/**
 * dsytrd.c — Blocked Householder tridiagonalization for jblas.
 *
 * Implements jblas_dsytrd_c: reduces a symmetric N x N matrix A (stored
 * row-major, lower triangle used) to tridiagonal form T via orthogonal
 * similarity A = Q T Q^T.
 *
 * Algorithm: Blocked DLATRD panel factorization + dsyr2k trailing update.
 *
 * For each NB-column panel (j = 0 to N-2 step NB):
 *   1. DLATRD: Factor NB columns, producing V[m x nb] and W[m x nb].
 *   2. dsyr2k: A_trail -= V_trail * W_trail^T + W_trail * V_trail^T.
 *   The last panel (or panels smaller than NB) uses unblocked factorization
 *   with no trailing update.
 *
 * DLATRD panel (for each column i within the NB block):
 *   1. dlarfg on A[j+i+1:N, j+i] to get reflector v_i, tau_i.
 *   2. dsymv: p = tau * A_trail * v_i (symmetric matrix-vector product).
 *   3. Correct for previously applied reflectors within this panel:
 *        p -= tau * V[:, 0:i] * (W[:, 0:i]^T * v)
 *        p -= tau * W[:, 0:i] * (V[:, 0:i]^T * v)
 *   4. alpha2 = (tau/2) * dot(p, v).
 *   5. w = p - alpha2 * v.
 *   6. Store v in V[:, i], w in W[:, i].
 *
 * dsymv_lower (static helper):
 *   Computes y = alpha * A * x for symmetric A where only the lower
 *   triangle is reliable (within DLATRD, A hasn't been updated by the
 *   current panel's rank-2k yet). Uses A[max(i,j)*lda + min(i,j)] access.
 *
 * On exit:
 *   d[i]   = diagonal element i  (i = 0..N-1)
 *   e[i]   = off-diagonal element i  (i = 0..N-2)
 *   tau[i] = Householder scalar for reflector i  (i = 0..N-2)
 *   Lower triangle of A holds the Householder vectors (LAPACK dsytrd convention).
 *
 * Memory:
 *   Workspace: V[N x NB] + W[N x NB] = 2*N*NB doubles, allocated once.
 *   Plus p[N] scratch vector.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

#define NB_DSYTRD 64

/* dlarfg — Generate a Householder reflector to zero out x.
 * On exit: alpha = beta (the new diagonal), tau = reflector scalar,
 * x is scaled to form the reflector vector v (with implicit leading 1). */
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

/* DSYMV_MIRROR_THRESHOLD — maximum dimension for mirror+GEMV path.
 * Above this, the temp buffer exceeds ~128MB; fall back to scalar. */
#define DSYMV_MIRROR_THRESHOLD 4096

/* dsymv_lower_scalar — symmetric matrix-vector product (scalar fallback).
 * y = alpha * A * x, where A[n x n] is symmetric, lower triangle stored.
 * x has stride incx, y is contiguous (stride 1). */
static void dsymv_lower_scalar(npy_intp n, double alpha,
                                const double *A, npy_intp lda,
                                const double *x, npy_intp incx,
                                double *y)
{
    for (npy_intp i = 0; i < n; i++) {
        double s = 0.0;
        /* Below and on diagonal: direct access */
        for (npy_intp j = 0; j <= i; j++)
            s += A[i * lda + j] * x[j * incx];
        /* Above diagonal: use symmetry */
        for (npy_intp j = i + 1; j < n; j++)
            s += A[j * lda + i] * x[j * incx];
        y[i] = alpha * s;
    }
}

/* dsymv_lower — symmetric matrix-vector product for lower-triangle storage.
 * y = alpha * A * x, where A[n x n] is symmetric, lower triangle stored.
 * x has stride incx, y is contiguous (stride 1).
 *
 * For n <= DSYMV_MIRROR_THRESHOLD and unit stride x, uses the mirror+GEMV
 * approach: mirrors the lower triangle to the upper triangle in a temp
 * buffer, then calls the ISA-dispatched dgemv for sequential memory access.
 * Falls back to scalar for large n or non-unit stride. */
static void dsymv_lower(npy_intp n, double alpha,
                         const double *A, npy_intp lda,
                         const double *x, npy_intp incx,
                         double *y,
                         double *mirror_buf)
{
    /* Scalar fallback for large n or no mirror buffer */
    if (n > DSYMV_MIRROR_THRESHOLD || mirror_buf == NULL) {
        dsymv_lower_scalar(n, alpha, A, lda, x, incx, y);
        return;
    }

    /* Mirror lower triangle of A into mirror_buf (dense, lda_sym = n).
     * Copy lower triangle and reflect to upper for a fully symmetric buffer. */
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j <= i; j++) {
            double val = A[i * lda + j];
            mirror_buf[i * n + j] = val;
            mirror_buf[j * n + i] = val;
        }
    }

    /* Copy strided x to contiguous buffer (reuse space after mirror_buf).
     * mirror_buf is n*n doubles; x_buf starts at offset n*n. */
    double *x_buf = mirror_buf + n * n;
    for (npy_intp i = 0; i < n; i++)
        x_buf[i] = x[i * incx];

    /* y = A_sym * x via ISA-dispatched dgemv (sequential access pattern) */
    jblas_dispatch.dgemv(n, n, mirror_buf, x_buf, y);

    /* Scale by alpha (dgemv has no alpha parameter) */
    if (alpha != 1.0) {
        for (npy_intp i = 0; i < n; i++)
            y[i] *= alpha;
    }
}

#ifdef JBLAS_DEBUG
/* Known-good unblocked fallback retained for reference/debugging. */
static int dsytrd_unblocked(npy_intp N, double *A, npy_intp lda,
                            double *d, double *e, double *tau)
{
    memset(tau, 0, (size_t)(N - 1) * sizeof(double));

    double *p = (double *)malloc((size_t)N * sizeof(double));
    double *w = (double *)malloc((size_t)N * sizeof(double));
    if (!p || !w) {
        free(p); free(w);
        return -1;
    }

    for (npy_intp j = 0; j < N - 1; j++) {
        d[j] = A[j * lda + j];
        npy_intp m = N - j - 1;

        double alpha_val = A[(j + 1) * lda + j];
        double *x_tail = (m > 1) ? &A[(j + 2) * lda + j] : NULL;
        dlarfg(m, &alpha_val, x_tail, lda, &tau[j]);
        e[j] = alpha_val;
        A[(j + 1) * lda + j] = alpha_val;

        if (tau[j] == 0.0)
            continue;

        double saved_e = A[(j + 1) * lda + j];
        A[(j + 1) * lda + j] = 1.0;
        dsymv_lower(m, tau[j],
                    A + (j + 1) * lda + (j + 1), lda,
                    A + (j + 1) * lda + j, lda,
                    p, NULL);
        A[(j + 1) * lda + j] = saved_e;

        double dot_pv = 0.0;
        dot_pv += p[0];
        for (npy_intp k = 1; k < m; k++)
            dot_pv += p[k] * A[(j + 1 + k) * lda + j];
        double alpha2 = (tau[j] / 2.0) * dot_pv;

        w[0] = p[0] - alpha2;
        for (npy_intp k = 1; k < m; k++)
            w[k] = p[k] - alpha2 * A[(j + 1 + k) * lda + j];

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

    d[N - 1] = A[(N - 1) * lda + (N - 1)];
    free(p);
    free(w);
    return 0;
}
#endif /* JBLAS_DEBUG */

/* dlatrd_panel — Factor nb columns of the symmetric matrix, producing V and W.
 *
 * A:   N x N symmetric matrix (row-major, lower triangle), modified in place.
 * lda: Leading dimension of A.
 * N:   Full matrix dimension.
 * j:   Starting column index in A.
 * nb:  Number of columns to factor.
 * d:   Diagonal output (d[j..j+nb-1] written).
 * e:   Off-diagonal output (e[j..j+nb-1] written).
 * tau: Householder scalars (tau[j..j+nb-1] written).
 * V:   m_panel x nb_alloc workspace (row-major). V[:, i] = i-th Householder vector.
 * W:   m_panel x nb_alloc workspace (row-major). W[:, i] = i-th update vector.
 * nb_alloc: Leading dimension of V and W (>= nb).
 * p:   Scratch vector of length m_panel.
 */
static void dlatrd_panel(double *A, npy_intp lda, npy_intp N,
                          npy_intp j, npy_intp nb,
                          double *d, double *e, double *tau,
                          double *V, double *W, npy_intp nb_alloc,
                          double *p, double *mirror_buf)
{
    npy_intp m_panel = N - j - 1;  /* total trailing rows for this panel */

    for (npy_intp i = 0; i < nb; i++) {
        npy_intp col = j + i;          /* absolute column index */
        npy_intp m   = N - col - 1;    /* trailing size for this column */
        npy_intp off = m_panel - m;    /* offset into V/W for this column's v */

        /* KNOWN BOTTLENECK: The deferred rank-2 update loop below is O(nb*m) per
         * column — a scalar bottleneck. Converting this to BLAS would require
         * restructuring dlatrd_panel to use a full DSYR2K trailing update rather
         * than column-by-column deferred application. Left as scalar for Phase 81+
         * consideration; the primary dsytrd bottleneck (dsymv_lower) is already
         * BLAS-backed via mirror+GEMV above. */

        /* Apply deferred rank-2 updates to column col before reading it.
         * The trailing update A -= V*W^T + W*V^T has not been applied yet;
         * we must update A[col, col] and A[col+1:N, col] using previous
         * V/W columns so that dlarfg and d[col] see correct values. */
        if (i > 0) {
            /* V/W row for A[col, col] is (i - 1).
             * V/W rows for A[col+1:N, col] are i..m_panel-1. */

            /* Update diagonal: A[col,col] -= 2 * sum_{prev} V_diag * W_diag */
            npy_intp diag_vw = i - 1;  /* V/W row index for diagonal */
            for (npy_intp prev = 0; prev < i; prev++) {
                double v_d = V[diag_vw * nb_alloc + prev];
                double w_d = W[diag_vw * nb_alloc + prev];
                A[col * lda + col] -= 2.0 * v_d * w_d;
            }

            /* Update sub-diagonal: A[col+1+k, col] -= V[off+k,prev]*W[diag,prev]
             *                                       + W[off+k,prev]*V[diag,prev]
             * off = i, so V/W rows off..m_panel-1 = rows for A[col+1..N-1]. */
            for (npy_intp prev = 0; prev < i; prev++) {
                double w_d = W[diag_vw * nb_alloc + prev];
                double v_d = V[diag_vw * nb_alloc + prev];
                for (npy_intp k = 0; k < m; k++) {
                    npy_intp vw_row = off + k;  /* = i + k */
                    A[(col + 1 + k) * lda + col] -=
                        V[vw_row * nb_alloc + prev] * w_d
                      + W[vw_row * nb_alloc + prev] * v_d;
                }
            }
        }

        /* Record diagonal (now updated for deferred rank-2) */
        d[col] = A[col * lda + col];

        if (m <= 0) {
            tau[col] = 0.0;
            continue;
        }

        /* Generate Householder reflector from A[col+1:N, col] (now updated) */
        double alpha_val = A[(col + 1) * lda + col];
        double *x_tail = (m > 1) ? &A[(col + 2) * lda + col] : NULL;
        dlarfg(m, &alpha_val, x_tail, lda, &tau[col]);
        e[col] = alpha_val;
        A[(col + 1) * lda + col] = alpha_val;

        if (tau[col] == 0.0) {
            /* No reflection — zero out V and W columns */
            for (npy_intp k = 0; k < m_panel; k++) {
                V[k * nb_alloc + i] = 0.0;
                W[k * nb_alloc + i] = 0.0;
            }
            continue;
        }

        /* Copy v into V[:, i] — rows 0..off-1 are zero, then v[0]=1, v[1..] from A */
        for (npy_intp k = 0; k < off; k++)
            V[k * nb_alloc + i] = 0.0;
        V[off * nb_alloc + i] = 1.0;
        for (npy_intp k = 1; k < m; k++)
            V[(off + k) * nb_alloc + i] = A[(col + 1 + k) * lda + col];

        /* Temporarily set A[col+1, col] = 1.0 for dsymv */
        double saved_e = A[(col + 1) * lda + col];
        A[(col + 1) * lda + col] = 1.0;

        /* p = tau * A_trail * v (using symmetric lower-triangle access)
         * A_trail = A[col+1:N, col+1:N], size m x m
         * v = A[col+1:N, col], stride lda */
        dsymv_lower(m, tau[col],
                     A + (col + 1) * lda + (col + 1), lda,
                     A + (col + 1) * lda + col, lda,
                     p + off,
                     mirror_buf);

        /* Restore A[col+1, col] */
        A[(col + 1) * lda + col] = saved_e;

        /* Zero the leading part of p */
        for (npy_intp k = 0; k < off; k++)
            p[k] = 0.0;

        /* Correct for previously applied reflectors (i > 0):
         * p -= tau * V[:, 0:i] * (W[:, 0:i]^T * v) + tau * W[:, 0:i] * (V[:, 0:i]^T * v) */
        if (i > 0) {
            double t = tau[col];
            for (npy_intp prev = 0; prev < i; prev++) {
                double d1 = 0.0, d2 = 0.0;
                for (npy_intp k = 0; k < m_panel; k++) {
                    d1 += W[k * nb_alloc + prev] * V[k * nb_alloc + i];
                    d2 += V[k * nb_alloc + prev] * V[k * nb_alloc + i];
                }
                for (npy_intp k = 0; k < m_panel; k++) {
                    p[k] -= t * (d1 * V[k * nb_alloc + prev]
                               + d2 * W[k * nb_alloc + prev]);
                }
            }
        }

        /* alpha2 = (tau/2) * dot(p, v) */
        double dot_pv = 0.0;
        for (npy_intp k = 0; k < m_panel; k++)
            dot_pv += p[k] * V[k * nb_alloc + i];
        double alpha2 = (tau[col] / 2.0) * dot_pv;

        /* w = p - alpha2 * v */
        for (npy_intp k = 0; k < m_panel; k++)
            W[k * nb_alloc + i] = p[k] - alpha2 * V[k * nb_alloc + i];
    }
}

/* jblas_dsytrd_c — Blocked Householder tridiagonalization (public API).
 *
 * Reduces symmetric N x N matrix A to tridiagonal form.
 * A is row-major, lower triangle used.  On exit, d/e/tau hold the
 * tridiagonal form and Householder scalars; the lower triangle of A
 * holds the Householder vectors.
 *
 * Returns 0 on success, -1 on allocation failure. */
int jblas_dsytrd_c(npy_intp N, double *A, npy_intp lda,
                   double *d, double *e, double *tau,
                   jblas_eigh_status_t *status)
{
    if (N <= 0) return 0;
    if (N == 1) { d[0] = A[0]; return 0; }

    memset(tau, 0, (size_t)(N - 1) * sizeof(double));

    npy_intp m_panel = N - 1;
    npy_intp nb_alloc = NB_DSYTRD;

    /* Allocate V[m_panel x nb_alloc], W[m_panel x nb_alloc], p[m_panel] */
    double *V = (double *)calloc((size_t)m_panel * (size_t)nb_alloc, sizeof(double));
    double *W = (double *)calloc((size_t)m_panel * (size_t)nb_alloc, sizeof(double));
    double *p = (double *)malloc((size_t)m_panel * sizeof(double));

    /* Mirror buffer for GEMV-backed dsymv_lower: n*n (mirror) + n (x_buf).
     * Allocated once and reused for all dsymv_lower calls within the panel.
     * NULL if m_panel exceeds the mirror threshold (scalar fallback). */
    double *mirror_buf = NULL;
    if (m_panel <= DSYMV_MIRROR_THRESHOLD) {
        mirror_buf = (double *)malloc(
            ((size_t)m_panel * (size_t)m_panel + (size_t)m_panel) * sizeof(double));
        if (!mirror_buf) {
            fprintf(stderr, "jblas dsytrd: mirror buffer allocation failed "
                    "(N=%ld, %zu bytes) — falling back to scalar dsymv\n",
                    (long)N,
                    ((size_t)m_panel * (size_t)m_panel + (size_t)m_panel) * sizeof(double));
            if (status) status->dsytrd_mirror_fallback = 1;
        }
    }

    if (!V || !W || !p) {
        free(V); free(W); free(p); free(mirror_buf);
        return -1;
    }

    for (npy_intp j = 0; j < N - 1; j += NB_DSYTRD) {
        npy_intp nb = (N - 1 - j < NB_DSYTRD) ? (N - 1 - j) : NB_DSYTRD;

        /* Zero V and W for this panel */
        memset(V, 0, (size_t)m_panel * (size_t)nb_alloc * sizeof(double));
        memset(W, 0, (size_t)m_panel * (size_t)nb_alloc * sizeof(double));

        /* DLATRD: factor nb columns */
        dlatrd_panel(A, lda, N, j, nb, d, e, tau, V, W, nb_alloc, p, mirror_buf);

        /* Trailing dsyr2k update on the unreduced block A[j+nb:N, j+nb:N].
         *
         * The panel rows V/W[nb-1, :] correspond to matrix row/col j+nb,
         * which is the first row/col of the trailing unreduced submatrix.
         * Using nb here would skip that row/col and leave the next panel stale. */
        npy_intp m_trail = N - j - nb;
        if (m_trail > 0) {
            jblas_dsyr2k_c(m_trail, nb,
                           V + (nb - 1) * nb_alloc, nb_alloc,
                           W + (nb - 1) * nb_alloc, nb_alloc,
                           A + (j + nb) * lda + (j + nb), lda);
        }
    }

    /* The final 1x1 trailing block is updated by the last dsyr2k call. */
    d[N - 1] = A[(N - 1) * lda + (N - 1)];

    free(V);
    free(W);
    free(p);
    free(mirror_buf);
    return 0;
}
