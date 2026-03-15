/**
 * dstedc.c — Divide-and-conquer tridiagonal eigensolver for jblas.
 *
 * Implements jblas_dstedc_c: computes all eigenvalues and eigenvectors of a
 * real symmetric tridiagonal matrix T given by diagonal d[N] and off-diagonal
 * e[N-1].  Z is caller-allocated N x N (jblas_dstedc_c initializes it to
 * identity internally).  On output d contains the eigenvalues
 * in ascending order and Z contains the corresponding eigenvectors as columns
 * (row-major: Z[i,j] is the i-th component of eigenvector j; equivalently
 * eigenvector for eigenvalue d[k] is column k of Z — the caller convention
 * matches LAPACK dstedc with COMPZ='I').
 *
 * NOTE ON ROW-MAJOR CONVENTION:
 *   jblas uses row-major layout throughout.  Here Z[N x N] row-major means
 *   Z[i*ldz + j] is element (i, j).  The eigenvectors are stored such that
 *   Z[:,k] (i.e. column k) is the k-th eigenvector.  In row-major, the k-th
 *   column is Z[0*ldz+k], Z[1*ldz+k], ... Z[(N-1)*ldz+k].
 *
 * Algorithm:
 *   Base case (N <= DSTEDC_BASE, currently 128): Implicit QR iteration (Francis shift) on the
 *   tridiagonal matrix.  Eigenvectors are accumulated via Givens rotations.
 *
 *   Recursive case (N > DSTEDC_BASE): Divide-and-conquer a la Cuppen (1981).
 *     1. Split at m = N/2; adjust d[m-1] -= |rho|, d[m] -= |rho|.
 *     2. Recurse on left half [0..m-1] and right half [m..N-1].
 *     3. Merge via rank-1 secular equation:
 *          f(lambda) = 1 + rho * sum_k z_k^2 / (d_k - lambda) = 0
 *        where z is the connecting rank-1 vector.
 *     4. Back-transform eigenvectors: Z = Z_halves @ Q_secular.
 *        Uses jblas_dgemm_c for the N x N matrix multiply.
 *
 * Deflation (LAPACK DLAED2-style local relative threshold):
 *   (a) rho*z[k]^2 <= 8*eps*max(|d[k]|, rho*z[k]^2): negligible contribution.
 *   (b) |d[i] - d[j]| <= 8*eps*max(|d[i]|, |d[j]|): merged via Givens rotation.
 *
 * Secular equation solver (dlaed4-like):
 *   Newton iteration with rational interpolation, guaranteed monotone
 *   convergence between poles.  Tolerance: 4*eps*|lambda|.
 *
 * Memory:
 *   dstedc_c allocates one N x N workspace + O(N) scratch at top level,
 *   passed through recursion. merge_rank1 uses the workspace for Q_sec
 *   (N x N) and additionally allocates Z_new (N x N), delta_mat (up to
 *   N_nd x N_nd), Q_nd (N_nd x N_nd), and Q_nd_full (N x N_nd) locally.
 *   Peak merge-step memory is ~5 * N^2 doubles (~40 bytes/element).
 *   For N=100k, expect ~400 GB peak during the top-level merge.
 *
 * References:
 *   Cuppen (1981), "A Divide and Conquer Method for the Symmetric
 *   Tridiagonal Eigenproblem."
 *   Li (1994), "Solving Secular Equations Stably and Efficiently."
 *   LAPACK Working Note 89.
 */

/* Hint to conforming compilers that FENV access is required.
 * Note: GCC/Clang largely ignore this pragma when -ffast-math is set.
 * The primary defense against -ffast-math is the lapack_sources compile
 * group in hatch_build.py (verified by test_lapack_no_ffast_math). */
#pragma STDC FENV_ACCESS ON

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))

/* Threshold for switching to base-case QR iteration.
 * LAPACK uses SMLSIZ ~25; 128 chosen for larger QR batches to reduce
 * D&C merge overhead. */
#define DSTEDC_BASE 128

/* Machine epsilon */
#define EPS DBL_EPSILON

/* ---------------------------------------------------------------------------
 * Givens rotation helpers
 * ---------------------------------------------------------------------------
 */

/** Compute Givens rotation (c, s) such that [c s; -s c] * [a; b] = [r; 0].
 *
 * That is: c*a + s*b = r  and  -s*a + c*b = 0,  with c²+s²=1.
 *
 * Derivation:
 *   From -s*a + c*b = 0: s/c = b/a  (for |a| >= |b|, use c as pivot)
 *                         c/s = a/b  (for |b| > |a|, use s as pivot)
 *
 * Sign convention: r >= 0 always (enforced by sign flip). Note: LAPACK
 * dlartg uses r = sigma * hypot(a,b) where sigma depends on the larger
 * component; this implementation differs by guaranteeing non-negative r.
 */
static void dlartg(double a, double b, double *c, double *s, double *r)
{
    if (b == 0.0) {
        *c = (a >= 0.0) ? 1.0 : -1.0;
        *s = 0.0;
        *r = fabs(a);
        return;
    }
    if (a == 0.0) {
        *c = 0.0;
        *s = (b >= 0.0) ? 1.0 : -1.0;
        *r = fabs(b);
        return;
    }
    if (fabs(b) > fabs(a)) {
        /* |b| > |a|: pivot on s.
         * t = a/b, s = 1/√(1+t²), c = s*t. */
        double t = a / b;
        *s = 1.0 / sqrt(1.0 + t * t);
        *c = (*s) * t;
    } else {
        /* |a| >= |b|: pivot on c.
         * t = b/a, c = 1/√(1+t²), s = c*t. */
        double t = b / a;
        *c = 1.0 / sqrt(1.0 + t * t);
        *s = (*c) * t;
    }
    /* r = c*a + s*b = hypot(a,b) in sign; enforce r >= 0. */
    *r = (*c) * a + (*s) * b;
    if (*r < 0.0) { *c = -(*c); *s = -(*s); *r = -(*r); }
}

/** Apply Givens rotation G to two rows i and j of Z (in-place).
 *  G: [c s; -s c] applied on the left.  Updates Z[i,:] and Z[j,:]. */
static void apply_givens_row(double *Z, npy_intp ldz, npy_intp n,
                              npy_intp i, npy_intp j, double c, double s)
{
    for (npy_intp k = 0; k < n; k++) {
        double zi = Z[i * ldz + k];
        double zj = Z[j * ldz + k];
        Z[i * ldz + k] =  c * zi + s * zj;
        Z[j * ldz + k] = -s * zi + c * zj;
    }
}

/** Accumulate Givens rotation G = [[c,s],[-s,c]] into eigenvector matrix Z:
 *    Z_new = Z_old @ G
 *  i.e. right-multiply Z by G, updating columns i and j.
 *  G[:,0] = [c,-s]: Z_new[:,i] = c*Z[:,i] - s*Z[:,j]
 *  G[:,1] = [s, c]: Z_new[:,j] = s*Z[:,i] + c*Z[:,j]
 */
static void apply_givens_col(double *Z, npy_intp ldz, npy_intp n,
                              npy_intp i, npy_intp j, double c, double s)
{
    for (npy_intp k = 0; k < n; k++) {
        double zi = Z[k * ldz + i];
        double zj = Z[k * ldz + j];
        Z[k * ldz + i] =  c * zi - s * zj;
        Z[k * ldz + j] =  s * zi + c * zj;
    }
}

/* ---------------------------------------------------------------------------
 * Eigenvalue/eigenvector sorting
 * ---------------------------------------------------------------------------
 */

/** Insertion sort for small N (base case, N <= DSTEDC_BASE).
 *  O(N^2) comparisons with O(N) column swaps per inversion — fine for N <= 128. */
static void sort_eig_insertion(double *d, double *Z, npy_intp ldz, npy_intp n)
{
    for (npy_intp i = 1; i < n; i++) {
        double key = d[i];
        npy_intp j = i - 1;
        while (j >= 0 && d[j] > key) {
            double tmp = d[j];
            d[j] = d[j + 1];
            d[j + 1] = tmp;
            for (npy_intp k = 0; k < n; k++) {
                double tz = Z[k * ldz + j];
                Z[k * ldz + j] = Z[k * ldz + j + 1];
                Z[k * ldz + j + 1] = tz;
            }
            j--;
        }
        d[j + 1] = key;
    }
}

/** Index-based sort for large N: argsort + single-pass permutation copy.
 *  O(N log N) sort + O(N^2) permutation copy — avoids O(N^3) of insertion sort.
 *  Falls back to insertion sort if index allocation fails. */
static void sort_eig(double *d, double *Z, npy_intp ldz, npy_intp n)
{
    if (n <= DSTEDC_BASE) {
        sort_eig_insertion(d, Z, ldz, n);
        return;
    }

    /* Argsort: build index array, sort by eigenvalue */
    npy_intp *idx = (npy_intp *)malloc((size_t)n * sizeof(npy_intp));
    double *d_tmp = (double *)malloc((size_t)n * sizeof(double));
    double *z_col = (double *)malloc((size_t)n * sizeof(double));
    if (!idx || !d_tmp || !z_col) {
        free(idx); free(d_tmp); free(z_col);
        sort_eig_insertion(d, Z, ldz, n);  /* fallback */
        return;
    }

    for (npy_intp i = 0; i < n; i++) idx[i] = i;

    /* Shell sort on idx by d[idx[i]] — O(N^(4/3)) worst case, no recursion. */
    npy_intp gap = 1;
    while (gap < n / 3) gap = gap * 3 + 1;
    for (; gap > 0; gap /= 3) {
        for (npy_intp i = gap; i < n; i++) {
            npy_intp tmp_idx = idx[i];
            double tmp_val = d[tmp_idx];
            npy_intp j = i;
            while (j >= gap && d[idx[j - gap]] > tmp_val) {
                idx[j] = idx[j - gap];
                j -= gap;
            }
            idx[j] = tmp_idx;
        }
    }

    /* Permute eigenvalues */
    for (npy_intp i = 0; i < n; i++) d_tmp[i] = d[idx[i]];
    memcpy(d, d_tmp, (size_t)n * sizeof(double));

    /* Permute eigenvector columns via cycle-following (O(N^2) total, O(N) extra memory).
     * For each cycle j → idx[j] → idx[idx[j]] → ... → j, we want:
     *   result[:,j] = original[:,idx[j]] for all j in the cycle.
     * Save original[:,j], shift columns backward along the cycle, then place saved. */
    for (npy_intp j = 0; j < n; j++) {
        if (idx[j] == j) continue;
        /* Save column at cycle start position j */
        for (npy_intp k = 0; k < n; k++)
            z_col[k] = Z[k * ldz + j];
        /* Follow the cycle: each position gets the column from its idx */
        npy_intp dst = j;
        npy_intp src = idx[j];
        while (src != j) {
            for (npy_intp k = 0; k < n; k++)
                Z[k * ldz + dst] = Z[k * ldz + src];
            idx[dst] = dst;
            dst = src;
            src = idx[src];
        }
        /* Last position in cycle gets the saved start column */
        for (npy_intp k = 0; k < n; k++)
            Z[k * ldz + dst] = z_col[k];
        idx[dst] = dst;
    }

    free(idx);
    free(d_tmp);
    free(z_col);
}

/* Relative Frobenius residual for a tridiagonal eigensystem:
 *   ||T * Z - Z * diag(d_out)||_F / ||T||_F
 * where T is defined by d_in/e_in and Z stores eigenvectors as columns. */
static double tridiag_eig_residual(npy_intp n,
                                   const double *d_in,
                                   const double *e_in,
                                   const double *d_out,
                                   const double *Z, npy_intp ldz)
{
    double normT2 = 0.0;
    for (npy_intp i = 0; i < n; i++)
        normT2 += d_in[i] * d_in[i];
    for (npy_intp i = 0; i < n - 1; i++)
        normT2 += 2.0 * e_in[i] * e_in[i];

    if (normT2 == 0.0)
        return 0.0;

    double resid2 = 0.0;
    for (npy_intp j = 0; j < n; j++) {
        double lam = d_out[j];
        for (npy_intp i = 0; i < n; i++) {
            double tz = d_in[i] * Z[i * ldz + j];
            if (i > 0)
                tz += e_in[i - 1] * Z[(i - 1) * ldz + j];
            if (i + 1 < n)
                tz += e_in[i] * Z[(i + 1) * ldz + j];
            double r = tz - lam * Z[i * ldz + j];
            resid2 += r * r;
        }
    }

    return sqrt(resid2 / normT2);
}

/* ---------------------------------------------------------------------------
 * Base case: implicit symmetric QR iteration with Wilkinson shift.
 *
 * Computes all eigenvalues and eigenvectors of the n x n symmetric
 * tridiagonal matrix defined by diag=d[0..n-1] and off-diag=e[0..n-2].
 * Z (n x n, row-major with stride ldz) accumulates Givens rotations.
 * On input Z should be identity; on output Z columns are eigenvectors.
 *
 * Algorithm follows LAPACK dsteqr (COMPZ='V') with Wilkinson shift.
 *
 * Returns 0 on success, positive n if iteration failed to converge.
 * ---------------------------------------------------------------------------
 */
static int dsteqr_base(npy_intp n, double *d, double *e,
                       double *Z, npy_intp ldz)
{
    if (n <= 1)
        return 0;

    /* Work on local copies to avoid aliasing issues with recursion */
    double *diag = (double *)malloc((size_t)n * sizeof(double));
    double *offd = (double *)malloc((size_t)(n - 1) * sizeof(double));
    if (!diag || !offd) {
        free(diag); free(offd);
        return -1;
    }
    memcpy(diag, d, (size_t)n * sizeof(double));
    memcpy(offd, e, (size_t)(n - 1) * sizeof(double));

    npy_intp max_iter = 30 * n;
    npy_intp l1 = 0;
    int converged = 0;

    for (npy_intp iter = 0; iter < max_iter; iter++) {
        if (l1 >= n - 1) {
            converged = 1;
            break;
        }

        /* Find the bottom of the unreduced submatrix: scan downward from l1
         * to find a tiny off-diagonal entry that splits the problem. */
        npy_intp l2 = l1;
        while (l2 < n - 1) {
            double eps1 = EPS * (fabs(diag[l2]) + fabs(diag[l2 + 1]));
            if (fabs(offd[l2]) <= eps1) {
                offd[l2] = 0.0;
                break;
            }
            l2++;
        }
        /* l2 is now either: the first zero off-diagonal, or n-1 (all connected) */

        if (l2 == l1) {
            /* Single element — deflated */
            l1++;
            continue;
        }

        /* Unreduced block is diag[l1..l2], offd[l1..l2-1] */

        /* Wilkinson shift from the 2x2 bottom of the block */
        double b  = (diag[l2] - diag[l2 - 1]) / 2.0;
        double e2 = offd[l2 - 1] * offd[l2 - 1];
        double shift = diag[l2] - e2 / (b + ((b >= 0.0) ? 1.0 : -1.0)
                                             * sqrt(b * b + e2));

        /* One implicit QR step (Francis / Givens bulge chase):
         *   - Initial vector: [diag[l1] - shift, offd[l1]]
         *   - Chase the 2x1 bulge from position l1 to l2-1.
         *
         * At each step m:
         *   1. Compute Givens (c,s) to zero the bulge y.
         *   2. Update prev off-diagonal (offd[m-1] = r for m > l1).
         *   3. Apply G^T T G to the 2x2 block [m, m+1].
         *   4. Propagate bulge: x = new_offd[m], y = -s * offd[m+1].
         *   5. Apply G to Z columns m and m+1.
         */
        double x = diag[l1] - shift;
        double y = offd[l1];

        for (npy_intp m = l1; m < l2; m++) {
            double c, s, r;
            dlartg(x, y, &c, &s, &r);

            /* Update previous off-diagonal: after right-mult G_left on cols [m, m+1],
             * T_new[m-1, m] = c * T[m-1,m] + s * T[m-1,m+1] = c*x + s*y = r.
             * Here G_left = [[c,s],[-s,c]] with G_left * [x;y] = [r;0], so
             * c*x + s*y = r and the fill-in T_new[m-1,m+1] = -s*x + c*y = 0. */
            if (m > l1)
                offd[m - 1] = r;

            /* Capture current diagonal/off-diagonal for the 2x2 update */
            double dm   = diag[m];
            double dm1  = diag[m + 1];
            double em   = offd[m];  /* off-diagonal at m (will be overwritten) */

            /* Similarity G T G^T where G = [[c,s],[-s,c]]:
             *   T_new = G T G^T (left G, right G^T = [[c,-s],[s,c]])
             *   new_dm   = c^2*dm + s^2*dm1 + 2*c*s*em
             *   new_dm1  = s^2*dm + c^2*dm1 - 2*c*s*em
             *   new_em   = c*s*(dm1 - dm) + (c^2 - s^2)*em
             *
             * Eigenvectors accumulate as Z_new = Z @ G^T (right-multiply by G^T).
             * apply_givens_col(c, s) does Z @ G = Z @ [[c,s],[-s,c]].
             * For Z @ G^T = Z @ [[c,-s],[s,c]] we negate s in the call. */
            diag[m]     = c * c * dm + s * s * dm1 + 2.0 * c * s * em;
            diag[m + 1] = s * s * dm + c * c * dm1 - 2.0 * c * s * em;
            offd[m]     = c * s * (dm1 - dm) + (c * c - s * s) * em;

            /* Z_new = Z @ G^T: apply_givens_col with negated s gives G^T */
            apply_givens_col(Z, ldz, n, m, m + 1, c, -s);

            /* Propagate the bulge for the next step.
             * In the G T G^T convention (G = [[c,s],[-s,c]]):
             *   Right-multiply G^T on cols [m,m+1]: T_right[m+2,m] = s*offd[m+1]  (+s)
             *   Left-multiply G on rows [m,m+1]: T_new[m+1,m+2] = c*offd[m+1]
             * So x = new offd[m], y = +s * offd[m+1] (positive, not negative). */
            if (m < l2 - 1) {
                x = offd[m];         /* new off-diagonal (bulge carrier) */
                y = s * offd[m + 1]; /* fill-in: T_new[m+2, m] = +s * offd[m+1] */
                offd[m + 1] *= c;    /* T_new[m+1, m+2] = c * offd[m+1] */
            }
        }
    }

    /* Copy results back */
    memcpy(d, diag, (size_t)n * sizeof(double));
    free(diag);
    free(offd);

    /* Sort eigenvalues and eigenvectors in ascending order */
    sort_eig(d, Z, ldz, n);

    return converged ? 0 : (int)n;
}

/* dlaed4 — Secular equation solver (LAPACK-style incremental delta).
 *
 * Finds the i-th root lambda of:
 *   f(lambda) = 1 + rho * sum_k z[k]^2 / (d[k] - lambda) = 0
 *
 * The root lies in (d[i], d[i+1]) for i < n-1, or above d[n-1] for i = n-1.
 *
 * Maintains delta[k] = d[k] - lambda incrementally: starts with
 * delta[k] = d[k] - initial_guess, then subtracts each Newton correction.
 * This avoids catastrophic cancellation — critical for eigenvector accuracy.
 *
 * delta: output array of length n.  delta[k] = d[k] - lambda on exit.
 */
static int dlaed4(npy_intp n, npy_intp i,
                  const double *d, const double *z, double rho,
                  double *lambda_out, double *delta)
{
    if (n == 1) {
        *lambda_out = d[0] + rho * z[0] * z[0];
        delta[0] = -rho * z[0] * z[0];
        return 0;
    }

    /* Bracket: lambda_i in (d[i], d[i+1]) for i<n-1, or (d[n-1], d[n-1]+rho*||z||^2).
     * Work in displacement form: tau = lambda - d[i].
     * delta[k] = d[k] - d[i] - tau.
     * lo_tau, hi_tau are bounds on tau. */
    double lo_tau, hi_tau;
    if (i < n - 1) {
        lo_tau = 0.0;
        hi_tau = d[i + 1] - d[i];
    } else {
        lo_tau = 0.0;
        double sum = 0.0;
        for (npy_intp k = 0; k < n; k++) sum += z[k] * z[k];
        hi_tau = rho * sum;
    }

    /* Initial guess: midpoint */
    double tau = (lo_tau + hi_tau) / 2.0;

    /* Initialize delta[k] = d[k] - d[i] - tau (cancellation-free since
     * d[k] - d[i] is exact for k != i, and delta[i] = -tau for k = i) */
    for (npy_intp k = 0; k < n; k++)
        delta[k] = (d[k] - d[i]) - tau;

    double rhoinv = 1.0 / rho;

    for (int iter = 0; iter < 60; iter++) {
        /* Evaluate g(lambda) = 1/rho + sum z[k]^2/delta[k] where g = f/rho */
        double f  = rhoinv;
        double df = 0.0;
        for (npy_intp k = 0; k < n; k++) {
            double dk = delta[k];
            if (fabs(dk) < 1e-300) dk = (dk >= 0.0) ? 1e-300 : -1e-300;
            double temp = z[k] / dk;
            f  += z[k] * temp;
            df += temp * temp;
        }

        /* Convergence test */
        double erretm = 8.0 * fabs(f) + fabs(rhoinv) + fabs(tau) * df;
        if (fabs(f) <= EPS * erretm) {
            *lambda_out = d[i] + tau;
            return 0;
        }

        /* Update bracket on tau */
        if (f <= 0.0) lo_tau = MAX(lo_tau, tau);
        else          hi_tau = MIN(hi_tau, tau);

        /* Newton step: tau_new = tau - f/df.
         * f(lambda)/rho = 1/rho + sum z^2/(d-lambda) is increasing in lambda.
         * f < 0 at small tau → root is above → tau increases (- neg/pos = +). ✓
         * f > 0 at large tau → root is below → tau decreases (- pos/pos = -). ✓ */
        double step = f / df;
        double tau_new = tau - step;

        /* Ensure within bracket */
        if (tau_new <= lo_tau)
            tau_new = (lo_tau + tau) / 2.0;
        if (tau_new >= hi_tau)
            tau_new = (hi_tau + tau) / 2.0;

        /* Incremental update */
        double correction = tau_new - tau;
        for (npy_intp k = 0; k < n; k++)
            delta[k] -= correction;
        tau = tau_new;
    }

    /* Did not converge — return best estimate */
    *lambda_out = d[i] + tau;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Forward declaration for recursion
 * ---------------------------------------------------------------------------
 */
static int dstedc_recurse(npy_intp n, double *d, double *e,
                          double *Z, npy_intp ldz,
                          double *work, npy_intp lwork,
                          npy_intp *iwork,
                          jblas_workspace_t *ws,
                          double *merge_scratch);

/* ---------------------------------------------------------------------------
 * merge_rank1 — Merge two eigensystems via rank-1 secular equation.
 *
 * Given:
 *   d[0..n-1]: eigenvalues of the two halves (left half 0..m-1, right m..n-1)
 *   Z[n x n]:  block-diagonal eigenvectors of the two halves (column eigenvectors)
 *   z[n]:      rank-1 connecting vector (rho * last-row-of-left | first-row-of-right)
 *   rho:       positive connecting scalar
 *   m:         split point
 *   n:         total size
 *
 * Overwrites d with merged eigenvalues (ascending) and Z with merged
 * eigenvectors.
 *
 * Returns 0 on success.
 * ---------------------------------------------------------------------------
 */
static int merge_rank1(npy_intp n, npy_intp m,
                       double *d, double *z_vec, double rho,
                       double *Z, npy_intp ldz,
                       double *work, npy_intp lwork,
                       npy_intp *iwork,
                       jblas_workspace_t *ws,
                       double *merge_scratch)
{
    /* O(N) work arrays */
    double *d_defl  = (double *)malloc((size_t)n * sizeof(double));
    double *z_defl  = (double *)malloc((size_t)n * sizeof(double));
    double *d_new   = (double *)malloc((size_t)n * sizeof(double));

    /* Q_sec uses the passed workspace (n*n, fits within lwork) */
    double *Q_sec   = work;

    if (!d_defl || !z_defl || !d_new) {
        free(d_defl); free(z_defl); free(d_new);
        return -1;
    }

    /* Step 1: Copy d, z to working arrays */
    memcpy(d_defl, d, (size_t)n * sizeof(double));
    memcpy(z_defl, z_vec, (size_t)n * sizeof(double));

    /* Track deflation: defl[k] = 1 means eigenvalue k is deflated */
    int *defl = (int *)calloc((size_t)n, sizeof(int));
    if (!defl) {
        free(d_defl); free(z_defl); free(d_new); return -1;
    }

    /* Initialize Q_sec to identity */
    memset(Q_sec, 0, (size_t)n * (size_t)n * sizeof(double));
    for (npy_intp k = 0; k < n; k++)
        Q_sec[k * n + k] = 1.0;

    /* Step 2: Type (a) deflation — local relative threshold.
     * Tests whether pole k's contribution rho*z[k]^2/(d[k]-lambda) is negligible. */
    for (npy_intp k = 0; k < n; k++) {
        double rz2 = rho * z_defl[k] * z_defl[k];
        if (rz2 <= 8.0 * EPS * fmax(fabs(d_defl[k]), rz2)) {
            defl[k] = 1;
        }
    }

    /* Step 3: Type (b) deflation: close eigenvalues — merge via Givens.
     * Local relative threshold: 8*eps*max(|d[k]|, |d[j]|). */
    for (npy_intp k = 0; k < n - 1; k++) {
        if (defl[k]) continue;
        for (npy_intp j = k + 1; j < n; j++) {
            if (defl[j]) continue;
            double tol_kj = 8.0 * EPS * fmax(fabs(d_defl[k]), fabs(d_defl[j]));
            if (fabs(d_defl[k] - d_defl[j]) <= tol_kj) {
                /* Rotate z to kill z[j] */
                double c, s, r;
                dlartg(z_defl[k], z_defl[j], &c, &s, &r);
                z_defl[k] = r;
                z_defl[j] = 0.0;
                defl[j]   = 1;
                d_new[j]  = d_defl[j];  /* deflated eigenvalue */
                /* Apply rotation to Q_sec columns k and j */
                for (npy_intp row = 0; row < n; row++) {
                    double qk = Q_sec[row * n + k];
                    double qj = Q_sec[row * n + j];
                    Q_sec[row * n + k] =  c * qk + s * qj;
                    Q_sec[row * n + j] = -s * qk + c * qj;
                }
            }
        }
    }

    /* Collect non-deflated indices */
    npy_intp *nondfl = (npy_intp *)malloc((size_t)n * sizeof(npy_intp));
    npy_intp *dfl    = (npy_intp *)malloc((size_t)n * sizeof(npy_intp));
    if (!nondfl || !dfl) {
        free(d_defl); free(z_defl); free(d_new);
        free(defl); free(nondfl); free(dfl);
        return -1;
    }
    npy_intp n_nd = 0, n_d = 0;
    for (npy_intp k = 0; k < n; k++) {
        if (!defl[k]) nondfl[n_nd++] = k;
        else          dfl   [n_d++]  = k;
    }

    /* Step 4: Solve secular equation for each non-deflated root */
    /* Build compact arrays for the solver */
    double *d_nd = (double *)malloc((size_t)n_nd * sizeof(double));
    double *z_nd = (double *)malloc((size_t)n_nd * sizeof(double));
    if (!d_nd || !z_nd) {
        free(d_defl); free(z_defl); free(d_new);
        free(defl); free(nondfl); free(dfl); free(d_nd); free(z_nd);
        return -1;
    }
    for (npy_intp k = 0; k < n_nd; k++) {
        d_nd[k] = d_defl[nondfl[k]];
        z_nd[k] = z_defl[nondfl[k]];
    }

    /* Sort d_nd ascending, keeping z_nd and nondfl aligned.
     * The input d array from two sub-recursions may not be globally sorted
     * (left and right halves are individually sorted but interleaved).
     * We must also track nondfl to correctly assign eigenvalues back. */
    for (npy_intp i = 1; i < n_nd; i++) {
        double kd = d_nd[i];
        double kz = z_nd[i];
        npy_intp ki = nondfl[i];
        npy_intp j = i - 1;
        while (j >= 0 && d_nd[j] > kd) {
            d_nd[j + 1]  = d_nd[j];
            z_nd[j + 1]  = z_nd[j];
            nondfl[j + 1] = nondfl[j];
            j--;
        }
        d_nd[j + 1]  = kd;
        z_nd[j + 1]  = kz;
        nondfl[j + 1] = ki;
    }

    double *lam_nd = (double *)malloc((size_t)n_nd * sizeof(double));
    /* Delta matrix: delta_mat[i * n_nd + k] = d[k] - lam[i].
     * Stored from dlaed4 to avoid recomputing d[k]-lam[i] (catastrophic
     * cancellation when they are close). Layout: row i = delta vector for
     * eigenvalue i. */
    double *delta_mat = (double *)malloc((size_t)n_nd * (size_t)n_nd * sizeof(double));
    if (!lam_nd || !delta_mat) {
        free(lam_nd); free(delta_mat);
        free(d_defl); free(z_defl); free(d_new);
        free(defl); free(nondfl); free(dfl); free(d_nd); free(z_nd);
        return -1;
    }

    int n_secular_failures = 0;
    for (npy_intp i = 0; i < n_nd; i++) {
        int info = dlaed4(n_nd, i, d_nd, z_nd, rho,
                          &lam_nd[i], delta_mat + i * n_nd);
        if (info != 0) {
            n_secular_failures++;
            lam_nd[i] = d_nd[i];
            for (npy_intp k = 0; k < n_nd; k++)
                delta_mat[i * n_nd + k] = d_nd[k] - d_nd[i];
        }
    }
    if (n_secular_failures > 0) {
        fprintf(stderr, "jblas dstedc: %d/%ld secular equation(s) "
                "failed to converge at merge size %ld — using fallback "
                "eigenvalues (residual check will catch bad results)\n",
                n_secular_failures, (long)n_nd, (long)n);
    }

    /* Step 5: Secular eigenvectors (LAPACK dlaed3 algorithm).
     *
     * Uses delta vectors from dlaed4 to avoid precision loss.
     *
     * For each pole k, compute weight W[k]:
     *   W[k] = delta_mat[k][k]  (= d[k] - lam[k], the "own" gap)
     *   then for each j != k: W[k] *= delta_mat[j][k] / (d[k] - d[j])
     *   W[k] = sgn(z[k]) * sqrt(|W[k]|)
     *
     * Eigenvector i, component k: q[k] = W[k] / delta_mat[i][k], normalize.
     */
    double *Q_nd = (double *)calloc((size_t)n_nd * (size_t)n_nd, sizeof(double));
    double *W_nd = (double *)malloc((size_t)n_nd * sizeof(double));
    if (!Q_nd || !W_nd) {
        free(Q_nd); free(W_nd);
        free(d_defl); free(z_defl); free(d_new);
        free(defl); free(nondfl); free(dfl); free(d_nd); free(z_nd);
        free(lam_nd); free(delta_mat);
        return -1;
    }

    for (npy_intp k = 0; k < n_nd; k++) {
        /* W[k] = delta_mat[k][k] = d[k] - lam[k] */
        double w = delta_mat[k * n_nd + k];
        for (npy_intp j = 0; j < n_nd; j++) {
            if (j == k) continue;
            /* delta_mat[j][k] = d[k] - lam[j] */
            double num = delta_mat[j * n_nd + k];
            double den = d_nd[k] - d_nd[j];
            if (fabs(den) < 1e-300)
                den = (den >= 0.0) ? 1e-300 : -1e-300;
            w *= num / den;
        }
        double sign_z = (z_nd[k] >= 0.0) ? 1.0 : -1.0;
        W_nd[k] = sign_z * sqrt(fabs(w));
    }

    for (npy_intp i = 0; i < n_nd; i++) {
        double norm2 = 0.0;
        for (npy_intp k = 0; k < n_nd; k++) {
            /* Use delta from dlaed4 directly */
            double dk = delta_mat[i * n_nd + k];
            if (fabs(dk) < 1e-300)
                dk = (dk >= 0.0) ? 1e-300 : -1e-300;
            double val = W_nd[k] / dk;
            Q_nd[k * n_nd + i] = val;
            norm2 += val * val;
        }
        double norm = sqrt(norm2);
        if (norm > 0.0) {
            for (npy_intp k = 0; k < n_nd; k++)
                Q_nd[k * n_nd + i] /= norm;
        }
    }
    free(W_nd);
    free(delta_mat);

    /* Step 6: Assemble the full secular eigenvector matrix Q_sec.
     *
     * At this point Q_sec holds Q_b = accumulated type-b Givens rotations
     * (identity if no type-b deflations occurred).
     *
     * The overall transformation is Q_total = Q_b @ Q_secular where:
     *   Q_secular[:,nondfl[i]] = Q_nd[:,i]  (secular eigenvectors)
     *   Q_secular[:,dfl[j]]    = e_{dfl[j]} (deflated: identity column)
     *
     * So:
     *   Q_total[:,nondfl[i]] = Q_b[:,nondfl] @ Q_nd[:,i]
     *   Q_total[:,dfl[j]]    = Q_b[:,dfl[j]]  (unchanged from Q_b)
     *
     * We compute the new non-deflated columns into a temporary buffer,
     * then write them back. Deflated columns of Q_sec are already correct.
     */

    /* Compute non-deflated columns: Q_total[:,nondfl[i]] = Q_b[:,nondfl] @ Q_nd[:,i]
     * Gather Q_b columns indexed by nondfl[] into contiguous Q_b_cols, then GEMM.
     * Use merge_scratch as temporary (size N*N >= n*n_nd always). */
    if (n_nd > 0) {
        /* Q_b_cols: n rows x n_nd cols — gather nondfl columns of Q_sec */
        double *Q_b_cols = merge_scratch;  /* reuse scratch (n*n_nd <= N*N) */
        for (npy_intp j = 0; j < n_nd; j++) {
            npy_intp src_col = nondfl[j];
            for (npy_intp row = 0; row < n; row++)
                Q_b_cols[row * n_nd + j] = Q_sec[row * n + src_col];
        }

        /* Q_nd_full = Q_b_cols(n x n_nd) @ Q_nd(n_nd x n_nd)
         * Result into merge_scratch offset past Q_b_cols.
         * But Q_b_cols IS merge_scratch, so we need a separate output area.
         * Allocate Q_nd_full locally (n x n_nd). */
        double *Q_nd_full = (double *)calloc((size_t)n * (size_t)n_nd, sizeof(double));
        if (!Q_nd_full) {
            free(d_defl); free(z_defl); free(d_new);
            free(defl); free(nondfl); free(dfl); free(d_nd); free(z_nd);
            free(lam_nd); free(Q_nd);
            return -1;
        }

        if (ws) {
            jblas_dgemm_ext_ws(n, n_nd, n_nd,
                               Q_b_cols, n_nd,
                               Q_nd, n_nd,
                               Q_nd_full, n_nd,
                               0, 0, 1.0, 0.0, ws);
        } else {
            jblas_dgemm_ext(n, n_nd, n_nd,
                            Q_b_cols, n_nd,
                            Q_nd, n_nd,
                            Q_nd_full, n_nd,
                            0, 0);
        }

        /* Write updated non-deflated columns back into Q_sec */
        for (npy_intp row = 0; row < n; row++) {
            for (npy_intp i = 0; i < n_nd; i++)
                Q_sec[row * n + nondfl[i]] = Q_nd_full[row * n_nd + i];
        }
        free(Q_nd_full);
    }

    /* Record eigenvalues */
    for (npy_intp i = 0; i < n_nd; i++)
        d_new[nondfl[i]] = lam_nd[i];
    for (npy_intp i = 0; i < n_d; i++)
        d_new[dfl[i]] = d_defl[dfl[i]];

    /* Step 7: Back-transform Z: Z_new = Z_old @ Q_sec
     * Both Z and Q_sec are n x n row-major.
     * Z_new[i,j] = sum_k Z_old[i,k] * Q_sec[k,j]
     *
     * Use merge_scratch as Z_new (size N*N >= n*ldz since n<=N and ldz=N).
     * If merge_scratch is NULL, fall back to malloc. */
    double *Z_new;
    int Z_new_malloced = 0;
    if (merge_scratch) {
        Z_new = merge_scratch;
    } else {
        Z_new = (double *)malloc((size_t)n * (size_t)ldz * sizeof(double));
        if (!Z_new) {
            free(d_defl); free(z_defl); free(d_new);
            free(defl); free(nondfl); free(dfl); free(d_nd); free(z_nd);
            free(lam_nd); free(Q_nd);
            return -1;
        }
        Z_new_malloced = 1;
    }

    if (ws) {
        jblas_dgemm_ext_ws(n, n, n,
                           Z, ldz, Q_sec, n, Z_new, ldz,
                           0, 0, 1.0, 0.0, ws);
    } else {
        jblas_dgemm_ext(n, n, n,
                        Z, ldz, Q_sec, n, Z_new, ldz,
                        0, 0);
    }

    /* Copy Z_new back to Z */
    for (npy_intp row = 0; row < n; row++)
        memcpy(Z + row * ldz, Z_new + row * ldz, (size_t)n * sizeof(double));

    /* Copy d_new back to d */
    memcpy(d, d_new, (size_t)n * sizeof(double));

    if (Z_new_malloced) free(Z_new);
    free(d_defl); free(z_defl); free(d_new);
    free(defl); free(nondfl); free(dfl); free(d_nd); free(z_nd);
    free(lam_nd); free(Q_nd);

    /* Sort eigenvalues and eigenvectors ascending */
    sort_eig(d, Z, ldz, n);

    return 0;
}

/* ---------------------------------------------------------------------------
 * dstedc_recurse — Recursive D&C eigensolver (operates on subproblem).
 *
 * n    : subproblem size.
 * d    : diagonal of subproblem (length n), overwritten with eigenvalues.
 * e    : off-diagonal of subproblem (length n-1).
 * Z    : n x n row-major matrix (stride ldz); on entry identity,
 *        on exit eigenvectors as columns.
 * ldz  : leading dimension of Z (= full N for top-level call).
 * ---------------------------------------------------------------------------
 */
static int dstedc_recurse(npy_intp n, double *d, double *e,
                          double *Z, npy_intp ldz,
                          double *work, npy_intp lwork,
                          npy_intp *iwork,
                          jblas_workspace_t *ws,
                          double *merge_scratch)
{
    if (n <= 0) return 0;
    if (n == 1) return 0;

    /* Base case */
    if (n <= DSTEDC_BASE) {
        return dsteqr_base(n, d, e, Z, ldz);
    }

    /* Split at m = n/2 */
    npy_intp m = n / 2;

    double rho_orig = fabs(e[m - 1]);

    /* Adjust diagonal: d[m-1] -= |e[m-1]|, d[m] -= |e[m-1]| */
    d[m - 1] -= rho_orig;
    d[m]     -= rho_orig;

    /* Build rank-1 connecting vector z:
     * z[k] = last column of Z_left  (k < m)  -> Z[k, m-1] (column m-1)
     * z[k] = first column of Z_right (k >= m) -> Z[k-m, 0] (column 0)
     * But since Z starts as identity (and sub-calls will overwrite),
     * at the point of the call Z is still identity for the sub-blocks.
     * After recursion, Z will have been updated to the subproblem eigenvectors.
     * The merge vector comes from the last row of the left eigenvectors
     * and the first row of the right eigenvectors.
     */

    /* Create local sub-Z for left half (m x m, within the larger Z) */
    /* The full Z is n x ldz.  Left half eigenvectors occupy top-left m x m block.
     * Right half eigenvectors occupy bottom-right (n-m) x (n-m) block.
     * Both start as identity (the full Z starts as identity passed by eigh.c).
     * We recurse in-place on the diagonal blocks. */

    /* Left half: rows 0..m-1, cols 0..m-1 of Z */
    int ret;
    ret = dstedc_recurse(m, d, e, Z, ldz, work, lwork, iwork,
                          ws, merge_scratch);
    if (ret != 0) return ret;

    /* Right half: rows m..n-1, cols m..n-1 of Z */
    ret = dstedc_recurse(n - m, d + m, e + m, Z + m * ldz + m, ldz,
                          work, lwork, iwork, ws, merge_scratch);
    if (ret != 0) return ret;

    /* Build z vector from the post-recursion eigenvectors.
     *
     * The D&C rank-1 decomposition is:
     *   T = diag(T_L_bar, T_R_bar) + rho * z_orig * z_orig^T
     * where z_orig = [0,...,0, 1, 1, 0,...,0] (1s at positions m-1 and m).
     *
     * After the recursive calls, the block-diagonal eigenvector matrix is:
     *   Q_block = block_diag(Q_L, Q_R)
     * stored in Z: left block in Z[0:m, 0:m], right block in Z[m:n, m:n].
     *
     * The z vector in the eigenvector basis (z_tilde = Q_block^T @ z_orig) is:
     *   z_tilde[j]   = Q_L[m-1, j]  for j = 0..m-1   (last ROW of Q_L)
     *   z_tilde[m+j] = Q_R[0,   j]  for j = 0..n-m-1 (first ROW of Q_R)
     *
     * In row-major Z (column eigenvectors):
     *   Q_L[m-1, j] = Z[(m-1)*ldz + j]   (row m-1, column j of left block)
     *   Q_R[0,   j] = Z[m*ldz + (m+j)]   (row m,   column m+j of full Z)
     */
    double *z_vec = (double *)malloc((size_t)n * sizeof(double));
    if (!z_vec) return -1;

    for (npy_intp j = 0; j < m; j++)
        z_vec[j] = Z[(m - 1) * ldz + j];          /* last row of left block  */
    for (npy_intp j = 0; j < n - m; j++)
        z_vec[m + j] = Z[m * ldz + (m + j)];      /* first row of right block */

    /* LAPACK DLAED1 convention: scale z by 1/sqrt(2), double rho.
     * This preserves rho * z * z^T since 2*rho * (z/sqrt2)*(z/sqrt2)^T = rho * z*z^T.
     * Sign absorption: if e[m-1] < 0, negate z so secular solver always gets rho > 0. */
    double rho_raw = e[m - 1];  /* signed off-diagonal */
    double rho = 2.0 * fabs(rho_raw);
    double sign_rho = (rho_raw >= 0.0) ? 1.0 : -1.0;
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    for (npy_intp j = 0; j < n; j++)
        z_vec[j] *= inv_sqrt2 * sign_rho;

    /* If rho is zero, the two sub-problems are decoupled */
    if (rho == 0.0) {
        free(z_vec);
        sort_eig(d, Z, ldz, n);
        return 0;
    }

    /* Merge */
    ret = merge_rank1(n, m, d, z_vec, rho, Z, ldz, work, lwork, iwork,
                      ws, merge_scratch);
    free(z_vec);

    return ret;
}

/* ---------------------------------------------------------------------------
 * jblas_dstedc_c — Public D&C tridiagonal eigensolver.
 *
 * Parameters: see jblas.h
 * Returns: 0 on success, -1 on allocation failure.
 * ---------------------------------------------------------------------------
 */
int jblas_dstedc_c(npy_intp N, double *d, double *e,
                   double *Z, npy_intp ldz,
                   jblas_workspace_t *ws)
{
    if (N <= 0) return 0;
    if (N == 1) return 0;

    /* Initialize Z to identity (caller must provide N x N buffer) */
    memset(Z, 0, (size_t)N * (size_t)ldz * sizeof(double));
    for (npy_intp k = 0; k < N; k++)
        Z[k * ldz + k] = 1.0;

    /* Allocate workspace: N*N for merge buffer + 5*N for index arrays.
     * Also allocate a single N*N merge scratch buffer passed through
     * recursion to avoid per-merge malloc in merge_rank1. */
    npy_intp lwork = N * N;
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    npy_intp *iwork = (npy_intp *)malloc(5 * (size_t)N * sizeof(npy_intp));
    double *d_orig = (double *)malloc((size_t)N * sizeof(double));
    double *e_orig = (double *)malloc((size_t)(N - 1) * sizeof(double));
    double *merge_scratch = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
    if (!work || !iwork || !d_orig || !e_orig || !merge_scratch) {
        free(work); free(iwork); free(d_orig); free(e_orig);
        free(merge_scratch);
        return -1;
    }

    memcpy(d_orig, d, (size_t)N * sizeof(double));
    memcpy(e_orig, e, (size_t)(N - 1) * sizeof(double));

    /* Run D&C eigensolver */
    int ret = dstedc_recurse(N, d, e, Z, ldz, work, lwork, iwork,
                              ws, merge_scratch);

    /* Check if D&C result needs QR fallback: either D&C returned non-zero
     * (convergence or secular equation failure), or the residual is bad. */
    int need_fallback = 0;
    if (ret != 0) {
        fprintf(stderr, "jblas dstedc: D&C returned %d (N=%ld), "
                "attempting QR fallback\n", ret, (long)N);
        need_fallback = 1;
    } else {
        double resid = tridiag_eig_residual(N, d_orig, e_orig, d, Z, ldz);
        if (resid > 1e-10) {
            fprintf(stderr, "jblas dstedc: D&C residual %.2e exceeds 1e-10 "
                    "(N=%ld), attempting QR fallback\n", resid, (long)N);
            need_fallback = 1;
        }
    }

    if (need_fallback) {
        if (N > 2000) {
            fprintf(stderr, "jblas dstedc: QR fallback on N=%ld — "
                    "this is O(N^3) and may take minutes\n", (long)N);
        }
        memcpy(d, d_orig, (size_t)N * sizeof(double));
        memcpy(e, e_orig, (size_t)(N - 1) * sizeof(double));
        memset(Z, 0, (size_t)N * (size_t)ldz * sizeof(double));
        for (npy_intp k = 0; k < N; k++)
            Z[k * ldz + k] = 1.0;
        ret = dsteqr_base(N, d, e, Z, ldz);
    }

    free(work);
    free(iwork);
    free(d_orig);
    free(e_orig);
    free(merge_scratch);

    /* Final sort (should already be sorted, but ensure) */
    if (ret == 0)
        sort_eig(d, Z, ldz, N);

    return ret;
}
