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
 *   Base case (N <= DSTEDC_BASE, currently 64): Implicit QR iteration (Francis shift) on the
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
 * Secular equation solver (LAPACK dlaed4 algorithm):
 *   ORGATI origin selection, rational interpolation with dlaed5 (N=2) and
 *   dlaed6 (3-pole) helpers.  Produces full-precision delta vectors for
 *   dlaed3 weight product.
 *
 * Memory:
 *   dstedc_c allocates its own internal workspace: 2*N*N (work + merge_scratch)
 *   + O(N) (d_orig, e_orig, iwork), passed through recursion.
 *   merge_rank1 uses the workspace for Q_sec (N x N) and additionally allocates
 *   Z_new (N x N), delta_mat (up to N_nd x N_nd), Q_nd (N_nd x N_nd), and
 *   Q_nd_full (N x N_nd) locally.
 *   Peak merge-step memory is ~6 * N^2 doubles (~48 bytes/element):
 *   Z (caller), work=Q_sec, merge_scratch=Z_new, delta_mat, Q_nd, Q_nd_full.
 *   This is the worst case (no deflation); with deflation, delta_mat/Q_nd are
 *   n_nd * n_nd where n_nd < N, reducing peak memory.
 *   For N=100k, expect ~480 GB peak during the top-level merge.
 *
 * References:
 *   Cuppen (1981), "A Divide and Conquer Method for the Symmetric
 *   Tridiagonal Eigenproblem."
 *   Li (1994), "Solving Secular Equations Stably and Efficiently",
 *   LAPACK Working Note 70.
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

/* Guard macro for PSI/PHI evaluation loops in dlaed4.
 * If delta[k] is exactly zero (tau landed on a pole due to FP rounding),
 * return the current best estimate rather than producing Inf/NaN.
 * LAPACK dlaed4.f relies on bracketing to prevent this, but accumulated
 * eta steps can cancel delta to zero in edge cases. */
#define DLAED4_CHECK_DELTA_ZERO(delta_k, origin, n, d, tau, lambda_out, delta) \
    do { \
        if ((delta_k) == 0.0) { \
            *(lambda_out) = d[(origin)] + (tau); \
            for (npy_intp _zk = 0; _zk < (n); _zk++) \
                (delta)[_zk] = (d[_zk] - d[(origin)]) - (tau); \
            return 0; \
        } \
    } while (0)

/* Threshold for switching to base-case QR iteration.
 * LAPACK uses SMLSIZ ~25; 64 provides a balance between QR base case
 * size and number of D&C merge levels.  Previously 128 as a workaround
 * for secular solver convergence failures; lowered after PSI/PHI rewrite.
 * Sweep of {25,32,48,64,96,128} showed no meaningful difference (QR
 * fallback from dlaed3 residuals dominates); 64 retained as standard. */
#define DSTEDC_BASE 64

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
        if (n > 10000) {
            /* O(N^2) insertion sort at this scale will never complete in
             * reasonable time — treat as fatal rather than silently hanging. */
            fprintf(stderr, "jblas dstedc: sort_eig allocation failed for n=%ld "
                    "(O(n^2) insertion sort would be impractical), aborting\n",
                    (long)n);
            abort();
        }
        fprintf(stderr, "jblas dstedc: sort_eig allocation failed for n=%ld, "
                "falling back to O(n^2) insertion sort\n", (long)n);
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
         *   4. Propagate bulge: x = new_offd[m], y = +s * offd[m+1].
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

/* dlaed5 — Analytical N=2 secular equation solver.
 *
 * Solves 1/rho + z[0]^2/(d[0]-lambda) + z[1]^2/(d[1]-lambda) = 0
 * analytically for the i-th root (i=0 or 1).
 *
 * Reference: LAPACK dlaed5.f
 */
static int dlaed5(npy_intp n, npy_intp i,
                  const double *d, const double *z, double rho,
                  double *lambda_out, double *delta)
{
    (void)n;  /* always 2 */
    double del = d[1] - d[0];  /* gap, positive since d sorted ascending */
    double tau;

    if (i == 0) {
        /* First root: lies in (d[0], d[1]).
         * LAPACK I=1: W-test determines which quadratic formulation to use.
         * Reference: LAPACK dlaed5.f */
        double w = 1.0 + 2.0 * rho * (z[1] * z[1] - z[0] * z[0]) / del;
        if (w > 0.0) {
            /* Root closer to d[0]: tau as displacement from d[0] */
            double b = del + rho * (z[0] * z[0] + z[1] * z[1]);
            double c = rho * z[0] * z[0] * del;
            double disc = b * b - 4.0 * c;
            if (disc < 0.0) disc = 0.0;
            tau = 2.0 * c / (b + sqrt(disc));
            *lambda_out = d[0] + tau;
            delta[0] = -tau;
            delta[1] = del - tau;
        } else {
            /* Root closer to d[1]: tau as displacement from d[1], tau < 0 */
            double b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
            double c = rho * z[1] * z[1] * del;
            double disc = b * b + 4.0 * c;
            if (b > 0.0)
                tau = -2.0 * c / (b + sqrt(disc));
            else
                tau = (b - sqrt(disc)) / 2.0;
            *lambda_out = d[1] + tau;
            delta[0] = -(del + tau);  /* d[0] - lambda */
            delta[1] = -tau;          /* d[1] - lambda */
        }
    } else {
        /* Second root: lies above d[1] (rho > 0).
         * LAPACK I=2: single quadratic, tau > 0 from d[1].
         * Reference: LAPACK dlaed5.f */
        double b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
        double c = rho * z[1] * z[1] * del;
        double disc = b * b + 4.0 * c;
        if (disc < 0.0) disc = 0.0;
        double sq = sqrt(disc);
        if (b > 0.0)
            tau = (b + sq) / 2.0;
        else
            tau = 2.0 * c / (-b + sq);
        *lambda_out = d[1] + tau;
        delta[0] = -(del + tau);
        delta[1] = -tau;
    }
    return 0;
}

/* dlaed6 — Three-pole cubic rational solver for secular equation.
 *
 * Solves the equation:
 *   finit + z2[0]/(delta0[0]-tau) + z2[1]/(delta0[1]-tau) + z2[2]/(delta0[2]-tau) = 0
 *
 * where finit already includes 1/rho and the contribution from all other poles.
 * z2[k] = z[k]^2, delta0[k] = initial delta values for the 3 active poles.
 *
 * The solver uses Halley's method (second-order rational) with Newton
 * fallback and bracket safeguarding.
 *
 * Parameters:
 *   kniter: max iterations (typically 40)
 *   orgati: 1 if origin is left pole, 0 if right
 *   d3: the 3 pole distances (delta values at entry)
 *   z2: z-squared values for the 3 poles
 *   finit: initial function value (without the 3 poles)
 *   tau_out: output displacement
 *   info_out: 0 on success, 1 on failure
 *
 * Reference: LAPACK dlaed6.f
 */
static void dlaed6(int kniter, int orgati,
                   const double d3[3], const double z2[3],
                   double finit,
                   double *tau_out, int *info_out)
{
    *info_out = 0;
    const int MAXIT = (kniter > 0) ? kniter : 40;
    const double SMALL = 2.0 * EPS;  /* convergence factor */

    /* The function is:
     *   f(tau) = finit + z2[0]/(d3[0]-tau) + z2[1]/(d3[1]-tau) + z2[2]/(d3[2]-tau)
     *
     * We need to find tau such that f(tau) = 0.
     * tau must stay within the bracket defined by the poles. */

    double a, b, c_val, fc, df, ddf;
    double lbd, ubd;

    /* Determine bracket from pole positions.
     * For orgati=1 (origin at left): tau in (0, d3[1]) or (0, d3[2])
     * For orgati=0 (origin at right): tau in (d3[0], 0) or (d3[1], 0) */
    if (orgati) {
        lbd = 0.0;
        /* Upper bound: min positive delta */
        ubd = d3[2];
        if (d3[1] > 0.0 && d3[1] < ubd) ubd = d3[1];
        if (d3[0] > 0.0 && d3[0] < ubd) ubd = d3[0];
    } else {
        ubd = 0.0;
        /* Lower bound: max negative delta */
        lbd = d3[0];
        if (d3[1] < 0.0 && d3[1] > lbd) lbd = d3[1];
        if (d3[2] < 0.0 && d3[2] > lbd) lbd = d3[2];
    }

    double tau = 0.0;

    /* Evaluate f(0), f'(0), f''(0) for initial step */
    fc = finit;
    df = 0.0;
    ddf = 0.0;
    for (int k = 0; k < 3; k++) {
        if (d3[k] == 0.0) {
            /* Origin is exactly at a pole — return zero shift. */
            *tau_out = 0.0;
            *info_out = 1;
            return;
        }
        double tmp = 1.0 / d3[k];
        double tmp2 = z2[k] * tmp;
        fc += tmp2;
        tmp2 *= tmp;
        df += tmp2;
        tmp2 *= tmp;
        ddf += tmp2;
    }

    /* If f(0) is already close enough, return 0 */
    if (fabs(fc) < SMALL) {
        *tau_out = tau;
        return;
    }

    /* Newton iteration with cubic rational safeguarding */
    for (int iter = 0; iter < MAXIT; iter++) {
        /* Newton step: eta = -fc / df (first order)
         * Halley step: eta = -fc / (df - 0.5*fc*ddf/df) (second order)
         * Use Halley when safe. */
        double eta;
        if (fabs(df) < 1e-300) {
            *info_out = 1;
            break;
        }

        /* Use Halley's method for faster convergence */
        double halley_denom = df - 0.5 * fc * ddf / df;
        if (fabs(halley_denom) > fabs(df) * 0.1) {
            eta = -fc / halley_denom;
        } else {
            eta = -fc / df;
        }

        /* Safeguard: keep tau + eta within bracket */
        double tau_new = tau + eta;
        if (tau_new <= lbd) {
            eta = (lbd - tau) * 0.5;
            tau_new = tau + eta;
        }
        if (tau_new >= ubd) {
            eta = (ubd - tau) * 0.5;
            tau_new = tau + eta;
        }

        tau = tau_new;

        /* Evaluate f(tau), f'(tau), f''(tau) */
        fc = finit;
        df = 0.0;
        ddf = 0.0;
        for (int k = 0; k < 3; k++) {
            double tmp = 1.0 / (d3[k] - tau);
            double tmp2 = z2[k] * tmp;
            fc += tmp2;
            tmp2 *= tmp;
            df += tmp2;
            tmp2 *= tmp;
            ddf += tmp2;
        }

        /* Update bracket */
        if (fc < 0.0) {
            if (tau > lbd) lbd = tau;
        } else {
            if (tau < ubd) ubd = tau;
        }

        /* Convergence check */
        double erretm = SMALL * (fabs(finit) + fabs(fc));
        if (fabs(fc) <= 8.0 * EPS * erretm || fabs(ubd - lbd) <= SMALL * fabs(tau)) {
            *tau_out = tau;
            return;
        }
    }

    /* Did not converge */
    *tau_out = tau;
    *info_out = 1;
}

/* dlaed4 — Secular equation solver with ORGATI origin selection and
 * rational interpolation.
 *
 * Finds the i-th root lambda of:
 *   f(lambda) = 1/rho + sum_k z[k]^2 / (d[k] - lambda) = 0
 *
 * The root lies in (d[i], d[i+1]) for i < n-1, or above d[n-1] for i = n-1.
 *
 * Algorithm (LAPACK dlaed4.f):
 *   - N=1: direct formula. N=2: delegate to dlaed5.
 *   - ORGATI: evaluate secular function at midpoint of (d[i], d[i+1]).
 *     Choose d[i] or d[i+1] as origin depending on sign.
 *   - Initial guess via stabilized quadratic from the two central poles.
 *   - Single-centre-pole PSI/PHI split with LAPACK II/IIM1/IIP1 indexing.
 *   - A/B/C rational interpolation as primary step (quadratic convergence).
 *     Newton step used only as fallback when A/B/C produces wrong-sign step.
 *   - SWTCH3: detects 3-pole clustering, dispatches to dlaed6.
 *   - SWTCH: detects slow convergence, adjusts C computation.
 *   - Geometric mean safeguard: combines Newton and bisection when
 *     rational step is out of bracket but Newton is in bracket.
 *   - Incremental delta maintenance: delta[k] -= eta each iteration.
 *
 * delta: output array of length n.  delta[k] = d[k] - lambda on exit.
 *
 * Reference: LAPACK dlaed4.f, Li (1994) LAPACK Working Note 70.
 */
static int dlaed4(npy_intp n, npy_intp i,
                  const double *d, const double *z, double rho,
                  double *lambda_out, double *delta)
{
    npy_intp k;
    double rhoinv = 1.0 / rho;

    /* N=1: trivial */
    if (n == 1) {
        *lambda_out = d[0] + rho * z[0] * z[0];
        delta[0] = -rho * z[0] * z[0];
        return 0;
    }

    /* N=2: delegate to analytical solver */
    if (n == 2) {
        return dlaed5(n, i, d, z, rho, lambda_out, delta);
    }

    /* ----------------------------------------------------------------
     * N >= 3: ORGATI determination, initial guess, iteration
     * ---------------------------------------------------------------- */
    const int MAXIT = 60;

    if (i < n - 1) {
        /* ============================================================
         * Interior eigenvalue: root in (d[i], d[i+1])
         * ============================================================ */
        double del = d[i + 1] - d[i];
        double midpt = del * 0.5;

        /* Evaluate secular function at midpoint to determine ORGATI.
         * Initialize delta relative to d[i] temporarily. */
        for (k = 0; k < n; k++)
            delta[k] = (d[k] - d[i]) - midpt;

        double psi = 0.0, phi = 0.0;
        for (k = 0; k < i; k++)
            psi += z[k] * z[k] / delta[k];
        for (k = i + 2; k < n; k++)
            phi += z[k] * z[k] / delta[k];
        double c = rhoinv + psi + phi;
        double w = c + z[i] * z[i] / delta[i] + z[i + 1] * z[i + 1] / delta[i + 1];

        /* ORGATI: if w > 0, root closer to d[i], origin at d[i].
         * Otherwise origin at d[i+1]. */
        int orgati = (w > 0.0) ? 1 : 0;

        /* Initial guess via stabilized quadratic */
        double tau, dltlb, dltub;
        if (orgati) {
            double a = c * del + z[i] * z[i] + z[i + 1] * z[i + 1];
            double b = z[i] * z[i] * del;
            if (a > 0.0)
                tau = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
            else
                tau = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
            dltlb = 0.0;
            dltub = midpt;
        } else {
            double a = c * del - z[i] * z[i] - z[i + 1] * z[i + 1];
            double b = z[i + 1] * z[i + 1] * del;
            if (a < 0.0)
                tau = 2.0 * b / (a - sqrt(fabs(a * a - 4.0 * b * c)));
            else
                tau = -(a + sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
            dltlb = -midpt;
            dltub = 0.0;
        }

        /* Clamp initial tau to bracket */
        if (tau <= dltlb) tau = dltlb * 0.5;
        if (tau >= dltub) tau = dltub * 0.5;

        /* Initialize delta relative to chosen origin */
        npy_intp origin = orgati ? i : i + 1;
        for (k = 0; k < n; k++)
            delta[k] = (d[k] - d[origin]) - tau;

        /* LAPACK IIM1/IIP1/II indexing (all 0-based C indices).
         * Maps from LAPACK 1-based (I is eigenvalue index = our i+1):
         *   ORGATI=TRUE:  II=i,   IIM1=i-1, IIP1=i+1
         *   ORGATI=FALSE: II=i+1, IIM1=i,   IIP1=i+2
         *
         * Single centre pole (II) with PSI below and PHI above:
         *   orgati:  PSI k=0..i-1, centre k=i, PHI k=i+1..n-1
         *   !orgati: PSI k=0..i,   centre k=i+1, PHI k=i+2..n-1
         */
        npy_intp ii, iim1, iip1;
        if (orgati) {
            ii = i;
            iim1 = i - 1;
            iip1 = i + 1;
        } else {
            ii = i + 1;
            iim1 = i;
            iip1 = i + 2;
        }

        npy_intp psi_end;   /* PSI sums k=0..psi_end-1 */
        npy_intp phi_start; /* PHI sums k=phi_start..n-1 */
        if (orgati) {
            psi_end = i;       /* k=0..i-1 */
            phi_start = i + 1; /* k=i+1..n-1 */
        } else {
            psi_end = i + 1;   /* k=0..i */
            phi_start = i + 2; /* k=i+2..n-1 */
        }

        /* First PSI/PHI/W evaluation (LAPACK lines 542-600) */
        double dpsi = 0.0, psi_val = 0.0, erretm = 0.0;
        for (k = 0; k < psi_end; k++) {
            DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
            double temp = z[k] / delta[k];
            psi_val += z[k] * temp;
            dpsi += temp * temp;
            erretm += psi_val;  /* running sum, not fabs — matches LAPACK */
        }
        erretm = fabs(erretm);

        double dphi = 0.0, phi_val = 0.0;
        for (k = n - 1; k >= phi_start; k--) {
            DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
            double temp = z[k] / delta[k];
            phi_val += z[k] * temp;
            dphi += temp * temp;
            erretm += phi_val;  /* running sum like LAPACK */
        }

        /* W without centre pole = RHOINV + PSI + PHI */
        double w_val = rhoinv + phi_val + psi_val;

        /* SWTCH3 detection (LAPACK lines 568-576) */
        int swtch3 = 0;
        if (orgati) {
            if (w_val < 0.0) swtch3 = 1;
        } else {
            if (w_val > 0.0) swtch3 = 1;
        }
        if (ii == 0 || ii == n - 1)
            swtch3 = 0;

        /* Add centre pole contribution (LAPACK lines 578-583) */
        double dw;
        int swtch = 0;
        double prew;
        {
            DLAED4_CHECK_DELTA_ZERO(delta[ii], origin, n, d, tau, lambda_out, delta);
            double temp_ii = z[ii] / delta[ii];
            dw = dpsi + dphi + temp_ii * temp_ii;
            double temp_contrib = z[ii] * temp_ii;  /* LAPACK's TEMP */
            w_val += temp_contrib;
            erretm = 8.0 * (phi_val - psi_val) + erretm + 2.0 * fabs(rhoinv)
                   + 3.0 * fabs(temp_contrib) + fabs(tau) * dw;
        }

        /* Test for convergence (LAPACK lines 585-592) */
        if (fabs(w_val) <= EPS * erretm) {
            *lambda_out = d[origin] + tau;
            for (k = 0; k < n; k++)
                delta[k] = (d[k] - d[origin]) - tau;
            return 0;
        }

        /* Update bracket */
        if (w_val <= 0.0) dltlb = fmax(dltlb, tau);
        else              dltub = fmin(dltub, tau);

        /* ---- First step: A/B/C rational interpolation ---- */
        /* LAPACK lines 599-662: NITER=2 step computation */
        {
            double eta;

            if (!swtch3) {
                /* Non-SWTCH3: A/B/C from two gap-bounding poles.
                 * LAPACK uses DELTA(I) and DELTA(IP1) which are the
                 * poles bounding the gap: delta[i] and delta[i+1] (0-based). */
                double C_val;
                if (orgati) {
                    DLAED4_CHECK_DELTA_ZERO(delta[i], origin, n, d, tau, lambda_out, delta);
                    C_val = w_val - delta[i + 1] * dw
                          - (d[i] - d[i + 1]) * (z[i] / delta[i]) * (z[i] / delta[i]);
                } else {
                    DLAED4_CHECK_DELTA_ZERO(delta[i + 1], origin, n, d, tau, lambda_out, delta);
                    C_val = w_val - delta[i] * dw
                          - (d[i + 1] - d[i]) * (z[i + 1] / delta[i + 1]) * (z[i + 1] / delta[i + 1]);
                }
                double A_val = (delta[i] + delta[i + 1]) * w_val
                             - delta[i] * delta[i + 1] * dw;
                double B_val = delta[i] * delta[i + 1] * w_val;

                if (C_val == 0.0) {
                    if (A_val == 0.0) {
                        if (orgati)
                            A_val = z[i] * z[i] + delta[i + 1] * delta[i + 1] * (dpsi + dphi);
                        else
                            A_val = z[i + 1] * z[i + 1] + delta[i] * delta[i] * (dpsi + dphi);
                    }
                    eta = B_val / A_val;
                } else if (A_val <= 0.0) {
                    eta = (A_val - sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val))) / (2.0 * C_val);
                } else {
                    eta = 2.0 * B_val / (A_val + sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val)));
                }
            } else {
                /* SWTCH3: three-pole interpolation via dlaed6.
                 * LAPACK lines 630-660. */
                double temp_rhoinv_psi_phi = rhoinv + psi_val + phi_val;
                double zz[3];
                double C_3;
                if (orgati) {
                    DLAED4_CHECK_DELTA_ZERO(delta[iim1], origin, n, d, tau, lambda_out, delta);
                    double t1 = z[iim1] / delta[iim1];
                    t1 = t1 * t1;
                    C_3 = temp_rhoinv_psi_phi - delta[iip1] * (dpsi + dphi)
                        - (d[iim1] - d[iip1]) * t1;
                    zz[0] = z[iim1] * z[iim1];
                    zz[2] = delta[iip1] * delta[iip1] * ((dpsi - t1) + dphi);
                } else {
                    DLAED4_CHECK_DELTA_ZERO(delta[iip1], origin, n, d, tau, lambda_out, delta);
                    double t1 = z[iip1] / delta[iip1];
                    t1 = t1 * t1;
                    C_3 = temp_rhoinv_psi_phi - delta[iim1] * (dpsi + dphi)
                        - (d[iip1] - d[iim1]) * t1;
                    zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - t1));
                    zz[2] = z[iip1] * z[iip1];
                }
                zz[1] = z[ii] * z[ii];

                double d3[3] = { delta[iim1], delta[ii], delta[iip1] };
                int info6;
                dlaed6(40, orgati, d3, zz, C_3, &eta, &info6);
                if (info6 != 0) {
                    /* dlaed6 failed — fall back to Newton step */
                    eta = -w_val / dw;
                }
            }

            /* Sign check: eta*w should be < 0 (LAPACK line 670) */
            if (w_val * eta >= 0.0)
                eta = -w_val / dw;

            /* Safeguard for first step (LAPACK lines 672-685):
             * Simple bisection if out of bracket. */
            {
                double temp_tau = tau + eta;
                if (temp_tau > dltub || temp_tau < dltlb) {
                    if (w_val < 0.0)
                        eta = (dltub - tau) / 2.0;
                    else
                        eta = (dltlb - tau) / 2.0;
                }
            }

            /* Save w for SWTCH detection (LAPACK line 688: PREW = W) */
            prew = w_val;

            /* Apply step */
            for (k = 0; k < n; k++)
                delta[k] -= eta;

            /* Re-evaluate PSI/PHI/W after first step (LAPACK lines 694-720) */
            dpsi = 0.0; psi_val = 0.0; erretm = 0.0;
            for (k = 0; k < psi_end; k++) {
                DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau + eta, lambda_out, delta);
                double t = z[k] / delta[k];
                psi_val += z[k] * t;
                dpsi += t * t;
                erretm += psi_val;
            }
            erretm = fabs(erretm);

            dphi = 0.0; phi_val = 0.0;
            for (k = n - 1; k >= phi_start; k--) {
                DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau + eta, lambda_out, delta);
                double t = z[k] / delta[k];
                phi_val += z[k] * t;
                dphi += t * t;
                erretm += phi_val;
            }

            {
                DLAED4_CHECK_DELTA_ZERO(delta[ii], origin, n, d, tau + eta, lambda_out, delta);
                double temp_ii = z[ii] / delta[ii];
                dw = dpsi + dphi + temp_ii * temp_ii;
                double temp_contrib = z[ii] * temp_ii;
                w_val = rhoinv + phi_val + psi_val + temp_contrib;
                erretm = 8.0 * (phi_val - psi_val) + erretm + 2.0 * fabs(rhoinv)
                       + 3.0 * fabs(temp_contrib) + fabs(tau + eta) * dw;
            }

            /* SWTCH detection (LAPACK lines 722-728) */
            if (orgati) {
                if (-w_val > fabs(prew) / 10.0) swtch = 1;
            } else {
                if (w_val > fabs(prew) / 10.0) swtch = 1;
            }

            tau += eta;
        }

        /* ---- Main iteration loop (LAPACK lines 736-944) ---- */
        for (int iter = 2; iter < MAXIT; iter++) {
            /* Test for convergence */
            if (fabs(w_val) <= EPS * erretm) {
                *lambda_out = d[origin] + tau;
                for (k = 0; k < n; k++)
                    delta[k] = (d[k] - d[origin]) - tau;
                return 0;
            }

            /* Update bracket */
            if (w_val <= 0.0) dltlb = fmax(dltlb, tau);
            else              dltub = fmin(dltub, tau);

            /* Compute step (LAPACK lines 752-887) */
            double eta;
            DLAED4_CHECK_DELTA_ZERO(delta[ii], origin, n, d, tau, lambda_out, delta);
            dw = dpsi + dphi + (z[ii] / delta[ii]) * (z[ii] / delta[ii]);

            if (!swtch3) {
                double C_val;
                if (!swtch) {
                    /* Normal case: C from gap-bounding poles */
                    if (orgati) {
                        DLAED4_CHECK_DELTA_ZERO(delta[i], origin, n, d, tau, lambda_out, delta);
                        C_val = w_val - delta[i + 1] * dw
                              - (d[i] - d[i + 1]) * (z[i] / delta[i]) * (z[i] / delta[i]);
                    } else {
                        DLAED4_CHECK_DELTA_ZERO(delta[i + 1], origin, n, d, tau, lambda_out, delta);
                        C_val = w_val - delta[i] * dw
                              - (d[i + 1] - d[i]) * (z[i + 1] / delta[i + 1]) * (z[i + 1] / delta[i + 1]);
                    }
                } else {
                    /* SWTCH: move II-th pole to PSI or PHI side.
                     * LAPACK lines 770-781. */
                    DLAED4_CHECK_DELTA_ZERO(delta[ii], origin, n, d, tau, lambda_out, delta);
                    double t_ii = z[ii] / delta[ii];
                    if (orgati) {
                        /* Move II-th pole into PSI */
                        double dpsi_adj = dpsi + t_ii * t_ii;
                        C_val = w_val - delta[i] * dpsi_adj - delta[i + 1] * dphi;
                    } else {
                        /* Move II-th pole into PHI */
                        double dphi_adj = dphi + t_ii * t_ii;
                        C_val = w_val - delta[i] * dpsi - delta[i + 1] * dphi_adj;
                    }
                }

                double A_val = (delta[i] + delta[i + 1]) * w_val
                             - delta[i] * delta[i + 1] * dw;
                double B_val = delta[i] * delta[i + 1] * w_val;

                if (C_val == 0.0) {
                    if (A_val == 0.0) {
                        if (!swtch) {
                            if (orgati)
                                A_val = z[i] * z[i] + delta[i + 1] * delta[i + 1] * (dpsi + dphi);
                            else
                                A_val = z[i + 1] * z[i + 1] + delta[i] * delta[i] * (dpsi + dphi);
                        } else {
                            A_val = delta[i] * delta[i] * dpsi
                                  + delta[i + 1] * delta[i + 1] * dphi;
                        }
                    }
                    eta = B_val / A_val;
                } else if (A_val <= 0.0) {
                    eta = (A_val - sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val))) / (2.0 * C_val);
                } else {
                    eta = 2.0 * B_val / (A_val + sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val)));
                }
            } else {
                /* SWTCH3: three-pole interpolation via dlaed6.
                 * LAPACK lines 845-887. */
                double temp_rpf = rhoinv + psi_val + phi_val;
                double zz[3];
                double C_3;

                if (swtch) {
                    /* SWTCH + SWTCH3: simplified form (LAPACK lines 848-851) */
                    C_3 = temp_rpf - delta[iim1] * dpsi - delta[iip1] * dphi;
                    zz[0] = delta[iim1] * delta[iim1] * dpsi;
                    zz[2] = delta[iip1] * delta[iip1] * dphi;
                } else {
                    if (orgati) {
                        DLAED4_CHECK_DELTA_ZERO(delta[iim1], origin, n, d, tau, lambda_out, delta);
                        double t1 = z[iim1] / delta[iim1];
                        t1 = t1 * t1;
                        C_3 = temp_rpf - delta[iip1] * (dpsi + dphi)
                            - (d[iim1] - d[iip1]) * t1;
                        zz[0] = z[iim1] * z[iim1];
                        zz[2] = delta[iip1] * delta[iip1] * ((dpsi - t1) + dphi);
                    } else {
                        DLAED4_CHECK_DELTA_ZERO(delta[iip1], origin, n, d, tau, lambda_out, delta);
                        double t1 = z[iip1] / delta[iip1];
                        t1 = t1 * t1;
                        C_3 = temp_rpf - delta[iim1] * (dpsi + dphi)
                            - (d[iip1] - d[iim1]) * t1;
                        zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - t1));
                        zz[2] = z[iip1] * z[iip1];
                    }
                }
                zz[1] = z[ii] * z[ii];

                double d3[3] = { delta[iim1], delta[ii], delta[iip1] };
                int info6;
                dlaed6(40, orgati, d3, zz, C_3, &eta, &info6);
                if (info6 != 0) {
                    /* dlaed6 failed — fall back to Newton step */
                    eta = -w_val / dw;
                }
            }

            /* Sign check (LAPACK line 895) */
            if (w_val * eta >= 0.0)
                eta = -w_val / dw;

            /* Geometric mean safeguard (LAPACK lines 897-917) */
            {
                double temp_tau = tau + eta;
                if (temp_tau > dltub || temp_tau < dltlb) {
                    double eta1 = -w_val / dw;
                    double temp_tau1 = tau + eta1;
                    double eta2;
                    if (w_val < 0.0)
                        eta2 = (dltub - tau) / 2.0;
                    else
                        eta2 = (dltlb - tau) / 2.0;
                    if (dltlb <= temp_tau1 && temp_tau1 <= dltub) {
                        eta = copysign(1.0, eta1) * sqrt(fabs(eta1)) * sqrt(fabs(eta2));
                    } else {
                        eta = eta2;
                    }
                }
            }

            /* Apply step */
            for (k = 0; k < n; k++)
                delta[k] -= eta;
            tau += eta;
            prew = w_val;

            /* Re-evaluate PSI/PHI/W (LAPACK lines 924-944) */
            dpsi = 0.0; psi_val = 0.0; erretm = 0.0;
            for (k = 0; k < psi_end; k++) {
                DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
                double t = z[k] / delta[k];
                psi_val += z[k] * t;
                dpsi += t * t;
                erretm += psi_val;
            }
            erretm = fabs(erretm);

            dphi = 0.0; phi_val = 0.0;
            for (k = n - 1; k >= phi_start; k--) {
                DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
                double t = z[k] / delta[k];
                phi_val += z[k] * t;
                dphi += t * t;
                erretm += phi_val;
            }

            {
                DLAED4_CHECK_DELTA_ZERO(delta[ii], origin, n, d, tau, lambda_out, delta);
                double t = z[ii] / delta[ii];
                double temp_contrib = z[ii] * t;
                dw = dpsi + dphi + t * t;
                w_val = rhoinv + phi_val + psi_val + temp_contrib;
                erretm = 8.0 * (phi_val - psi_val) + erretm + 2.0 * fabs(rhoinv)
                       + 3.0 * fabs(temp_contrib) + fabs(tau) * dw;
            }

            /* SWTCH update (LAPACK line 943) */
            if (w_val * prew > 0.0 && fabs(w_val) > fabs(prew) / 10.0)
                swtch = !swtch;
        }

        /* Did not converge — return best estimate.
         * Recompute delta from final tau to avoid accumulated incremental error. */
        *lambda_out = d[origin] + tau;
        for (k = 0; k < n; k++)
            delta[k] = (d[k] - d[origin]) - tau;
        return 1;

    } else {
        /* ============================================================
         * Last eigenvalue (i = n-1): root above d[n-1]
         * LAPACK dlaed4.f lines 175-440 (I=N case).
         *
         * Structure: PSI k=0..n-2 (II=n-2 in 0-based), centre k=n-1.
         * A/B/C rational step uses DELTA(N-1) and DELTA(N) in LAPACK
         * 1-based = delta[n-2] and delta[n-1] in C.
         * ============================================================ */
        npy_intp origin = n - 1;
        npy_intp ii_last = n - 2;  /* LAPACK II = N-1 (1-based) = n-2 (0-based) */

        /* LAPACK: initial guess via midpoint evaluation.
         * Delta at midpoint rho/2: delta[k] = (d[k] - d[n-1]) - rho/2 */
        double midpt = rho / 2.0;
        for (k = 0; k < n; k++)
            delta[k] = (d[k] - d[n - 1]) - midpt;

        double psi_val = 0.0;
        for (k = 0; k < n - 2; k++)
            psi_val += z[k] * z[k] / delta[k];
        double c = rhoinv + psi_val;
        double w = c + z[n - 2] * z[n - 2] / delta[n - 2]
                     + z[n - 1] * z[n - 1] / delta[n - 1];

        double tau, dltlb, dltub;
        double del_last = d[n - 1] - d[n - 2];

        if (w <= 0.0) {
            /* Root in [d[n-1]+rho/2, d[n-1]+rho).
             * LAPACK: special handling for W<=0 at midpoint. */
            double temp_w = z[n - 2] * z[n - 2] / (d[n - 1] - d[n - 2] + rho)
                          + z[n - 1] * z[n - 1] / rho;
            if (c <= temp_w) {
                tau = rho;
            } else {
                double a = -c * del_last + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
                double b = z[n - 1] * z[n - 1] * del_last;
                if (a < 0.0)
                    tau = 2.0 * b / (sqrt(a * a + 4.0 * b * c) - a);
                else
                    tau = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);
            }
            dltlb = midpt;
            dltub = rho;
        } else {
            /* Root in (d[n-1], d[n-1]+rho/2].
             * Standard quadratic. */
            double a = -c * del_last + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
            double b = z[n - 1] * z[n - 1] * del_last;
            if (a < 0.0)
                tau = 2.0 * b / (sqrt(a * a + 4.0 * b * c) - a);
            else
                tau = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);
            dltlb = 0.0;
            dltub = midpt;
        }

        /* Clamp tau to bracket */
        if (tau <= dltlb) tau = dltlb + EPS * (dltub - dltlb);
        if (tau >= dltub) tau = dltub * 0.5;

        /* Initialize delta with tau */
        for (k = 0; k < n; k++)
            delta[k] = (d[k] - d[n - 1]) - tau;

        /* First evaluation: PSI (k=0..n-2), PHI (k=n-1 only).
         * LAPACK dlaed4.f lines 270-300. */
        double dpsi = 0.0;
        psi_val = 0.0;
        double erretm = 0.0;
        for (k = 0; k < ii_last + 1; k++) {
            DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
            double temp = z[k] / delta[k];
            psi_val += z[k] * temp;
            dpsi += temp * temp;
            erretm += psi_val;  /* running sum like LAPACK */
        }
        erretm = fabs(erretm);

        DLAED4_CHECK_DELTA_ZERO(delta[n - 1], origin, n, d, tau, lambda_out, delta);
        double temp_phi = z[n - 1] / delta[n - 1];
        double phi_val = z[n - 1] * temp_phi;
        double dphi = temp_phi * temp_phi;
        erretm = 8.0 * (-phi_val - psi_val) + erretm - phi_val
               + fabs(rhoinv) + fabs(tau) * (dpsi + dphi);

        w = rhoinv + phi_val + psi_val;

        /* Test for convergence */
        if (fabs(w) <= EPS * erretm) {
            *lambda_out = d[n - 1] + tau;
            for (k = 0; k < n; k++)
                delta[k] = (d[k] - d[n - 1]) - tau;
            return 0;
        }

        if (w <= 0.0) dltlb = fmax(dltlb, tau);
        else          dltub = fmin(dltub, tau);

        /* First step: A/B/C rational interpolation.
         * LAPACK dlaed4.f lines 310-370 (NITER=2 step).
         * Uses delta[n-2] and delta[n-1] as the two nearest poles. */
        {
            double C_val = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
            double A_val = (delta[n - 2] + delta[n - 1]) * w
                         - delta[n - 2] * delta[n - 1] * (dpsi + dphi);
            double B_val = delta[n - 2] * delta[n - 1] * w;
            double eta;

            /* LAPACK: if C < 0, C = |C| */
            if (C_val < 0.0) C_val = -C_val;

            if (C_val == 0.0) {
                eta = -w / (dpsi + dphi);
            } else if (A_val >= 0.0) {
                eta = (A_val + sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val))) / (2.0 * C_val);
            } else {
                eta = 2.0 * B_val / (A_val - sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val)));
            }

            /* Sign check */
            if (w * eta > 0.0)
                eta = -w / (dpsi + dphi);

            /* Safeguard for first step: geometric mean (LAPACK lines 340-360) */
            {
                double temp_tau = tau + eta;
                if (temp_tau > dltub || temp_tau < dltlb) {
                    double eta1 = -w / (dpsi + dphi);
                    double temp_tau1 = tau + eta1;
                    double eta2;
                    if (w < 0.0)
                        eta2 = (dltub - tau) / 2.0;
                    else
                        eta2 = (dltlb - tau) / 2.0;
                    if (dltlb <= temp_tau1 && temp_tau1 <= dltub) {
                        eta = copysign(1.0, eta1) * sqrt(fabs(eta1)) * sqrt(fabs(eta2));
                    } else {
                        eta = eta2;
                    }
                }
            }

            /* Apply step */
            for (k = 0; k < n; k++)
                delta[k] -= eta;
            tau += eta;

            /* Re-evaluate PSI/PHI (LAPACK lines 375-400) */
            dpsi = 0.0; psi_val = 0.0; erretm = 0.0;
            for (k = 0; k < ii_last + 1; k++) {
                DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
                double t = z[k] / delta[k];
                psi_val += z[k] * t;
                dpsi += t * t;
                erretm += psi_val;
            }
            erretm = fabs(erretm);

            DLAED4_CHECK_DELTA_ZERO(delta[n - 1], origin, n, d, tau, lambda_out, delta);
            temp_phi = z[n - 1] / delta[n - 1];
            phi_val = z[n - 1] * temp_phi;
            dphi = temp_phi * temp_phi;
            erretm = 8.0 * (-phi_val - psi_val) + erretm - phi_val
                   + fabs(rhoinv) + fabs(tau) * (dpsi + dphi);

            w = rhoinv + phi_val + psi_val;
        }

        /* ---- Main iteration loop (LAPACK lines 405-440) ---- */
        for (int iter = 2; iter < MAXIT; iter++) {
            /* Test for convergence */
            if (fabs(w) <= EPS * erretm) {
                *lambda_out = d[n - 1] + tau;
                for (k = 0; k < n; k++)
                    delta[k] = (d[k] - d[n - 1]) - tau;
                return 0;
            }

            if (w <= 0.0) dltlb = fmax(dltlb, tau);
            else          dltub = fmin(dltub, tau);

            /* A/B/C rational step (LAPACK lines 415-425) */
            double C_val = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
            double A_val = (delta[n - 2] + delta[n - 1]) * w
                         - delta[n - 2] * delta[n - 1] * (dpsi + dphi);
            double B_val = delta[n - 2] * delta[n - 1] * w;
            double eta;

            /* LAPACK: if C < 0, C = |C| */
            if (C_val < 0.0) C_val = -C_val;

            if (C_val == 0.0) {
                eta = -w / (dpsi + dphi);
            } else if (A_val >= 0.0) {
                eta = (A_val + sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val))) / (2.0 * C_val);
            } else {
                eta = 2.0 * B_val / (A_val - sqrt(fabs(A_val * A_val - 4.0 * B_val * C_val)));
            }

            /* Sign check */
            if (w * eta > 0.0)
                eta = -w / (dpsi + dphi);

            /* Safeguard: simple bisection for main loop (LAPACK lines 430-435) */
            {
                double temp_tau = tau + eta;
                if (temp_tau > dltub || temp_tau < dltlb) {
                    if (w < 0.0)
                        eta = (dltub - tau) / 2.0;
                    else
                        eta = (dltlb - tau) / 2.0;
                }
            }

            /* Apply step */
            for (k = 0; k < n; k++)
                delta[k] -= eta;
            tau += eta;

            /* Re-evaluate PSI/PHI */
            dpsi = 0.0; psi_val = 0.0; erretm = 0.0;
            for (k = 0; k < ii_last + 1; k++) {
                DLAED4_CHECK_DELTA_ZERO(delta[k], origin, n, d, tau, lambda_out, delta);
                double t = z[k] / delta[k];
                psi_val += z[k] * t;
                dpsi += t * t;
                erretm += psi_val;
            }
            erretm = fabs(erretm);

            DLAED4_CHECK_DELTA_ZERO(delta[n - 1], origin, n, d, tau, lambda_out, delta);
            temp_phi = z[n - 1] / delta[n - 1];
            phi_val = z[n - 1] * temp_phi;
            dphi = temp_phi * temp_phi;
            erretm = 8.0 * (-phi_val - psi_val) + erretm - phi_val
                   + fabs(rhoinv) + fabs(tau) * (dpsi + dphi);

            w = rhoinv + phi_val + psi_val;
        }

        /* Did not converge */
        *lambda_out = d[n - 1] + tau;
        for (k = 0; k < n; k++)
            delta[k] = (d[k] - d[n - 1]) - tau;
        return 1;
    }
}

/* ---------------------------------------------------------------------------
 * GEMM dispatch helper: uses workspace-explicit path when ws is available,
 * otherwise falls back to global-mutex path.  Avoids repeating the
 * ws ? ext_ws : ext pattern at every GEMM call site in merge_rank1.
 * ---------------------------------------------------------------------------
 */
static void dgemm_dispatch(npy_intp M, npy_intp N, npy_intp K,
                            const double *A, npy_intp lda,
                            const double *B, npy_intp ldb,
                            double *C, npy_intp ldc,
                            int transa, int transb,
                            double alpha, double beta,
                            jblas_workspace_t *ws)
{
    if (ws) {
        jblas_dgemm_ext_ws(M, N, K, A, lda, B, ldb, C, ldc,
                           transa, transb, alpha, beta, ws);
    } else {
        /* Global-mutex path: jblas_dgemm_ext only supports alpha=1, beta=0.
         * All merge_rank1 calls use those values. */
        if (alpha != 1.0 || beta != 0.0) {
            fprintf(stderr, "FATAL: dgemm_dispatch: non-ws path requires "
                    "alpha=1, beta=0 but got alpha=%.1f, beta=%.1f\n",
                    alpha, beta);
            abort();
        }
        jblas_dgemm_ext(M, N, K, A, lda, B, ldb, C, ldc, transa, transb);
    }
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
                          double *merge_scratch,
                          jblas_eigh_status_t *status);

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
                       double *merge_scratch,
                       jblas_eigh_status_t *status)
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
            /* dlaed4 already stored its best estimate in lam_nd[i] and
             * delta_mat row i.  Keep them — they are closer to the true root
             * than the pole value d_nd[i].  The residual check in
             * jblas_dstedc_c will trigger QR fallback if results are bad. */
        }
    }
    if (n_secular_failures > 0) {
        fprintf(stderr, "jblas dstedc: %d/%ld secular equation(s) "
                "failed to converge at merge size %ld — using best estimates "
                "(residual check will trigger QR fallback if needed)\n",
                n_secular_failures, (long)n_nd, (long)n);
        if (status) status->secular_failures += n_secular_failures;
    }

    /* Step 5: Secular eigenvectors (LAPACK dlaed3 algorithm).
     *
     * Uses delta vectors from dlaed4 to avoid precision loss.
     *
     * For each pole k, compute weight W[k]:
     *   W[k] = delta_mat[k][k]  (= d[k] - lam[k], the "own" gap)
     *   then for each j != k:
     *     numerator:   delta_mat[j][k]  (= d[k] - lam[j], from dlaed4)
     *     denominator: delta_mat[j][k] - delta_mat[j][j]
     *       Algebraically = d[k] - d[j], but computed via dlaed4 deltas
     *       to avoid catastrophic cancellation when d[k] ~ d[j].
     *       This is the LAPACK dlaed3 technique (reference: dlaed3.f).
     *   W[k] = sgn(z[k]) * sqrt(-W[k])   (product MUST be negative by
     *          interlacing theorem; positive signals a precision issue)
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
            /* delta_mat[j][k] = d[k] - lam[j] (from dlaed4, full precision)
             * den = delta_mat[j][k] - delta_mat[j][j]
             *     = (d[k] - lam[j]) - (d[j] - lam[j])
             *     = d[k] - d[j]   algebraically
             * but computed via dlaed4 deltas for numerical stability
             * when d[k] ~ d[j] (LAPACK dlaed3 technique). */
            double num = delta_mat[j * n_nd + k];
            double den = delta_mat[j * n_nd + k] - delta_mat[j * n_nd + j];
            if (fabs(den) < DBL_MIN) {
                /* Near-duplicate eigenvalues that deflation missed.
                 * Clamp to DBL_MIN preserving sign to avoid Inf/NaN
                 * while keeping the factor's directional contribution. */
                fprintf(stderr, "jblas dlaed3: near-zero denominator at k=%ld j=%ld "
                        "(|den|=%.2e, deflation gap too tight)\n",
                        (long)k, (long)j, fabs(den));
                den = copysign(DBL_MIN, den != 0.0 ? den : 1.0);
            }
            w *= num / den;
        }
        /* Product MUST be negative (eigenvalue interlacing theorem).
         * A positive product indicates a bug in dlaed4 deltas.
         * Small positive values from FP rounding are tolerated with sqrt(fabs),
         * but large positive values trigger a diagnostic warning. */
        if (w > 0.0) {
            /* FP rounding can make the product slightly positive.
             * Warn if it's large enough to indicate a real problem. */
            if (w > 1e-100) {
                fprintf(stderr, "jblas dlaed3: positive weight product w=%.2e "
                        "at pole k=%ld (n_nd=%ld) -- interlacing violated\n",
                        w, (long)k, (long)n_nd);
            }
            W_nd[k] = copysign(sqrt(fabs(w)), z_nd[k]);
        } else {
            W_nd[k] = copysign(sqrt(-w), z_nd[k]);
        }
    }

    for (npy_intp i = 0; i < n_nd; i++) {
        double norm2 = 0.0;
        for (npy_intp k = 0; k < n_nd; k++) {
            /* Delta from dlaed4 is in-bracket, never zero */
            double dk = delta_mat[i * n_nd + k];
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

        dgemm_dispatch(n, n_nd, n_nd,
                       Q_b_cols, n_nd, Q_nd, n_nd, Q_nd_full, n_nd,
                       0, 0, 1.0, 0.0, ws);

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

    dgemm_dispatch(n, n, n,
                   Z, ldz, Q_sec, n, Z_new, ldz,
                   0, 0, 1.0, 0.0, ws);

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
                          double *merge_scratch,
                          jblas_eigh_status_t *status)
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
                          ws, merge_scratch, status);
    if (ret != 0) return ret;

    /* Right half: rows m..n-1, cols m..n-1 of Z */
    ret = dstedc_recurse(n - m, d + m, e + m, Z + m * ldz + m, ldz,
                          work, lwork, iwork, ws, merge_scratch, status);
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
    /* Left half: scale only (no sign flip).
     * Right half: scale AND apply sign_rho.
     * Matches LAPACK DLAED2: IF(RHO.LT.ZERO) CALL DSCAL(N2,-ONE,Z(N1+1),1)
     * Only the right half (Q_R rows) gets negated when rho < 0.
     * Applying sign_rho to ALL of z preserves z^2 (eigenvalues correct)
     * but flips cross-terms z[i]*z[j] for i<m, j>=m, corrupting
     * eigenvectors with residuals of 0.05-0.13 at N>=128. */
    for (npy_intp j = 0; j < m; j++)
        z_vec[j] *= inv_sqrt2;
    for (npy_intp j = m; j < n; j++)
        z_vec[j] *= inv_sqrt2 * sign_rho;

    /* If rho is zero, the two sub-problems are decoupled */
    if (rho == 0.0) {
        free(z_vec);
        sort_eig(d, Z, ldz, n);
        return 0;
    }

    /* Merge */
    ret = merge_rank1(n, m, d, z_vec, rho, Z, ldz, work, lwork, iwork,
                      ws, merge_scratch, status);
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
                   jblas_workspace_t *ws,
                   jblas_eigh_status_t *status)
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
                              ws, merge_scratch, status);

    /* Check if D&C result needs QR fallback.
     *
     * With LAPACK-quality dlaed4 (ORGATI + SWTCH3/dlaed6) producing
     * full-precision delta vectors, and the LAPACK multi-pass weight
     * product (delta_mat subtraction for denominators), D&C achieves
     * good results for small N.  At larger N (>= ~100), the O(n)
     * error accumulation in the n-1 ratio weight product can still
     * produce elevated residuals.  QR fallback is an emergency-only
     * safety net for these cases and for pathological inputs.
     *
     * QR fallback triggers when:
     *   (a) dstedc_recurse returned non-zero (allocation or convergence
     *       failure), OR
     *   (b) the tridiagonal residual exceeds 1e-8.
     *
     * Residuals between 1e-14 and 1e-8 produce a diagnostic warning but
     * do not trigger fallback.  The O(N^2) residual check is acceptable
     * since dstedc is already O(N^2 log N) for D&C. */
    int need_fallback = 0;
    if (ret != 0) {
        fprintf(stderr, "jblas dstedc: D&C returned %d (N=%ld), "
                "attempting QR fallback\n", ret, (long)N);
        need_fallback = 1;
    } else {
        double resid = tridiag_eig_residual(N, d_orig, e_orig, d, Z, ldz);
        if (resid > 1e-8) {
            fprintf(stderr, "jblas dstedc: D&C residual %.2e (N=%ld), "
                    "attempting QR fallback\n", resid, (long)N);
            need_fallback = 1;
        } else if (resid > 1e-14) {
            fprintf(stderr, "jblas dstedc: D&C residual %.2e (N=%ld) -- "
                    "above machine epsilon but below QR threshold\n",
                    resid, (long)N);
        }
    }

    if (need_fallback) {
        /* Skip QR retry if D&C failed due to allocation error — QR will
         * also fail with the same malloc limitation. */
        if (ret == -1) {
            fprintf(stderr, "jblas dstedc: D&C allocation failure (N=%ld), "
                    "skipping QR fallback (same allocation will fail)\n",
                    (long)N);
        } else {
            if (status) status->qr_fallback = 1;
            if (N > 2000) {
                fprintf(stderr, "jblas dstedc: QR fallback on N=%ld — "
                        "this is O(N^3) and may take a very long time\n",
                        (long)N);
            }
            memcpy(d, d_orig, (size_t)N * sizeof(double));
            memcpy(e, e_orig, (size_t)(N - 1) * sizeof(double));
            memset(Z, 0, (size_t)N * (size_t)ldz * sizeof(double));
            for (npy_intp k = 0; k < N; k++)
                Z[k * ldz + k] = 1.0;
            ret = dsteqr_base(N, d, e, Z, ldz);
        }
    }

    free(work);
    free(iwork);
    free(d_orig);
    free(e_orig);
    free(merge_scratch);

    /* Final sort: D&C merge and QR fallback both sort internally, but
     * the QR fallback path re-runs dsteqr_base on the full matrix which
     * may not produce globally sorted output when N > DSTEDC_BASE. */
    if (ret == 0)
        sort_eig(d, Z, ldz, N);

    return ret;
}
