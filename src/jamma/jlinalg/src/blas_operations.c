/**
 * blas_operations.c -- Operations over the backend selected by blas_dispatch.c.
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blas_dispatch_internal.h"

static inline const blas_candidate_t *active_backend(void) {
    return blas_dispatch_active();
}

/* ---------------------------------------------------------------------------
 * jlinalg_dsyrk_ext — Vendor-dispatch dsyrk: C = X @ X.T + beta*C
 * ---------------------------------------------------------------------------
 */
void jlinalg_dsyrk_ext(npy_intp N, npy_intp K, const double *X, npy_intp ldx, double *C,
                       npy_intp ldc, double beta) {
    if (N <= 0) return;
    if (K <= 0) {
        for (npy_intp i = 0; i < N; i++) {
            if (beta == 0.0)
                memset(C + i * ldc, 0, (size_t)N * sizeof(*C));
            else
                for (npy_intp j = 0; j <= i; j++)
                    C[i * ldc + j] *= beta;
        }
        for (npy_intp i = 0; i < N; i++)
            for (npy_intp j = i + 1; j < N; j++)
                C[i * ldc + j] = C[j * ldc + i];
        return;
    }
    if (active_backend()->is_ilp64) {
        if (active_backend()->cblas_dsyrk_ilp64) {
            /* Row-major, lower, no-trans: C = X @ X.T + beta * C */
            active_backend()->cblas_dsyrk_ilp64(JLINALG_CblasRowMajor, JLINALG_CblasLower,
                                                JLINALG_CblasNoTrans, (long)N, (long)K, 1.0, X,
                                                (long)ldx, beta, C, (long)ldc);
            /* Mirror lower to upper (vendor only fills lower) */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++)
                    C[i * ldc + j] = C[j * ldc + i];
            return;
        }
        /* Fortran ILP64 fallback: row-major lower = col-major upper */
        if (active_backend()->dsyrk_ilp64) {
            const long long n = (long long)N, k = (long long)K;
            const long long lda = (long long)ldx, ldc_f = (long long)ldc;
            const double alpha = 1.0;
            active_backend()->dsyrk_ilp64("U", "T", &n, &k, &alpha, X, &lda, &beta, C, &ldc_f);
            /* Fortran col-major upper = row-major lower; mirror lower to upper */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++)
                    C[i * ldc + j] = C[j * ldc + i];
            return;
        }
    }
    /* No vendor dsyrk available -- caller should use numpy fallback. */
    fprintf(stderr, "FATAL: jlinalg_dsyrk_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}

/* ---------------------------------------------------------------------------
 * jlinalg_dsyevd_ext — Vendor-dispatch dsyevd for eigh
 *
 * Prefers LAPACKE C interface (row-major, no transpose) when available (MKL).
 * Falls back to Fortran dsyevd + eigenvector transpose (Accelerate, OpenBLAS).
 *
 * Input: K is row-major symmetric, lower triangle populated.
 * Output: K overwritten with eigenvectors stored columnwise in row-major
 *         (K[i*ldk+j] = component i of eigenvector j).
 *         eigenvalues[k] = k-th eigenvalue, ascending.
 *
 * Returns: JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL,
 *          or positive int for LAPACK error (info capped to INT_MAX for ILP64).
 * ---------------------------------------------------------------------------
 */

/* Safely narrow LAPACK info (long long) to int return.  Logs the full value
 * when truncation would occur (ILP64 eigenvalue index > INT_MAX). */
static int _info_to_int(long long info, npy_intp N) {
    if (info > INT_MAX || info < INT_MIN) {
        fprintf(stderr,
                "jlinalg: LAPACK info=%lld exceeds int range (N=%ld) "
                "— returning capped value\n",
                info, (long)N);
        return info > 0 ? INT_MAX : INT_MIN;
    }
    return (int)info;
}

int jlinalg_dsyevd_ext(npy_intp N, double *K, npy_intp ldk, double *eigenvalues) {
    if (!blas_has_dsyevd() || !active_backend()->is_ilp64) return JLINALG_EXT_UNAVAILABLE;

    /* --- LAPACKE path (MKL): row-major natively, no transpose needed.
     * Only used when Fortran ILP64 dsyevd is NOT available.  When both
     * exist, we prefer Fortran because dsyevd_64_ is an unambiguous ILP64
     * symbol, whereas LAPACKE_dsyevd is unsuffixed and could resolve to
     * the LP64 variant on systems with mixed LP64/ILP64 MKL. --- */
    if (active_backend()->lapacke_dsyevd_ilp64 && !active_backend()->dsyevd_ilp64) {
        long long info = active_backend()->lapacke_dsyevd_ilp64(
            JLINALG_LAPACK_ROW_MAJOR, 'V', 'L', (long long)N, K, (long long)ldk, eigenvalues);
        if (info != 0) return _info_to_int(info, N);
        return JLINALG_EXT_SUCCESS;
    }

    /* --- Fortran path (Accelerate, MKL, OpenBLAS): col-major + transpose.
     * Preferred when available because ILP64 symbol names (dsyevd_64_,
     * dsyevd$NEWLAPACK$ILP64) are unambiguous — no LP64/ILP64 confusion. --- */
    if (active_backend()->dsyevd_ilp64) {
        long long n = (long long)N;
        long long lda = (long long)ldk;
        long long info = 0;

        /* Workspace query */
        long long lwork = -1, liwork = -1;
        double work_query;
        long long iwork_query;
        active_backend()->dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues, &work_query, &lwork,
                                       &iwork_query, &liwork, &info);
        if (info != 0) {
            fprintf(stderr,
                    "jlinalg_dsyevd_ext: Fortran dsyevd workspace query failed "
                    "(info=%lld, N=%lld) — likely ABI mismatch or corrupt LAPACK\n",
                    info, n);
            return (int)info;
        }

        lwork = (long long)work_query + 1; /* +1 for double→integer rounding */
        liwork = iwork_query;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        long long *iwork = (long long *)malloc((size_t)liwork * sizeof(long long));
        if (!work || !iwork) {
            /* CRITICAL: ALLOC_FAIL must be returned BEFORE K is modified.
             * eigh.c relies on K being unmodified when K == eigenvectors
             * so it can fall through to DSYEVR with the original data. */
            free(work);
            free(iwork);
            return JLINALG_EXT_ALLOC_FAIL;
        }

        /* Compute: UPLO='U' because row-major lower = col-major upper.
         * The matrix is symmetric so A = A^T — no input transpose needed,
         * just the UPLO swap. */
        active_backend()->dsyevd_ilp64("V", "U", &n, K, &lda, eigenvalues, work, &lwork, iwork,
                                       &liwork, &info);
        free(work);
        free(iwork);
        if (info != 0) return _info_to_int(info, N);

        /* Transpose eigenvectors: Fortran dsyevd writes eigenvectors as
         * columns in column-major layout.  In our row-major memory, those
         * columns appear as rows.  Transpose to get the standard row-major
         * columnwise convention (K[i*ldk+j] = component i of eigvec j). */
        for (npy_intp i = 0; i < N; i++)
            for (npy_intp j = i + 1; j < N; j++) {
                double tmp = K[i * ldk + j];
                K[i * ldk + j] = K[j * ldk + i];
                K[j * ldk + i] = tmp;
            }
        return JLINALG_EXT_SUCCESS;
    }
    return JLINALG_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * jlinalg_dsyevr_ext — Vendor-dispatch dsyevr for eigh (memory-pressure fallback)
 *
 * DSYEVR uses O(N) workspace vs O(N^2) for DSYEVD.  Eigenvectors are written
 * into a separate Z output buffer (does not require an N x N copy of K).
 *
 * Input: K is row-major symmetric, lower triangle populated (overwritten).
 * Output: eigenvectors in row-major columnwise (Z[i*ldz+j] = component i of eigvec j).
 *         eigenvalues[k] = k-th eigenvalue, ascending.
 *
 * Returns: JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL,
 *          or positive int for LAPACK error (info capped to INT_MAX for ILP64).
 * ---------------------------------------------------------------------------
 */
int jlinalg_dsyevr_ext(npy_intp N, double *K, npy_intp ldk, double *eigenvalues,
                       double *eigenvectors, npy_intp ldz) {
    if (!blas_has_dsyevr()) return JLINALG_EXT_UNAVAILABLE;

    if (active_backend()->dsyevr_ilp64) {
        long long n = (long long)N;
        long long lda = (long long)ldk;
        long long ldz_f = (long long)N; /* tightly packed Z for Fortran */
        long long info = 0;
        long long m_out = 0;       /* number of eigenvalues found */
        double abstol = 0.0;       /* use default (DLAMCH) */
        long long il = 1, iu = n;  /* all eigenvalues (range='A' ignores these) */
        double vl = 0.0, vu = 0.0; /* unused for range='A' */

        /* Workspace query */
        long long lwork = -1, liwork = -1;
        double work_query;
        long long iwork_query;
        long long isuppz_dummy[2];
        active_backend()->dsyevr_ilp64("V", "A", "U", &n, K, &lda, &vl, &vu, &il, &iu, &abstol,
                                       &m_out, eigenvalues, eigenvectors, &ldz_f, isuppz_dummy,
                                       &work_query, &lwork, &iwork_query, &liwork, &info);
        if (info != 0) {
            fprintf(stderr, "jlinalg_dsyevr_ext: workspace query failed (info=%lld, N=%lld)\n",
                    info, n);
            return _info_to_int(info, N);
        }

        lwork = (long long)work_query + 1;
        liwork = iwork_query;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        long long *iwork = (long long *)malloc((size_t)liwork * sizeof(long long));
        long long *isuppz = (long long *)malloc((size_t)(2 * N) * sizeof(long long));
        /* Reuse the caller's tightly packed output buffer when possible. */
        int use_output_as_z = (ldz == N && eigenvectors != K);
        double *Z_col = use_output_as_z ? eigenvectors
                                        : (double *)malloc((size_t)N * (size_t)N * sizeof(double));
        if (!work || !iwork || !isuppz || !Z_col) {
            free(work);
            free(iwork);
            free(isuppz);
            if (!use_output_as_z) free(Z_col);
            return JLINALG_EXT_ALLOC_FAIL;
        }

        /* Compute: UPLO='U' because row-major lower = col-major upper. */
        active_backend()->dsyevr_ilp64("V", "A", "U", &n, K, &lda, &vl, &vu, &il, &iu, &abstol,
                                       &m_out, eigenvalues, Z_col, &ldz_f, isuppz, work, &lwork,
                                       iwork, &liwork, &info);
        free(work);
        free(iwork);
        free(isuppz);
        if (info != 0) {
            if (!use_output_as_z) free(Z_col);
            return _info_to_int(info, N);
        }

        /* Verify all eigenvalues were found (range='A' should always give m_out == N) */
        if (m_out != n) {
            fprintf(stderr,
                    "jlinalg_dsyevr_ext: expected %lld eigenvalues but DSYEVR found %lld "
                    "(range='A', N=%lld) — vendor LAPACK ABI mismatch or bug\n",
                    n, m_out, n);
            if (!use_output_as_z) free(Z_col);
            return JLINALG_EXT_COUNT_MISMATCH;
        }

        if (use_output_as_z) {
            /* DSYEVR wrote col-major data into a tight contiguous buffer.
             * Interpreted as row-major, that is the transpose of what Python
             * expects. Transpose in-place to restore row-major columnwise form. */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = i + 1; j < N; j++) {
                    double tmp = eigenvectors[i * ldz + j];
                    eigenvectors[i * ldz + j] = eigenvectors[j * ldz + i];
                    eigenvectors[j * ldz + i] = tmp;
                }
        } else {
            /* Transpose col-major Z to row-major eigenvectors.
             * Z_col is col-major: Z_col[i + j*N] = component i of eigvec j.
             * eigenvectors is row-major: eigenvectors[i*ldz + j] = component i of eigvec j. */
            for (npy_intp i = 0; i < N; i++)
                for (npy_intp j = 0; j < N; j++)
                    eigenvectors[i * ldz + j] = Z_col[i + j * N];
            free(Z_col);
        }
        return JLINALG_EXT_SUCCESS;
    }

    return JLINALG_EXT_UNAVAILABLE;
}

/* ---------------------------------------------------------------------------
 * Full-signature external dgemm wrapper
 * ---------------------------------------------------------------------------
 */
static int _dgemm_external_full(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                                const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                                int transb, double alpha, double beta) {
    if (active_backend()->cblas_dgemm_ilp64) {
        int ta = transa ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        int tb = transb ? JLINALG_CblasTrans : JLINALG_CblasNoTrans;
        long llda = (long)(lda > 0 ? lda : 1);
        long lldb = (long)(ldb > 0 ? ldb : 1);
        long lldc = (long)(ldc > 0 ? ldc : 1);
        active_backend()->cblas_dgemm_ilp64(JLINALG_CblasRowMajor, ta, tb, (long)M, (long)N,
                                            (long)K, alpha, A, llda, B, lldb, beta, C, lldc);
        return 1;
    }

    /* Fortran ILP64 interface fallback: row-major -> column-major swap.
     * LP64 dgemm is never wired, so the ILP64 pointer is the only path here. */
    const char *transa_f = transb ? "T" : "N";
    const char *transb_f = transa ? "T" : "N";

    const long long lM = (long long)M, lN = (long long)N, lK = (long long)K;
    const long long llda = (long long)lda, lldb = (long long)ldb;
    const long long lldc = (long long)ldc;
    active_backend()->dgemm_ilp64(transa_f, transb_f, &lN, &lM, &lK, &alpha, B, &lldb, A, &llda,
                                  &beta, C, &lldc);
    return 1;
}

/* ---------------------------------------------------------------------------
 * Public full-signature dispatch API
 * ---------------------------------------------------------------------------
 */

void jlinalg_dgemm_ext(npy_intp M, npy_intp N, npy_intp K, const double *A, npy_intp lda,
                       const double *B, npy_intp ldb, double *C, npy_intp ldc, int transa,
                       int transb) {
    if (blas_dispatch_has_vendor_dgemm() &&
        _dgemm_external_full(M, N, K, A, lda, B, ldb, C, ldc, transa, transb, 1.0, 0.0)) {
        return;
    }
    /* No external BLAS wired.
     * Caller should check blas_has_external() and use numpy fallback. */
    fprintf(stderr, "FATAL: jlinalg_dgemm_ext called without vendor BLAS. "
                    "Results would be silently wrong. Aborting.\n");
    abort();
}
