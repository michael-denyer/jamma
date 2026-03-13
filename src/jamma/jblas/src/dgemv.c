/**
 * dgemv.c — Double-precision matrix-vector product y = A*x (BLAS Level 2).
 *
 * A is row-major (m x n), x is length-n, y is length-m output.
 * No alpha/beta/transpose for this internal primitive.
 *
 * Implements:
 *   jblas_dgemv_generic — row-by-row calls to jblas_ddot_generic
 *   jblas_dgemv_avx2    — row-by-row calls to jblas_ddot_avx2
 *
 * Strategy: each row of y is a dot product of the corresponding row of A with
 * x.  Both implementations call their ddot directly (not through the dispatch
 * table) so they are self-contained — no dependency on jblas_init().
 */

#include <string.h>  /* memset */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  /* npy_intp */
#include "jblas.h"

/* ---------------------------------------------------------------------------
 * Generic implementation — row-by-row dispatch through ddot
 * ---------------------------------------------------------------------------
 */
void jblas_dgemv_generic(
        npy_intp m, npy_intp n,
        const double *A,
        const double *x,
        double       *y)
{
    if (m <= 0)
        return;
    if (n <= 0) {
        memset(y, 0, (size_t)m * sizeof(double));
        return;
    }

    for (npy_intp i = 0; i < m; i++)
        y[i] = jblas_ddot_generic(n, A + i * n, 1, x, 1);
}

/* ---------------------------------------------------------------------------
 * AVX2 implementation (x86_64 only) — row-by-row jblas_ddot_avx2
 *
 * Same structure as generic but calls jblas_ddot_avx2 directly for SIMD
 * speedup on each row dot product.
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__)

void jblas_dgemv_avx2(
        npy_intp m, npy_intp n,
        const double *A,
        const double *x,
        double       *y)
{
    if (m <= 0)
        return;
    if (n <= 0) {
        memset(y, 0, (size_t)m * sizeof(double));
        return;
    }

    for (npy_intp i = 0; i < m; i++)
        y[i] = jblas_ddot_avx2(n, A + i * n, 1, x, 1);
}

#endif /* __x86_64__ */
