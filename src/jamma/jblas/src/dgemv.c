/**
 * dgemv.c — Double-precision matrix-vector product y = A*x (BLAS Level 2).
 *
 * A is row-major (m x n), x is length-n, y is length-m output.
 * No alpha/beta/transpose for this internal primitive.
 *
 * Implements:
 *   jblas_dgemv_generic — row-by-row dispatch to jblas_dispatch.ddot
 *   jblas_dgemv_avx2    — delegates to generic (ddot AVX2 gives the speedup)
 *
 * Strategy: each row of y is a dot product of the corresponding row of A with
 * x.  By dispatching through jblas_dispatch.ddot, this function automatically
 * benefits from the AVX2 ddot microkernel when available — no separate SIMD
 * path is needed for dgemv itself.
 */

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
    if (m <= 0 || n <= 0)
        return;

    for (npy_intp i = 0; i < m; i++)
        y[i] = jblas_dispatch.ddot(n, A + i * n, 1, x, 1);
}

/* ---------------------------------------------------------------------------
 * AVX2 stub (x86_64 only) — delegates to generic
 *
 * The performance benefit for dgemv comes from the dispatch.ddot call being
 * AVX2-accelerated, not from a separate SIMD path in dgemv itself.
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__)

void jblas_dgemv_avx2(
        npy_intp m, npy_intp n,
        const double *A,
        const double *x,
        double       *y)
{
    jblas_dgemv_generic(m, n, A, x, y);
}

#endif /* __x86_64__ */
