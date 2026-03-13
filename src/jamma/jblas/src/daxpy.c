/**
 * daxpy.c — Double-precision y += alpha*x (BLAS Level 1 daxpy).
 *
 * Implements:
 *   jblas_daxpy_generic — portable scalar with 4x unroll for unit strides
 *   jblas_daxpy_avx2    — x86_64 AVX2 with broadcast alpha and FMA unroll
 *
 * Both variants short-circuit on alpha == 0.0 (no mutation needed).
 * The AVX2 path processes 16 doubles per iteration using four __m256d
 * lanes; strided inputs fall back to generic.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  /* npy_intp */
#include "jblas.h"

/* ---------------------------------------------------------------------------
 * Generic scalar implementation
 * ---------------------------------------------------------------------------
 */
void jblas_daxpy_generic(
        npy_intp n,
        double alpha,
        const double *x, int incx,
        double       *y, int incy)
{
    if (n <= 0 || alpha == 0.0)
        return;

    if (incx == 1 && incy == 1) {
        /* Unit-stride fast path: 4x scalar unroll */
        npy_intp i = 0;
        npy_intp n4 = n - (n % 4);
        for (; i < n4; i += 4) {
            y[i]   += alpha * x[i];
            y[i+1] += alpha * x[i+1];
            y[i+2] += alpha * x[i+2];
            y[i+3] += alpha * x[i+3];
        }
        for (; i < n; i++)
            y[i] += alpha * x[i];
    } else {
        /* Strided slow path */
        for (npy_intp i = 0; i < n; i++)
            y[i * incy] += alpha * x[i * incx];
    }
}

/* ---------------------------------------------------------------------------
 * AVX2 implementation (x86_64 only)
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__)

#include <immintrin.h>

void jblas_daxpy_avx2(
        npy_intp n,
        double alpha,
        const double *x, int incx,
        double       *y, int incy)
{
    if (n <= 0 || alpha == 0.0)
        return;

    /* Strided inputs: fall back to generic */
    if (incx != 1 || incy != 1) {
        jblas_daxpy_generic(n, alpha, x, incx, y, incy);
        return;
    }

    /* Broadcast alpha into all four lanes of a 256-bit register */
    __m256d valpha = _mm256_broadcast_sd(&alpha);

    npy_intp i = 0;
    npy_intp n16 = n - (n % 16);
    for (; i < n16; i += 16) {
        _mm256_storeu_pd(y + i,
            _mm256_fmadd_pd(valpha, _mm256_loadu_pd(x + i),
                            _mm256_loadu_pd(y + i)));
        _mm256_storeu_pd(y + i + 4,
            _mm256_fmadd_pd(valpha, _mm256_loadu_pd(x + i + 4),
                            _mm256_loadu_pd(y + i + 4)));
        _mm256_storeu_pd(y + i + 8,
            _mm256_fmadd_pd(valpha, _mm256_loadu_pd(x + i + 8),
                            _mm256_loadu_pd(y + i + 8)));
        _mm256_storeu_pd(y + i + 12,
            _mm256_fmadd_pd(valpha, _mm256_loadu_pd(x + i + 12),
                            _mm256_loadu_pd(y + i + 12)));
    }

    /* Scalar tail */
    for (; i < n; i++)
        y[i] += alpha * x[i];
}

#endif /* __x86_64__ */
