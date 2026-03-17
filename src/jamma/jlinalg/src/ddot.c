/**
 * ddot.c — Double-precision inner product (BLAS Level 1 ddot).
 *
 * Implements:
 *   jlinalg_ddot_generic — portable scalar with 4x unroll for unit strides
 *   jlinalg_ddot_avx2    — x86_64 AVX2 with 4-accumulator FMA unroll (x86_64 only)
 *
 * The AVX2 path processes 16 doubles per iteration using four __m256d
 * accumulators to hide FMA latency (throughput-bound at ~4 GFLOP/s/GHz for
 * DRAM-resident vectors; higher for cache-resident data).
 * Strided inputs fall back to the generic scalar path.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  /* npy_intp */
#include "jlinalg.h"

/* ---------------------------------------------------------------------------
 * Generic scalar implementation
 * ---------------------------------------------------------------------------
 */
double jlinalg_ddot_generic(
        npy_intp n,
        const double *x, int incx,
        const double *y, int incy)
{
    if (n <= 0)
        return 0.0;

    double sum = 0.0;

    if (incx == 1 && incy == 1) {
        /* Unit-stride fast path: 4x scalar unroll */
        npy_intp i = 0;
        npy_intp n4 = n - (n % 4);
        for (; i < n4; i += 4) {
            sum += x[i]   * y[i];
            sum += x[i+1] * y[i+1];
            sum += x[i+2] * y[i+2];
            sum += x[i+3] * y[i+3];
        }
        for (; i < n; i++)
            sum += x[i] * y[i];
    } else {
        /* Strided slow path */
        for (npy_intp i = 0; i < n; i++)
            sum += x[i * incx] * y[i * incy];
    }

    return sum;
}

/* ---------------------------------------------------------------------------
 * AVX2 implementation (x86_64 only)
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__)

#include <immintrin.h>

double jlinalg_ddot_avx2(
        npy_intp n,
        const double *x, int incx,
        const double *y, int incy)
{
    if (n <= 0)
        return 0.0;

    /* Strided inputs: fall back to generic (no benefit from SIMD) */
    if (incx != 1 || incy != 1)
        return jlinalg_ddot_generic(n, x, incx, y, incy);

    /* Unit-stride fast path: 4 accumulators x 4 doubles = 16 per iteration */
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    npy_intp i = 0;
    npy_intp n16 = n - (n % 16);
    for (; i < n16; i += 16) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i),
                               _mm256_loadu_pd(y + i), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4),
                               _mm256_loadu_pd(y + i + 4), acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 8),
                               _mm256_loadu_pd(y + i + 8), acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 12),
                               _mm256_loadu_pd(y + i + 12), acc3);
    }

    /* Combine accumulators: (acc0+acc1) + (acc2+acc3) */
    __m256d sum256 = _mm256_add_pd(_mm256_add_pd(acc0, acc1),
                                   _mm256_add_pd(acc2, acc3));

    /* Horizontal reduction: 256->128 (lo+hi), then 128-bit hadd to scalar */
    __m128d lo = _mm256_castpd256_pd128(sum256);
    __m128d hi = _mm256_extractf128_pd(sum256, 1);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    double sum = _mm_cvtsd_f64(s);

    /* Scalar tail */
    for (; i < n; i++)
        sum += x[i] * y[i];

    return sum;
}

#endif /* __x86_64__ */
