/**
 * dscal.c — Double-precision x *= alpha (BLAS Level 1 dscal).
 *
 * Implements:
 *   jblas_dscal_generic — portable scalar with alpha=0/1 special cases
 *   jblas_dscal_avx2    — x86_64 AVX2 with broadcast alpha and vectorised multiply
 *
 * Special cases:
 *   alpha == 0.0  → zero the vector via memset (sets all elements to +0.0,
 *                    matching reference BLAS; note: differs from NumPy's
 *                    x *= 0.0 which produces NaN per IEEE 754: NaN*0=NaN, Inf*0=NaN)
 *   alpha == 1.0  → no-op (return immediately)
 *
 * Both variants short-circuit on n <= 0.  AVX2 processes 16 doubles per
 * iteration; strided inputs fall back to generic.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  /* npy_intp */
#include "jblas.h"
#include <string.h>  /* memset */

/* ---------------------------------------------------------------------------
 * Generic scalar implementation
 * ---------------------------------------------------------------------------
 */
void jblas_dscal_generic(
        npy_intp n,
        double alpha,
        double *x, int incx)
{
    if (n <= 0)
        return;

    if (alpha == 0.0) {
        if (incx == 1) {
            memset(x, 0, (size_t)n * sizeof(double));
        } else {
            for (npy_intp i = 0; i < n; i++)
                x[i * incx] = 0.0;
        }
        return;
    }

    if (alpha == 1.0)
        return;

    if (incx == 1) {
        /* Unit-stride fast path: 4x scalar unroll */
        npy_intp i = 0;
        npy_intp n4 = n - (n % 4);
        for (; i < n4; i += 4) {
            x[i]   *= alpha;
            x[i+1] *= alpha;
            x[i+2] *= alpha;
            x[i+3] *= alpha;
        }
        for (; i < n; i++)
            x[i] *= alpha;
    } else {
        /* Strided slow path */
        for (npy_intp i = 0; i < n; i++)
            x[i * incx] *= alpha;
    }
}

/* ---------------------------------------------------------------------------
 * AVX2 implementation (x86_64 only)
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__)

#include <immintrin.h>

void jblas_dscal_avx2(
        npy_intp n,
        double alpha,
        double *x, int incx)
{
    if (n <= 0)
        return;

    /* Strided inputs or special cases: fall back to generic */
    if (incx != 1) {
        jblas_dscal_generic(n, alpha, x, incx);
        return;
    }

    if (alpha == 0.0) {
        memset(x, 0, (size_t)n * sizeof(double));
        return;
    }

    if (alpha == 1.0)
        return;

    /* Broadcast alpha into all four lanes */
    __m256d valpha = _mm256_broadcast_sd(&alpha);

    npy_intp i = 0;
    npy_intp n16 = n - (n % 16);
    for (; i < n16; i += 16) {
        _mm256_storeu_pd(x + i,      _mm256_mul_pd(valpha, _mm256_loadu_pd(x + i)));
        _mm256_storeu_pd(x + i + 4,  _mm256_mul_pd(valpha, _mm256_loadu_pd(x + i + 4)));
        _mm256_storeu_pd(x + i + 8,  _mm256_mul_pd(valpha, _mm256_loadu_pd(x + i + 8)));
        _mm256_storeu_pd(x + i + 12, _mm256_mul_pd(valpha, _mm256_loadu_pd(x + i + 12)));
    }

    /* Scalar tail */
    for (; i < n; i++)
        x[i] *= alpha;
}

#endif /* __x86_64__ */
