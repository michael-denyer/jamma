/**
 * dnrm2.c — Double-precision Euclidean norm (BLAS Level 1 dnrm2).
 *
 * Implements the Blue (1978) three-accumulator algorithm for overflow and
 * underflow protection.  The naïve sqrt(ddot(x,x)) overflows when any
 * element exceeds sqrt(DBL_MAX) ≈ 1.34e+154 and underflows to zero when
 * all elements are below 2^-511 ≈ 1.49e-154 (Blue's underflow threshold).
 *
 * Blue's algorithm maintains three partial sums (small, medium, big) and
 * combines them in a single pass without intermediate overflow.
 *
 * Implements:
 *   jblas_dnrm2_generic — Blue algorithm, fully portable
 *   jblas_dnrm2_avx2    — dispatches to generic (SIMD dnrm2 not yet optimised)
 *
 * Reference: Blue, J.L. (1978) "A portable Fortran program to find the
 * Euclidean norm of a vector", ACM TOMS 4(1):15-23.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  /* npy_intp */
#include "jblas.h"
#include <math.h>
#include <float.h>

/* ---------------------------------------------------------------------------
 * Blue algorithm bounds (pre-computed constants)
 *
 * sml_bound: scale factor for small accumulators (prevents underflow)
 * big_bound: scale factor for large accumulators (prevents overflow)
 *
 *   sml_bound = ibeta^floor((iemin-1)/2) = 2^-511  ≈ 1.4917e-154
 *   big_bound = sqrt(DBL_MAX)                      ≈ 1.3408e+154
 * ---------------------------------------------------------------------------
 */
static const double _DNRM2_SML_BOUND = 1.4916681462400413e-154;  /* 2^-511, Blue's underflow threshold */
static const double _DNRM2_BIG_BOUND = 1.3407807929942596e+154;  /* sqrt(DBL_MAX) */

/* ---------------------------------------------------------------------------
 * Generic scalar implementation (Blue algorithm)
 * ---------------------------------------------------------------------------
 */
double jblas_dnrm2_generic(
        npy_intp n,
        const double *x, int incx)
{
    if (n <= 0)
        return 0.0;
    if (n == 1)
        return fabs(x[0]);

    double asml = 0.0;  /* sum of (x/sml_bound)^2 for |x| < sml_bound */
    double amed = 0.0;  /* sum of x^2 for sml_bound <= |x| <= big_bound */
    double abig = 0.0;  /* sum of (x/big_bound)^2 for |x| > big_bound */
    npy_intp n_sml = 0, n_med = 0, n_big = 0;

    for (npy_intp i = 0; i < n; i++) {
        double ax = fabs(x[i * incx]);
        if (ax == 0.0)
            continue;
        if (ax < _DNRM2_SML_BOUND) {
            double t = ax / _DNRM2_SML_BOUND;
            asml += t * t;
            n_sml++;
        } else if (ax > _DNRM2_BIG_BOUND) {
            double t = ax / _DNRM2_BIG_BOUND;
            abig += t * t;
            n_big++;
        } else {
            amed += ax * ax;
            n_med++;
        }
    }

    /* Combine accumulators in order of magnitude */
    if (n_big > 0) {
        if (n_med > 0) {
            /* Rescale medium into big scale */
            double t = sqrt(amed) / _DNRM2_BIG_BOUND;
            abig += t * t;
        }
        return _DNRM2_BIG_BOUND * sqrt(abig);
    }

    if (n_sml > 0) {
        if (n_med > 0) {
            /* Combine medium (exact) with rescaled small */
            double scale = _DNRM2_SML_BOUND * sqrt(asml);
            amed += scale * scale;
        } else {
            /* Only small accumulators */
            return _DNRM2_SML_BOUND * sqrt(asml);
        }
    }

    return sqrt(amed);
}

/* ---------------------------------------------------------------------------
 * AVX2 stub (x86_64 only) — dispatches to generic
 *
 * A SIMD dnrm2 with Blue algorithm is complex to implement correctly and
 * dnrm2 is not on the hot path (dgemm dominates at scale).  The dispatch
 * pointer points here but the implementation falls through to generic.
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__)

double jblas_dnrm2_avx2(
        npy_intp n,
        const double *x, int incx)
{
    return jblas_dnrm2_generic(n, x, incx);
}

#endif /* __x86_64__ */
