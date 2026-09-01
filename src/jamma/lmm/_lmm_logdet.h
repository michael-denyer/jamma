/*
 * _lmm_logdet.h — logdet(H) = sum_i log(lambda * ev_i + 1) as a product of
 * mantissas with an exact integer exponent.
 *
 * Every REML and MLE evaluation needs this sum over all n_samples
 * eigenvalues, and the golden-section refinement evaluates it about 20 times
 * per SNP. A per-element log() dominated that loop: Apple clang does not
 * vectorize log, and gcc only does under fast-math, which the build forbids.
 * Multiplying instead costs one multiply and a few integer ops per element
 * and calls log() once at the end.
 *
 * Each v = lambda * ev + 1 splits into v = m * 2^e with m in [0.5, 1) by bit
 * manipulation. The mantissas multiply, the exponents add exactly as
 * integers, and log(v_1 ... v_n) = log(prod m) + (sum e) * ln 2. The running
 * product is renormalised every 16 elements per lane, so it stays in
 * (2^-16, 1] and cannot underflow. Four independent lanes keep the multiply
 * chains from serialising.
 *
 * Invariant: every v is finite and >= 1. Eigenvalues are >= 0 (eigen.py
 * zeroes anything below its threshold; validate_eigenvalues() rejects
 * non-finite values at the workspace boundary) and lambda is > 0, so
 * lambda * ev + 1 >= 1. The bit split is correct only for positive normal
 * doubles; there is no sign, zero, subnormal or NaN handling, and the
 * invariant means none is needed.
 *
 * Measured against the per-element sum over eigenvalues spread 0..55 and
 * lambda in 1e-5..1e5 with n = 1410: max abs diff 3.6e-12 on values up to
 * 1.6e4, max relative diff 2.1e-14, 6.9x faster.
 *
 * Header-only and static inline: the callers are the optimizer's inner
 * kernels, which _lmm_stats.h's once-per-SNP charter excludes. Pure C, no
 * CPython, so it needs none of _lmm_support.h's import_array() handling.
 */

#ifndef JAMMA_LMM_LOGDET_H
#define JAMMA_LMM_LOGDET_H

#include <math.h>
#include <stdint.h>

/* ln 2 spelled out rather than M_LN2: like M_PI it is not C11, and the
 * kernel units include no Python.h to switch it on under glibc's -std=c11. */
#define LOGDET_LN2 0.693147180559945309417232121458

/* v = m * 2^e with m in [0.5, 1). Valid for positive normal doubles only. */
static inline double logdet_frexp_bits(double v, int64_t *e)
{
    union { double d; uint64_t u; } x = { v };
    *e = (int64_t)((x.u >> 52) & 0x7ff) - 1022;
    x.u = (x.u & 0x800fffffffffffffULL) | ((uint64_t)1022 << 52);
    return x.d;
}

static inline double logdet_h_lambda(const double *restrict eigenvalues,
                                     int n_samples, double lambda)
{
    double p0 = 1.0, p1 = 1.0, p2 = 1.0, p3 = 1.0;
    int64_t e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    int64_t a, b, c, d;
    int i = 0;
    for (; i + 4 <= n_samples; i += 4) {
        p0 *= logdet_frexp_bits(lambda * eigenvalues[i] + 1.0, &a);
        p1 *= logdet_frexp_bits(lambda * eigenvalues[i + 1] + 1.0, &b);
        p2 *= logdet_frexp_bits(lambda * eigenvalues[i + 2] + 1.0, &c);
        p3 *= logdet_frexp_bits(lambda * eigenvalues[i + 3] + 1.0, &d);
        e0 += a; e1 += b; e2 += c; e3 += d;
        if ((i & 60) == 60) {
            p0 = logdet_frexp_bits(p0, &a); e0 += a;
            p1 = logdet_frexp_bits(p1, &b); e1 += b;
            p2 = logdet_frexp_bits(p2, &c); e2 += c;
            p3 = logdet_frexp_bits(p3, &d); e3 += d;
        }
    }
    for (; i < n_samples; i++) {
        p0 *= logdet_frexp_bits(lambda * eigenvalues[i] + 1.0, &a);
        e0 += a;
    }
    int64_t e = e0 + e1 + e2 + e3;
    p0 = logdet_frexp_bits(p0, &a); e += a;
    p1 = logdet_frexp_bits(p1, &b); e += b;
    p2 = logdet_frexp_bits(p2, &c); e += c;
    p3 = logdet_frexp_bits(p3, &d); e += d;
    double p = logdet_frexp_bits(p0 * p1, &a); e += a;
    p = logdet_frexp_bits(p * p2, &b); e += b;
    p *= p3;
    return log(p) + (double)e * LOGDET_LN2;
}

#endif /* JAMMA_LMM_LOGDET_H */
