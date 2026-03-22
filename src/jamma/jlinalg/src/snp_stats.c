/**
 * snp_stats.c — Single-pass per-SNP statistics (mean, variance, miss, HWE).
 *
 * Computes mean, population variance, missing count, and optionally HWE
 * genotype counts (n_aa, n_ab, n_bb) per SNP column in a single pass over
 * the data.  Replaces the multi-pass Python/NumPy pattern (separate isnan,
 * nanmean, nanvar, and genotype counting calls).
 *
 * Two entry points: snp_stats_chunk_f32 (float input) and snp_stats_chunk_f64
 * (double input).  Both accumulate in double precision internally.
 *
 * Data layout: (n_samples, n_snps_chunk) C-contiguous row-major.
 * Element (i, j) is at data[i * n_snps_chunk + j].
 */

#include "jlinalg.h"

#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Macro-based implementation to avoid code duplication for f32/f64 paths.
 *
 * SUFFIX:  f32 or f64
 * DTYPE:   float or double
 *
 * Uses C99 isnan() directly (type-generic, works for both float and double).
 */
#define SNP_STATS_IMPL(SUFFIX, DTYPE)                                         \
void snp_stats_chunk_##SUFFIX(                                                \
    const DTYPE *data,                                                        \
    npy_intp n_samples,                                                       \
    npy_intp n_snps_chunk,                                                    \
    double *means,                                                            \
    npy_intp *miss_counts,                                                    \
    double *variances,                                                        \
    int64_t *n_aa,                                                            \
    int64_t *n_ab,                                                            \
    int64_t *n_bb,                                                            \
    int compute_hwe                                                           \
)                                                                             \
{                                                                             \
    _Pragma("omp parallel for schedule(static) if(n_snps_chunk > 256)")       \
    for (npy_intp j = 0; j < n_snps_chunk; j++) {                            \
        double sum = 0.0;                                                     \
        double sum_sq = 0.0;                                                  \
        npy_intp n_miss = 0;                                                  \
        npy_intp n_valid = 0;                                                 \
        int64_t cnt_aa = 0, cnt_ab = 0, cnt_bb = 0;                          \
                                                                              \
        for (npy_intp i = 0; i < n_samples; i++) {                           \
            DTYPE val = data[i * n_snps_chunk + j];                           \
            if (isnan(val)) {                                                 \
                n_miss++;                                                     \
                continue;                                                     \
            }                                                                 \
            double dval = (double)val;                                        \
            sum += dval;                                                      \
            sum_sq += dval * dval;                                            \
            n_valid++;                                                        \
                                                                              \
            if (compute_hwe) {                                                \
                if (val == (DTYPE)0.0) cnt_aa++;                              \
                else if (val == (DTYPE)1.0) cnt_ab++;                        \
                else if (val == (DTYPE)2.0) cnt_bb++;                        \
            }                                                                 \
        }                                                                     \
                                                                              \
        miss_counts[j] = n_miss;                                              \
                                                                              \
        if (n_valid > 0) {                                                    \
            double mean = sum / (double)n_valid;                              \
            double var = sum_sq / (double)n_valid - mean * mean;              \
            if (var < 0.0) var = 0.0;  /* E[X^2]-E[X]^2 cancellation guard */ \
            means[j] = mean;                                                  \
            variances[j] = var;                                               \
        } else {                                                              \
            means[j] = 0.0;       /* all-missing: return 0.0 instead of NaN */                 \
            variances[j] = 0.0;                                               \
        }                                                                     \
                                                                              \
        if (compute_hwe) {                                                    \
            n_aa[j] = cnt_aa;                                                 \
            n_ab[j] = cnt_ab;                                                 \
            n_bb[j] = cnt_bb;                                                 \
        }                                                                     \
    }                                                                         \
}

/* Instantiate for float (f32) and double (f64) */
SNP_STATS_IMPL(f32, float)
SNP_STATS_IMPL(f64, double)
